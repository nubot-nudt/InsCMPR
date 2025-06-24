from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import faiss
from dataset import *

from tensorboardX import SummaryWriter
import numpy as np
import os
from models.model import *
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch.autograd as autograd
from configs.config_setting import setting_config as config
from utils import *

rank = int(os.environ.get('RANK', 0))
world_size = torch.cuda.device_count()
torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
# dist.barrier()
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

print("PID: ", os.getpid())
writer = SummaryWriter(config.log_dir)
# device = torch.device("cuda")

model = Backbone(nmf=config.nmf, mlp=config.mlp)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
print("resume =", "\033[1;32m %s \033[0m" % config.resume)


    # from collections import OrderedDict
    # state_dict = torch.load(config.pth,map_location=None)
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     name = k.replace('module.','')
    #     new_state_dict[name]=v
    # model.load_state_dict(new_state_dict['state_dict'])
    # optimizer.load_state_dict(state_dict['optimizer'])

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=0,
                                                  find_unused_parameters=True)
if config.resume:
    checkpoint = torch.load(config.pth,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


criterion = TripletLossSimple(config.margin).to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
dataset = NCLTData(resize_shape=config.resize_shape)
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size)
print("resize shape =", config.resize_shape, "margin =", config.margin, "lr =", config.lr, "mid margin = ",
      config.mid_margin)
dataloader = DataLoader(dataset, batch_size=config.Batch_Size, shuffle=False, num_workers=config.num_works,
                        collate_fn=None, pin_memory=True, sampler=train_sampler)


test_set = ["2012-02-05"]
best_recall = [0.94]
max_recall = [0 for i in test_set]

for epoch in range(config.epoches):
    # print("epoch:", epoch)
    # print(scheduler.get_last_lr())
    loss_batch = 0
    loss_mid_batch = 0
    dataloader.sampler.set_epoch(epoch)
    pbar = tqdm(total=len(dataloader))
    print(epoch % 10, end="")
    print(" ", end="")
    for index, (query, pos, neg, sam_q, sam_pos, sam_neg) in enumerate(dataloader):

        B, n_neg, C, H, W = neg.shape
        neg = torch.flatten(neg, start_dim=0, end_dim=1)
        sam_neg = torch.flatten(sam_neg, start_dim=0, end_dim=1)
        model.train()
        # input = torch.cat([query, pos, neg])
        input = torch.cat([query, pos, neg, sam_q, sam_pos, sam_neg])
        input = input.to(device)
        mid, output = model(input)
        vladQ, vladP, vladN = torch.split(output, [B, B, B * n_neg])
        midQ, midP, midN = torch.split(mid, [B, B, B * n_neg])
        optimizer.zero_grad()

        loss = 0
        loss_mid = 0
        for i in range(config.Batch_Size):
            max_loss = 0
            for n in range(n_neg):
                negIx = i * n_neg + n
                loss_tmp = criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])
                if loss_tmp >= max_loss:
                    max_loss = loss_tmp
                loss_mid_tmp = F.relu(torch.mean(torch.abs(midQ[i:i + 1] - midP[i:i + 1])) - torch.mean(
                    torch.abs(midQ[i:i + 1] - midN[i:i + 1])) + config.mid_margin)
                # loss_mid_tmp = F.relu(torch.mean(torch.abs(midQ[i:i+1] - midP[i:i+1])))
            loss += max_loss
            loss_mid += loss_mid_tmp
        loss /= config.Batch_Size
        loss_mid /= config.Batch_Size
        loss += loss_mid
        # print(loss)
        loss.backward()
        loss_batch += loss.item()
        loss_mid_batch += loss_mid.item()
        writer.add_scalar("train loss", loss_batch / len(dataloader), epoch)
        optimizer.step()
        pbar.update(1)
        pbar.set_postfix({"Loss": loss.item(), "Loss Mid": loss_mid.item()})
    pbar.close()
    optimizer.zero_grad()
    scheduler.step()

    if epoch >= 1 and epoch % 10 != 0:
        continue
    print("")

    deslen = 16384*2
    # if config.nmf:
    #     deslen += config.K*64
    # if config.mlp:
    #     deslen += config.K*64
    for nnni in range(len(test_set)):
        nnn = test_set[nnni]
        query_path = config.dataset_toot + nnn + "/image/"+"Cam.txt"
        database_path = config.dataset_toot  + nnn + "/lidar/depth_Ours.txt"

        with open(query_path, 'r') as f:
            query = f.readlines()
            for i in range(len(query)):
                query[i] = query[i].strip()
        with open(database_path, 'r') as f:
            database = f.readlines()
            for i in range(len(database)):
                database[i] = database[i].strip()
                # database[i] = database[i][:48]+database[i][49:57]+"npy"

        # 用训练提取LiDAR的描述子添加到des_list，也就是database数据库
        des_list = np.zeros((len(database), deslen))
        for i in range(len(database)):
            head, tail = os.path.split(database[i])
            new_head = head.replace('depth_Ours', 'depth_SAM')
            # new_tail = tail.replace('tiff', 'png')
            sam_pos = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
            sam_pos = cv2.resize(sam_pos, config.resize_shape)
            sam_pos = input_transform()(sam_pos).cuda()
            sam_pos = torch.unsqueeze(sam_pos, 0)

            # heada, taila = os.path.split(database[i])
            # new_head1 = heada.replace('lidar', 'lidar_D50t')
            # # new_tail1 = taila.replace('tiff', 'png')
            # pos = np.array(Image.open(os.path.join(new_head1, taila))).astype(np.float32)
            pos = np.array(Image.open(database[i])).astype(np.float32)
            # pos = np.load(database[i])
            pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
            # pos[pos<0] = 0
            # pos = pos*255
            pos = cv2.resize(pos, config.resize_shape)
            lidar = input_transform()(pos).cuda()
            lidar = torch.unsqueeze(lidar, 0)
            lidar = torch.cat([lidar, sam_pos])
            model.eval()
            _, lidar = model(lidar)
            des_list[(i), :] = lidar[0, :].cpu().detach().numpy()
        des_list = des_list.astype('float32')
        quantizer = faiss.IndexFlatL2(deslen)
        faiss_index = faiss.IndexIVFFlat(quantizer, deslen, 1, faiss.METRIC_L2)
        assert not faiss_index.is_trained
        faiss_index.train(des_list)
        assert faiss_index.is_trained
        faiss_index.add(des_list)
        recog_list = []
        # print("database number:", len(database), "query number:", len(query))

        # 用训练提取当前帧image的描述子，也就是查询描述子
        for i in range(len(query)):
            head, tail = os.path.split(query[i])
            new_head = head.replace('Cam5', 'Cam5_SAM')
            sam_q = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
            sam_q = cv2.resize(sam_q, config.resize_shape)
            sam_q = input_transform()(sam_q).cuda()
            sam_q = torch.unsqueeze(sam_q, 0)

            q = np.array(Image.open(query[i])).astype(np.float32)
            # q = np.load(query[i]).transpose(1,2,0).astype(np.float32)
            q = cv2.resize(q, config.resize_shape)
            q = input_transform()(q).cuda()  # [3, 55, 400]
            rgb = torch.unsqueeze(q, 0)
            rgb = torch.cat([rgb, sam_q])
            model.eval()
            _, rgb = model(rgb)
            des_list_current = rgb[0, :].cpu().detach().numpy()
            # D是距离， I是索引
            D, I = faiss_index.search(des_list_current.reshape(1, -1), 1)  # top 1
            for j in range(D.shape[1]):
                one_recog = np.zeros((1, 3))
                one_recog[:, 0] = i
                one_recog[:, 1] = I[:, j]
                one_recog[:, 2] = D[:, j]
                recog_list.append(one_recog)
            # print("query:"+query[i] + "---->" + "database:" + database[I[:, j][0]] + "  " + str(D[:, j]))
        t_error = []
        f = open("/media/nubot/data/NCLT_data/data/ground_truth/" + "groundtruth_" +nnn+ ".csv", 'r')
        poses = f.readlines()
        for j in range(len(poses)):
            poses[j] = poses[j].strip().split(",")
        for i in range(len(recog_list)):
            pose_query = poses[int(recog_list[i][0][0])]
            pose_database = poses[int(recog_list[i][0][1])]
            t_error_temp = 0
            for j in [1, 2]:
                t_error_temp += np.square(float(pose_query[j]) - float(pose_database[j]))
            t_error_temp = np.sqrt(t_error_temp)
            t_error.append(t_error_temp)
        ratio = []
        for dist in [0.5, 10.0]:
            ratio.append(np.sum(np.array(t_error) < dist) / len(t_error))

        max_recall[nnni] = max(max_recall[nnni], ratio[-1])

        bestlabel = " "
        if ratio[-1] == max_recall[nnni]:
            bestlabel = "+"
        if ratio[-1] > best_recall[nnni]:
            print(epoch, "\t| %.4f" % (loss_batch / len(dataloader)), " | %.4f" % (loss_mid_batch / len(dataloader)),
                  " |", query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |10m %.4f" % ratio[-1],
                  " |best %.4f" % max_recall[nnni], " *" + bestlabel, sep='')
        else:
            print(epoch, "\t| %.4f" % (loss_batch / len(dataloader)), " | %.4f" % (loss_mid_batch / len(dataloader)),
                  " |", query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |10m %.4f" % ratio[-1],
                  " |best %.4f" % max_recall[nnni], "  " + bestlabel, sep='')
        writer.add_scalar('0.5m recall', ratio[0], global_step=epoch)
        writer.add_scalar('10m recall', ratio[-1], global_step=epoch)

    torch.save({'epoch': i, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               config.log_dir + str(epoch) + '_' + config.spth)
writer.close()

