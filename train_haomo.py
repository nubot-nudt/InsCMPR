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
torch.distributed.init_process_group(backend="nccl",world_size=world_size,rank=rank)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print("PID: ", os.getpid())
writer = SummaryWriter(config.log_dir)
model = Backbone(nmf=config.nmf, mlp=config.mlp)
model = model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=0,find_unused_parameters=True)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
print("resume =", "\033[1;32m %s \033[0m" % config.resume)

if config.resume:
    checkpoint = torch.load(config.pth)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

criterion = TripletLossSimple(config.margin).to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
dataset = HaomoSAM(resize_shape=config.resize_shape)
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size)
print("resize shape =", config.resize_shape, "margin =", config.margin, "lr =", config.lr, "mid margin = ",
      config.mid_margin)
dataloader = DataLoader(dataset, batch_size=config.Batch_Size, shuffle=False, num_workers=config.num_works, collate_fn=None, pin_memory=True,sampler=train_sampler)

for epoch in range(config.epoches):
    loss_batch = 0
    loss_mid_batch = 0
    pbar = tqdm(total=len(dataloader))
    print(epoch % 10, end="")
    print(" ", end="")
    # for index, (query, pos, neg) in enumerate(dataloader):
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


    if epoch >= 1 and epoch % 20 != 0:
        continue
    print("")
    # print("*"*100)
    # print("loss_batch:", loss_batch/len(dataloader))
    # test *****************************************************
    # 加载test pair

    deslen = 16384*2

    query_path = "/media/nubot/data/HAOMO/data/test_s/img_test.txt"
    database_path = "/media/nubot/data/HAOMO/data/test_s/lidar_test.txt"

    with open(query_path, 'r') as f:
        query = f.readlines()
        for i in range(len(query)):
            query[i] = query[i].strip()

    with open(database_path, 'r') as f:
        database = f.readlines()
        for i in range(len(database)):
            database[i] = database[i].strip()
    #
    #  # 用训练提取LiDAR的描述子添加到des_list，也就是database数据库
    des_list = np.zeros((len(database), deslen))
    for i in range(len(database)):
        head, tail = os.path.split(database[i])
        new_head = head.replace('lidar_test', 'lidar_SAM_test')
        new_tail = tail.replace('tiff', 'png')

        sam_pos = np.array(Image.open(os.path.join(new_head, new_tail)).convert('RGB')).astype(np.float32)
        sam_pos = cv2.resize(sam_pos, config.resize_shape)
        sam_pos = input_transform()(sam_pos).cuda()
        sam_pos = torch.unsqueeze(sam_pos, 0)

        pos = np.array(Image.open(database[i])).astype(np.float32)
        # pos = np.load(database[i])
        pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
        # pos[pos<0] = 0
        pos = pos * 255
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

    # 用训练提取当前帧image的描述子，也就是查询描述子
    for i in range(len(query)):
        head, tail = os.path.split(query[i])
        new_head = head.replace('img_test', 'img_SAM_test')
        sam_q = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
        sam_q = input_transform()(sam_q).cuda()
        sam_q = torch.unsqueeze(sam_q, 0)

        q = np.array(Image.open(query[i])).astype(np.float32)
        # sam_q = np.load(query[i]).transpose(1,2,0).astype(np.float32)
        # q = cv2.resize(q, config.resize_shape)
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
            one_recog[:, 0] = i  # 当前查询图像的索引
            one_recog[:, 1] = I[:, j]  # 查询结果最接近的数据库图像的索引
            one_recog[:, 2] = D[:, j]  # 查询图像与最接近的数据库图像之间的距离

            recog_list.append(one_recog)
    t_error = []
    f = open("/media/nubot/data/HAOMO/data/poses/test_s.txt", 'r')
    poses = f.readlines()
    for j in range(len(poses)):
        poses[j] = poses[j].strip().split()
    for i in range(len(recog_list)):
        # print(recog_list[i][0][0], "--->", recog_list[i][0][1])
        pose_query = poses[int(recog_list[i][0][0])]
        pose_database = poses[int(recog_list[i][0][1])]
        t_error_temp = 0
        for j in [4, 8]:
            t_error_temp += np.square(float(pose_query[j]) - float(pose_database[j]))
        t_error_temp = np.sqrt(t_error_temp)
        t_error.append(t_error_temp)
        # print(len(t_error))
    ratio = []
    for dist in [0.5, 5, 10.0]:
        ratio.append(np.sum(np.array(t_error) < dist) / len(t_error))
    print(epoch, "|","haomo", " |0.5m %.4f" % ratio[0], " |5m %.4f" % ratio[1], " |10m %.4f" % ratio[-1],
              sep='')
    writer.add_scalar('10m recall', ratio[-1], global_step=epoch)
    torch.save({'epoch': i, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               config.log_dir + str(epoch) + '_' + config.spth)
writer.close()
