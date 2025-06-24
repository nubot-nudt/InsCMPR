from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import faiss
import math
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

print("PID: ", os.getpid())

device = torch.device("cuda")

#数据加载
dataset = KITTI(resize_shape=config.resize_shape)
dataloader = DataLoader(dataset, batch_size=config.Batch_Size, shuffle=False, num_workers=config.num_works, collate_fn=None, pin_memory=False)
#模型加载
model = Backbone(nmf=config.nmf, mlp=config.mlp)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
print("resume =", "\033[1;32m %s \033[0m" % config.resume)

checkpoint = torch.load(config.pth, map_location=device)
state_dict = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # 去除 'module' 前缀
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
optimizer.load_state_dict(checkpoint['optimizer'])

test_set = [0,2,5,6,8]
best_recall = [0.94,0.68,0.90,0.90,0.86]


# test_set = [0]
# best_recall = [0.96]
max_recall = [0 for i in test_set]
deslen = 16384*2
# if config.mlp:
#     deslen += config.K*64
for nnni in range(len(test_set)):
    nnn = test_set[nnni]
    query_path = config.dataset_toot + "/sequences/" + f"{nnn:02d}" + "/image_2.txt"
    database_path = config.dataset_toot + "/sequences/" + f"{nnn:02d}" + "/lidar.txt"

    with open(query_path, 'r') as f:
        query = f.readlines()
        for i in range(len(query)):
            query[i] = query[i].strip()

    #################
    num_queries = len(query)
    top_k = max(math.ceil(num_queries * 0.01), 1)  # 至少为1
    print("top_k Num:",top_k)
    #################
    with open(database_path, 'r') as f:
        database = f.readlines()
        for i in range(len(database)):
            database[i] = database[i].strip()

    # 用训练提取LiDAR的描述子添加到des_list，也就是database数据库
    des_list = np.zeros((len(database), deslen))
    for i in range(len(database)):
        head, tail = os.path.split(database[i])
        new_head = head.replace('lidar', 'lidarSAM_t')
        # new_tail = tail.replace('tiff', 'png')
        sam_pos = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
        sam_pos = input_transform()(sam_pos).cuda()
        sam_pos = torch.unsqueeze(sam_pos, 0)

        heada, taila = os.path.split(database[i])
        new_head1 = heada.replace('lidar', 'lidar_D50t')
        pos = np.array(Image.open(os.path.join(new_head1, taila))).astype(np.float32)
        # pos = np.array(Image.open(database[i])).astype(np.float32)
        # pos = np.load(database[i])
        pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
        # pos[pos<0] = 0
        # pos = pos * 255
        # pos = cv2.resize(pos, config.resize_shape)
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
    recog_list_recall1_percent = []
    recall1_percent = []
    f = open(config.dataset_toot + "/poses/" + query[0][-21:-19] + ".txt", 'r')
    poses = f.readlines()
    for j in range(len(poses)):
        poses[j] = poses[j].strip().split()
    # 计算 recall@1%
    correct_matches_at_1_percent = 0
    # 用训练提取当前帧image的描述子，也就是查询描述子
    for i in range(len(query)):
        head, tail = os.path.split(query[i])
        new_head = head.replace('image_2', 'imgSAM')
        sam_q = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
        sam_q = input_transform()(sam_q).cuda()
        sam_q = torch.unsqueeze(sam_q, 0)

        q = np.array(Image.open(query[i])).astype(np.float32)[165:]
        # sam_q = np.load(query[i]).transpose(1,2,0).astype(np.float32)
        q = cv2.resize(q, config.resize_shape)
        q = input_transform()(q).cuda()  # [3, 55, 400]
        rgb = torch.unsqueeze(q, 0)
        rgb = torch.cat([rgb, sam_q])
        model.eval()
        _, rgb = model(rgb)
        des_list_current = rgb[0, :].cpu().detach().numpy()
        D, I = faiss_index.search(des_list_current.reshape(1, -1), 1)  # top 1

        for j in range(D.shape[1]):
            one_recog = np.zeros((1, 3))
            one_recog[:, 0] = i  # 当前查询图像的索引
            one_recog[:, 1] = I[:, j]  # 查询结果最接近的数据库图像的索引
            one_recog[:, 2] = D[:, j]  # 查询图像与最接近的数据库图像之间的距离
            recog_list.append(one_recog)

        # print("query:"+query[i] + "---->" + "database:" + database[I[:, j][0]] + "  " + str(D[:, j]))
        D_recall1_percent, I_recall1_percent = faiss_index.search(des_list_current.reshape(1, -1), top_k)
        recog_list_recall1_percent = []
        for j in range(D_recall1_percent.shape[1]):
            one_recog = np.zeros((1, 3))
            one_recog[:, 0] = i  # 当前查询图像的索引
            one_recog[:, 1] = I_recall1_percent[:, j]  # 查询结果最接近的数据库图像的索引
            one_recog[:, 2] = D_recall1_percent[:, j]  # 查询图像与最接近的数据库图像之间的距离
            recog_list_recall1_percent.append(one_recog)
        recall1_percent.append(recog_list_recall1_percent)
    for i in range(len(recall1_percent)):
        recall = recall1_percent[i]
        for j in range(len(recall)):
            pose_query = poses[int(query[int(recall[j][0][0])][-10:-4])]
            pose_database = poses[int(database[int(recall[j][0][1])][-11:-5])]
            t_error_temp = 0
            for k in [3, 7]:
                t_error_temp += np.square(float(pose_query[k]) - float(pose_database[k]))
            t_error_temp = np.sqrt(t_error_temp)
            if t_error_temp < 10:
                correct_matches_at_1_percent += 1
                break  # 如果找到了一个正确匹配，则跳出循环

    recall_at_1_percent = correct_matches_at_1_percent / len(query)
    print("recall@1%:", recall_at_1_percent)

    t_error = []
    for i in range(len(recog_list)):
        pose_query = poses[int(query[int(recog_list[i][0][0])][-10:-4])]
        pose_database = poses[int(database[int(recog_list[i][0][1])][-11:-5])]
        t_error_temp = 0
        for j in [3, 7]:
            t_error_temp += np.square(float(pose_query[j]) - float(pose_database[j]))
        t_error_temp = np.sqrt(t_error_temp)
        t_error.append(t_error_temp)
    ratio = []
    for dist in [0.5, 5, 10.0]:
        ratio.append(np.sum(np.array(t_error) < dist) / len(t_error))

    max_recall[nnni] = max(max_recall[nnni], ratio[-1])
    bestlabel = " "
    if ratio[-1] == max_recall[nnni]:
        bestlabel = "+"
    if ratio[-1] > best_recall[nnni]:
        print( " |",query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |5m %.4f" % ratio[1], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni]," *" + bestlabel, sep='')
    else:
        print(" |",query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |5m %.4f" % ratio[1], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni],"  " + bestlabel, sep='')





# # -*-coding: Utf-8 -*-
# # @File : evalue_kitti.py
# # @author: Jiao Shuaifeng
# # @Time：2024/8/19
# # @brief :
# from __future__ import print_function
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'
# from tqdm import tqdm
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from datetime import datetime
# import faiss
# import math
# from dataset import *
#
# from tensorboardX import SummaryWriter
# import numpy as np
# import os
# from models.model import *
# import datetime
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch.autograd as autograd
# from configs.config_setting import setting_config as config
# from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# rank = int(os.environ.get('RANK', 0))
# world_size = torch.cuda.device_count()
# torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
# # dist.barrier()
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
#
#
# print("PID: ", os.getpid())
#
# #创建日志目录
# writer = SummaryWriter(config.log_dir)
#
# model = Backbone(nmf=config.nmf, mlp=config.mlp)
# model = model.to(device)
#
# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=0,
#                                                   find_unused_parameters=True)
#
# optimizer = optim.Adam(model.parameters(), lr=config.lr)
# print("resume =", "\033[1;32m %s \033[0m" % config.resume)
#
# if config.resume:
#     checkpoint = torch.load(config.pth)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#
# criterion = TripletLossSimple(config.margin).to(device)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# dataset = KITTI(resize_shape=config.resize_shape)
# train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size)
# print("resize shape =", config.resize_shape, "margin =", config.margin, "lr =", config.lr, "mid margin = ",
#       config.mid_margin)
# dataloader = DataLoader(dataset, batch_size=config.Batch_Size, shuffle=False, num_workers=config.num_works,
#                         collate_fn=None, pin_memory=True, sampler=train_sampler)
# test_set = [5]
# best_recall = [0.94]
#
#
# # test_set = [0]
# # best_recall = [0.96]
# max_recall = [0 for i in test_set]
# deslen = 16384*2
# # if config.mlp:
# #     deslen += config.K*64
# for nnni in range(len(test_set)):
#     nnn = test_set[nnni]
#     query_path = config.dataset_toot + "/sequences/" + f"{nnn:02d}" + "/image_2.txt"
#     database_path = config.dataset_toot + "/sequences/" + f"{nnn:02d}" + "/lidar.txt"
#
#     with open(query_path, 'r') as f:
#         query = f.readlines()
#         for i in range(len(query)):
#             query[i] = query[i].strip()
#
#     #################
#     num_queries = len(query)
#     top_k = max(math.ceil(num_queries * 0.01), 1)  # 至少为1
#     print("top_k Num:",top_k)
#     #################
#     with open(database_path, 'r') as f:
#         database = f.readlines()
#         for i in range(len(database)):
#             database[i] = database[i].strip()
#
#     # 用训练提取LiDAR的描述子添加到des_list，也就是database数据库
#     des_list = np.zeros((len(database), deslen))
#     for i in range(len(database)):
#         head, tail = os.path.split(database[i])
#         new_head = head.replace('lidar', 'lidarSAM_t')
#         # new_tail = tail.replace('tiff', 'png')
#         sam_pos = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
#         sam_pos = input_transform()(sam_pos).cuda()
#         sam_pos = torch.unsqueeze(sam_pos, 0)
#
#         heada, taila = os.path.split(database[i])
#         new_head1 = heada.replace('lidar', 'lidar_D50t')
#         pos = np.array(Image.open(os.path.join(new_head1, taila))).astype(np.float32)
#         # pos = np.array(Image.open(database[i])).astype(np.float32)
#         # pos = np.load(database[i])
#         pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
#         # pos[pos<0] = 0
#         # pos = pos * 255
#         # pos = cv2.resize(pos, config.resize_shape)
#         lidar = input_transform()(pos).cuda()
#         lidar = torch.unsqueeze(lidar, 0)
#         lidar = torch.cat([lidar, sam_pos])
#         model.eval()
#         _, lidar = model(lidar)
#         des_list[(i), :] = lidar[0, :].cpu().detach().numpy()
#     des_list = des_list.astype('float32')
#     quantizer = faiss.IndexFlatL2(deslen)
#     faiss_index = faiss.IndexIVFFlat(quantizer, deslen, 1, faiss.METRIC_L2)
#     assert not faiss_index.is_trained
#     faiss_index.train(des_list)
#     assert faiss_index.is_trained
#     faiss_index.add(des_list)
#
#     recog_list = []
#     recog_list_recall1_percent = []
#     recall1_percent = []
#     f = open(config.dataset_toot + "/poses/" + query[0][-21:-19] + ".txt", 'r')
#     poses = f.readlines()
#     for j in range(len(poses)):
#         poses[j] = poses[j].strip().split()
#     # 计算 recall@1%
#     correct_matches_at_1_percent = 0
#     # 用训练提取当前帧image的描述子，也就是查询描述子
#     for i in range(len(query)):
#         head, tail = os.path.split(query[i])
#         new_head = head.replace('image_2', 'imgSAM')
#         sam_q = np.array(Image.open(os.path.join(new_head, tail)).convert('RGB')).astype(np.float32)
#         sam_q = input_transform()(sam_q).cuda()
#         sam_q = torch.unsqueeze(sam_q, 0)
#
#         q = np.array(Image.open(query[i])).astype(np.float32)[165:]
#         # sam_q = np.load(query[i]).transpose(1,2,0).astype(np.float32)
#         q = cv2.resize(q, config.resize_shape)
#         q = input_transform()(q).cuda()  # [3, 55, 400]
#         rgb = torch.unsqueeze(q, 0)
#         rgb = torch.cat([rgb, sam_q])
#         model.eval()
#         _, rgb = model(rgb)
#         des_list_current = rgb[0, :].cpu().detach().numpy()
#         # D是距离， I是索引
#         D, I = faiss_index.search(des_list_current.reshape(1, -1), 1)  # top 1
#
#         for j in range(D.shape[1]):
#             one_recog = np.zeros((1, 3))
#             one_recog[:, 0] = i  # 当前查询图像的索引
#             one_recog[:, 1] = I[:, j]  # 查询结果最接近的数据库图像的索引
#             one_recog[:, 2] = D[:, j]  # 查询图像与最接近的数据库图像之间的距离
#             recog_list.append(one_recog)
#
#         # print("query:"+query[i] + "---->" + "database:" + database[I[:, j][0]] + "  " + str(D[:, j]))
#         D_recall1_percent, I_recall1_percent = faiss_index.search(des_list_current.reshape(1, -1), top_k)
#         recog_list_recall1_percent = []
#         for j in range(D_recall1_percent.shape[1]):
#             one_recog = np.zeros((1, 3))
#             one_recog[:, 0] = i  # 当前查询图像的索引
#             one_recog[:, 1] = I_recall1_percent[:, j]  # 查询结果最接近的数据库图像的索引
#             one_recog[:, 2] = D_recall1_percent[:, j]  # 查询图像与最接近的数据库图像之间的距离
#             recog_list_recall1_percent.append(one_recog)
#         recall1_percent.append(recog_list_recall1_percent)
#     for i in range(len(recall1_percent)):
#         recall = recall1_percent[i]
#         for j in range(len(recall)):
#             pose_query = poses[int(query[int(recall[j][0][0])][-10:-4])]
#             pose_database = poses[int(database[int(recall[j][0][1])][-11:-5])]
#             t_error_temp = 0
#             for k in [3, 7]:
#                 t_error_temp += np.square(float(pose_query[k]) - float(pose_database[k]))
#             t_error_temp = np.sqrt(t_error_temp)
#             if t_error_temp < 10:
#                 correct_matches_at_1_percent += 1
#                 break  # 如果找到了一个正确匹配，则跳出循环
#
#     recall_at_1_percent = correct_matches_at_1_percent / len(query)
#     print("recall@1%:", recall_at_1_percent)
#
#     t_error = []
#     for i in range(len(recog_list)):
#         pose_query = poses[int(query[int(recog_list[i][0][0])][-10:-4])]
#         pose_database = poses[int(database[int(recog_list[i][0][1])][-11:-5])]
#         t_error_temp = 0
#         for j in [3, 7]:
#             t_error_temp += np.square(float(pose_query[j]) - float(pose_database[j]))
#         t_error_temp = np.sqrt(t_error_temp)
#         t_error.append(t_error_temp)
#     ratio = []
#     for dist in [0.5, 5, 10.0]:
#         ratio.append(np.sum(np.array(t_error) < dist) / len(t_error))
#
#     max_recall[nnni] = max(max_recall[nnni], ratio[-1])
#     bestlabel = " "
#     if ratio[-1] == max_recall[nnni]:
#         bestlabel = "+"
#     if ratio[-1] > best_recall[nnni]:
#         print( " |",query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |5m %.4f" % ratio[1], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni]," *" + bestlabel, sep='')
#     else:
#         print(" |",query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |5m %.4f" % ratio[1], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni],"  " + bestlabel, sep='')




