# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import h5py
# from models import netvlad
# from models.NMF import *
# from .mamba_vision import MambaVision
# import numpy as np
#
# class Backbone(nn.Module):
#     def __init__(self, nmf=True, mlp=False):
#         super(Backbone, self).__init__()
#
#         self.nmf = nmf
#         self.K = 16
#         self.max_iter = 50
#         print("nmf: ", self.nmf)
#         print("K: ", self.K)
#
#         self.mlp = mlp
#         print("mlp: ", self.mlp)
#
#         encoder = models.resnet34(pretrained=True) # resnet34 capture only features and remove last relu and maxpool
#         layers = list(encoder.children())[:-3]
#         self.encoder = nn.Sequential(*layers)
#         self.relu = nn.ReLU(inplace=True)
#         self.model_path = "weights/mambavision_small_1k.pth.tar"
#         self.mamba_vision1 = MambaVision(
#                         depths=[3, 3, 7, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=96,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2)
#         self.mamba_vision1._load_state_dict(self.model_path)
#
#         self.mamba_vision2 = MambaVision(
#                         depths=[3, 3, 7, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=96,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2)
#         self.mamba_vision2._load_state_dict(self.model_path)
#         # self.mamba_vision1 = MambaVision(
#         #     depths=[1, 3, 8, 4],
#         #     num_heads=[2, 4, 8, 16],
#         #     window_size=[8, 8, 14, 7],
#         #     dim=80,
#         #     in_dim=32,
#         #     mlp_ratio=4,
#         #     resolution=224,
#         #     drop_path_rate=0.2,
#         # )
#         # self.mamba_vision1._load_state_dict(self.model_path)
#         #
#         # self.mamba_vision2 = MambaVision(
#         #     depths=[1, 3, 8, 4],
#         #     num_heads=[2, 4, 8, 16],
#         #     window_size=[8, 8, 14, 7],
#         #     dim=80,
#         #     in_dim=32,
#         #     mlp_ratio=4,
#         #     resolution=224,
#         #     drop_path_rate=0.2,
#         # )
#         # self.mamba_vision2._load_state_dict(self.model_path)
#
#         self.mlp_c = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1)),
#                                  nn.ReLU(),
#                                  nn.Conv2d(128, self.K, kernel_size=(1, 1)),
#                                  nn.ReLU())
#         self.mamba_mlp1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=(1, 1)),
#                                        nn.ReLU())
#         self.mamba_mlp2 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=(1, 1)),
#                                        nn.ReLU())
#         self.net_vlad_cnn1 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
#         self.net_vlad_cnn2 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
#         self.net_vlad_nmf = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
#         self.net_vlad_mlp = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
#
#
#
#     def forward(self, x):
#         tmp =int(x.shape[0]/2)
#         x_img = x[:tmp,:,:,:]
#         x_sam = x[tmp:,:,:,:]
#
#         # depth
#         x_img = self.mamba_vision1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#         x_img = self.mamba_mlp1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#         x_mid = x_img
#         x_img = self.net_vlad_cnn1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#
#
#         # MobileSAM
#         x_sam = self.mamba_vision2(x_sam)
#         x_sam = F.normalize(x_sam, p=2, dim=1)
#         x_sam = self.mamba_mlp2(x_sam)
#         x_sam = F.normalize(x_sam,p=2,dim=1)
#         # x_mid = x_sam
#         x_sam = self.net_vlad_cnn2(x_sam)
#         x_sam = F.normalize(x_sam,p=2,dim=1)
#
#         out = torch.cat((x_img, x_sam), 1)
#
#
#         return x_mid, out
#
# class TripletLossSimple(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletLossSimple, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative):
#
#         pos_dist = torch.sqrt((anchor - positive).pow(2).sum(1))
#         neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
#         loss = F.relu(pos_dist-neg_dist + self.margin)
#         return loss.mean()
#
#
# # 定义一个函数来判断两个图像是否匹配
# def is_correct_match(query_index, database_index, poses, threshold=10.0):
#     pose_query = poses[query_index]
#     pose_database = poses[database_index]
#     distance = 0
#     for j in [3, 7]:  # 根据你的需求选择合适的索引
#         distance += np.square(float(pose_query[j]) - float(pose_database[j]))
#     distance = np.sqrt(distance)
#     return distance < threshold
################################################# base_model ############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import h5py
from models import netvlad
from models.NMF import *
from .mamba_vision import MambaVision
import numpy as np


class Backbone(nn.Module):
    def __init__(self, nmf=True, mlp=False):
        super(Backbone, self).__init__()

        self.nmf = nmf
        self.K = 16
        self.max_iter = 50
        print("nmf: ", self.nmf)
        print("K: ", self.K)

        self.mlp = mlp
        print("mlp: ", self.mlp)

        encoder = models.resnet34(pretrained=True)  # resnet34 capture only features and remove last relu and maxpool
        layers = list(encoder.children())[:-3]
        self.encoder = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        self.model_path = "weights/mambavision_base_1k.pth.tar"
        self.mamba_vision1 = MambaVision(depths=[3, 3, 10, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[8, 8, 14, 7],
                        dim=128,
                        in_dim=64,
                        mlp_ratio=4,
                        resolution=224,
                        drop_path_rate=0.3,
                        layer_scale=1e-5,
                        layer_scale_conv=None)
        self.mamba_vision1._load_state_dict(self.model_path)

        self.mamba_vision2 = MambaVision(depths=[3, 3, 10, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[8, 8, 14, 7],
                        dim=128,
                        in_dim=64,
                        mlp_ratio=4,
                        resolution=224,
                        drop_path_rate=0.3,
                        layer_scale=1e-5,
                        layer_scale_conv=None)
        self.mamba_vision2._load_state_dict(self.model_path)

        self.mlp_c = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, self.K, kernel_size=(1, 1)),
                                   nn.ReLU())
        self.mamba_mlp1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 256, kernel_size=(1, 1)),
                                        nn.ReLU()
                                        )
        self.mamba_mlp2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 256, kernel_size=(1, 1)),
                                        nn.ReLU()
                                        )
        self.net_vlad_cnn1 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
        self.net_vlad_cnn2 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
        self.net_vlad_nmf = netvlad.NetVLAD(num_clusters=64, dim=self.K, vladv2=True)
        self.net_vlad_mlp = netvlad.NetVLAD(num_clusters=64, dim=self.K, vladv2=True)

    def forward(self, x):
        tmp = int(x.shape[0] / 2)
        x_img = x[:tmp, :, :, :]
        x_sam = x[tmp:, :, :, :]

        # depth
        x_img = self.mamba_vision1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        x_img = self.mamba_mlp1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)
        x_mid = x_img
        x_img = self.net_vlad_cnn1(x_img)
        x_img = F.normalize(x_img, p=2, dim=1)

        # MobileSAM
        x_sam = self.mamba_vision2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)
        x_sam = self.mamba_mlp2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)
        # x_mid = x_sam
        x_sam = self.net_vlad_cnn2(x_sam)
        x_sam = F.normalize(x_sam, p=2, dim=1)

        out = torch.cat((x_img, x_sam), 1)

        return x_mid, out


class TripletLossSimple(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLossSimple, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum(1))
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

#################################################  res_mamba_model  ####################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import h5py
# from models import netvlad
# from models.NMF import *
# from .mamba_vision_res import MambaVision
# import numpy as np
#
# class Backbone(nn.Module):
#     def __init__(self, nmf=True, mlp=False):
#         super(Backbone, self).__init__()
#
#         self.nmf = nmf
#         self.K = 16
#         self.max_iter = 50
#         print("nmf: ", self.nmf)
#         print("K: ", self.K)
#
#         self.mlp = mlp
#         print("mlp: ", self.mlp)
#
#         encoder = models.resnet34(pretrained=True) # resnet34 capture only features and remove last relu and maxpool
#         layers = list(encoder.children())[:-3]
#         self.encoder = nn.Sequential(*layers)
#         self.relu = nn.ReLU(inplace=True)
#         self.model_path = "weights/mambavision_small_1k.pth.tar"
#         self.mamba_vision1 = MambaVision(
#                         depths=[3, 3, 7, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=96,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2)
#         self.mamba_vision1._load_state_dict(self.model_path)
#
#         self.mamba_vision2 = MambaVision(
#                         depths=[3, 3, 7, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=96,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2)
#         self.mamba_vision2._load_state_dict(self.model_path)
#         # self.mamba_vision1 = MambaVision(
#         #     depths=[1, 3, 8, 4],
#         #     num_heads=[2, 4, 8, 16],
#         #     window_size=[8, 8, 14, 7],
#         #     dim=80,
#         #     in_dim=32,
#         #     mlp_ratio=4,
#         #     resolution=224,
#         #     drop_path_rate=0.2,
#         # )
#         # self.mamba_vision1._load_state_dict(self.model_path)
#         #
#         # self.mamba_vision2 = MambaVision(
#         #     depths=[1, 3, 8, 4],
#         #     num_heads=[2, 4, 8, 16],
#         #     window_size=[8, 8, 14, 7],
#         #     dim=80,
#         #     in_dim=32,
#         #     mlp_ratio=4,
#         #     resolution=224,
#         #     drop_path_rate=0.2,
#         # )
#         # self.mamba_vision2._load_state_dict(self.model_path)
#
#         self.mlp_c = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1)),
#                                  nn.ReLU(),
#                                  nn.Conv2d(128, self.K, kernel_size=(1, 1)),
#                                  nn.ReLU())
#         self.mamba_mlp1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=(1, 1)),
#                                        nn.ReLU())
#         self.mamba_mlp2 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=(1, 1)),
#                                        nn.ReLU())
#         self.net_vlad_cnn1 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
#         self.net_vlad_cnn2 = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
#         self.net_vlad_nmf = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
#         self.net_vlad_mlp = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
#
#
#
#     def forward(self, x):
#         tmp =int(x.shape[0]/2)
#         x_img = x[:tmp,:,:,:]
#         x_sam = x[tmp:,:,:,:]
#
#         # depth
#         x_img = self.mamba_vision1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#         x_img = self.mamba_mlp1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#         x_mid = x_img
#         x_img = self.net_vlad_cnn1(x_img)
#         x_img = F.normalize(x_img, p=2, dim=1)
#
#
#         # MobileSAM
#         x_sam = self.mamba_vision2(x_sam)
#         x_sam = F.normalize(x_sam, p=2, dim=1)
#         x_sam = self.mamba_mlp2(x_sam)
#         x_sam = F.normalize(x_sam,p=2,dim=1)
#         # x_mid = x_sam
#         x_sam = self.net_vlad_cnn2(x_sam)
#         x_sam = F.normalize(x_sam,p=2,dim=1)
#
#         out = torch.cat((x_img, x_sam), 1)
#
#
#         return x_mid, out
#
# class TripletLossSimple(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletLossSimple, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative):
#
#         pos_dist = torch.sqrt((anchor - positive).pow(2).sum(1))
#         neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
#         loss = F.relu(pos_dist-neg_dist + self.margin)
#         return loss.mean()
#
#
# # 定义一个函数来判断两个图像是否匹配
# def is_correct_match(query_index, database_index, poses, threshold=10.0):
#     pose_query = poses[query_index]
#     pose_database = poses[database_index]
#     distance = 0
#     for j in [3, 7]:  # 根据你的需求选择合适的索引
#         distance += np.square(float(pose_query[j]) - float(pose_database[j]))
#     distance = np.sqrt(distance)
#     return distance < threshold