import os
from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import h5py
import random
from skimage import feature
import cv2
from matplotlib import pyplot as plt
# from utils.utils import *

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class KITTI(data.Dataset):
    def __init__(self, root="/home/shorwin/work/SemanticKitti/dataset", seq="00", n_neg=10, resize_shape=(800, 128)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 5
        self.resize_shape = resize_shape
        
        pose_path = os.path.join(root, "poses", seq+".txt")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:,3], self.poses[:,7]]).transpose()
        
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        self.img_sam_q = []
        self.img_sam_db = []
        for i in range(3000):
            self.img_q.append('sequences/'+seq+'/image_2/'+f"{i:06d}"+'.png')
            self.img_sam_q.append('sequences/'+seq+'/imgSAM/'+f"{i:06d}"+'.png')
            self.pose_q.append(self.poses[i])
        for i in range(3000):
            self.img_db.append('sequences/' + seq + "/lidar_D50t/" + f"{i:06d}" + '.tiff')
            self.img_sam_db.append('sequences/'+seq+'/lidarSAM_t/'+f"{i:06d}"+'.tiff')
            self.pose_db.append(self.poses[i])
            
            
        
        self.pose_q  = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)
        
    def __getitem__(self, index):
        resize_shape = self.resize_shape
        
        # query image
        q = np.array(Image.open(os.path.join(self.root, self.img_q[index]))).astype(np.float32)[165:]
        q = cv2.resize(q, resize_shape)
        q = input_transform()(q)

        sam_q = np.array(Image.open(os.path.join(self.root, self.img_sam_q[index])).convert('RGB')).astype(np.float32)
        sam_q = input_transform()(sam_q)

        
        
        # positive lidar
        pos = np.array(Image.open(os.path.join(self.root, self.img_db[index]))).astype(np.float32)
        pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
        # pos[pos<0] = 0
        # pos = pos*255
        # pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos) #[3, 55, 400]

        sam_pos = np.array(Image.open(os.path.join(self.root, self.img_sam_db[index])).convert('RGB')).astype(np.float32)
        sam_pos = input_transform()(sam_pos)

        # negtive lidar
        hard_mine = 1
        neg = []
        sam_neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)
        for i in range(self.n_neg-hard_mine):
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[choice[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            # neg_one = neg_one*255
            # neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)

            sam_neg_one = np.array(Image.open(os.path.join(self.root, self.img_sam_db[choice[i]])).convert('RGB')).astype(np.float32)
            sam_neg_one = input_transform()(sam_neg_one)
            sam_neg.append(sam_neg_one)

        rang = rang[np.argsort(diff[diff > self.distThr])]
        for i in range(hard_mine):  # hard mining
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[rang[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[rang[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            # neg_one = neg_one*255
            # neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)

            sam_neg_one = np.array(Image.open(os.path.join(self.root, self.img_sam_db[choice[i]])).convert('RGB')).astype(np.float32)
            sam_neg_one = input_transform()(sam_neg_one)
            sam_neg.append(sam_neg_one)

        neg = np.stack(neg) # [10, 3, 55, 400] 
        sam_neg = np.stack(sam_neg)
        
        return q, pos, neg, sam_q, sam_pos, sam_neg
    
    def __len__(self):
        return len(self.img_q)

class NCLTData(data.Dataset):
    def __init__(self, root="/home/shorwin/work/NCLT/data", seq="2012-01-08", n_neg=10, resize_shape=(1024, 96)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 5
        self.resize_shape = resize_shape

        pose_path = os.path.join(root, "ground_truth", "groundtruth_"+ seq + ".csv")
        self.poses = np.loadtxt(pose_path, delimiter = ",")
        self.poses = np.array([self.poses[:, 1], self.poses[:, 2]]).transpose()
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        self.img_sam_q = []
        self.img_sam_db = []
        lidar_path = os.path.join(root, seq , "lidar/depth_Ours")
        lidar_files = sorted(os.listdir(lidar_path))

        lidar_sam_path = os.path.join(root, seq , "lidar/depth_SAM")
        lidar_sam_files = sorted(os.listdir(lidar_sam_path))

        image_path = os.path.join(root, seq , "image/Cam")
        image_files = sorted(os.listdir(image_path))

        image_sam_path = os.path.join(root, seq , "image/Cam_SAM")
        image_sam_files = sorted(os.listdir(image_sam_path))

        for i in range(len(image_files)):
            self.img_q.append(seq+"/image/Cam/"+image_files[i])
            self.img_sam_q.append(seq+"/image/Cam_SAM/"+image_sam_files[i])
            self.pose_q.append(self.poses[i])
        for i in range(len(lidar_files)):
            self.img_db.append(seq+"/lidar/depth_Ours/"+lidar_files[i])
            self.img_sam_db.append(seq+"/lidar/depth_SAM/"+lidar_sam_files[i])
            self.pose_db.append(self.poses[i])

        self.pose_q = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)

    def __getitem__(self, index):
        resize_shape = self.resize_shape

        # query image
        q = np.array(Image.open(os.path.join(self.root, self.img_q[index]))).astype(np.float32)
        # q = np.load(os.path.join(self.root, self.img_q[index])).astype(np.float32).transpose([1,2,0])
        q = cv2.resize(q, resize_shape)
        q = input_transform()(q)

        sam_q = np.array(Image.open(os.path.join(self.root, self.img_sam_q[index])).convert('RGB')).astype(np.float32)
        sam_q = cv2.resize(sam_q, resize_shape)
        sam_q = input_transform()(sam_q)

        # positive lidar
        pos = np.array(Image.open(os.path.join(self.root, self.img_db[index]))).astype(np.float32)
        # pos = np.load(os.path.join(self.root, self.img_db[index])).astype(np.float32)
        pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
        # pos[pos<0] = 0
        # pos = pos * 255
        pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos)  # [3, 55, 400]

        sam_pos = np.array(Image.open(os.path.join(self.root, self.img_sam_db[index])).convert('RGB')).astype(np.float32)
        sam_pos = cv2.resize(sam_pos, resize_shape)
        sam_pos = input_transform()(sam_pos)

        # negtive lidar
        hard_mine = 1
        neg = []
        sam_neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)

        for i in range(self.n_neg - hard_mine):
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[choice[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            # neg_one = neg_one * 255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one)  # [3, 55, 400]
            neg.append(neg_one)

            sam_neg_one = np.array(Image.open(os.path.join(self.root, self.img_sam_db[choice[i]])).convert('RGB')).astype(np.float32)
            sam_neg_one = cv2.resize(sam_neg_one, resize_shape)
            sam_neg_one = input_transform()(sam_neg_one)
            sam_neg.append(sam_neg_one)

        rang = rang[np.argsort(diff[diff > self.distThr])]
        for i in range(hard_mine):  # hard mining
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[rang[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[rang[i]]))
            neg_one = np.array([neg_one]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            # neg_one = neg_one * 255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one)  # [3, 55, 400]
            neg.append(neg_one)

            sam_neg_one = np.array(
                Image.open(os.path.join(self.root, self.img_sam_db[choice[i]])).convert('RGB')).astype(np.float32)
            sam_neg_one = cv2.resize(sam_neg_one, resize_shape)
            sam_neg_one = input_transform()(sam_neg_one)
            sam_neg.append(sam_neg_one)
        neg = np.stack(neg)  # [10, 3, 55, 400]
        sam_neg = np.stack(sam_neg)

        return q, pos, neg,sam_q, sam_pos, sam_neg

    def __len__(self):
        return len(self.img_q)

class HaomoSAM(data.Dataset):
    def __init__(self, root="/media/shorwin/Shorwin/HAOMO/data", seq="train", n_neg=10, resize_shape=(800, 128)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 5
        self.resize_shape = resize_shape

        pose_path = os.path.join(root, "poses", seq + ".txt")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:, 4], self.poses[:, 8]]).transpose()

        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        self.img_sam_q = []
        self.img_sam_db = []
        for i in range(7583):
            self.img_q.append(seq + '/img_train/' + f"{i:05d}" + '.jpg')
            self.img_sam_q.append(seq + '/img_SAM_train/' + f"{i:05d}" + '.jpg')
            # self.img_q.append('sequences/'+seq+'/rgb/'+str(1000000+i)[1:]+'.npy')
            self.pose_q.append(self.poses[i])
        for i in range(7583):
            self.img_db.append(seq + "/lidar_train/" + f"{i:05d}" + '.tiff')
            self.img_sam_db.append(seq + "/lidar_SAM_train/" + f"{i:05d}" + '.png')
            self.pose_db.append(self.poses[i])

        self.pose_q = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)
        # print(self.img_q )
    def __getitem__(self, index):
        resize_shape = self.resize_shape

        # query image
        q = np.array(Image.open(os.path.join(self.root, self.img_q[index]))).astype(np.float32)
        # q = np.load(os.path.join(self.root, self.img_q[index])).astype(np.float32).transpose([1,2,0])
        q = cv2.resize(q, resize_shape)
        # plt.imsave("crop.png", q, cmap='jet')
        q = input_transform()(q)

        sam_q = np.array(Image.open(os.path.join(self.root, self.img_sam_q[index])).convert('RGB')).astype(np.float32)
        sam_q = cv2.resize(sam_q, resize_shape)
        sam_q = input_transform()(sam_q)

        # positive lidar
        pos = np.array(Image.open(os.path.join(self.root, self.img_db[index]))).astype(np.float32)
        # pos = np.load(os.path.join(self.root, self.img_db[index])).astype(np.float32)
        pos = np.array([pos]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
        # pos[pos<0] = 0
        pos = pos * 255
        pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos)  # [3, 55, 400]

        sam_pos = np.array(Image.open(os.path.join(self.root, self.img_sam_db[index])).convert('RGB')).astype(np.float32)
        sam_pos = cv2.resize(sam_pos, resize_shape)
        sam_pos = input_transform()(sam_pos)

        # negtive lidar
        neg = []
        sam_neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)
        for i in range(self.n_neg):
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[choice[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            neg_one = neg_one * 255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one)  # [3, 55, 400]
            neg.append(neg_one)

            sam_neg_one = np.array(
                Image.open(os.path.join(self.root, self.img_sam_db[choice[i]])).convert('RGB')).astype(np.float32)
            sam_neg_one = cv2.resize(sam_neg_one, resize_shape)
            sam_neg_one = input_transform()(sam_neg_one)
            sam_neg.append(sam_neg_one)

        neg = np.stack(neg)  # [10, 3, 55, 400]
        sam_neg = np.stack(sam_neg)
        return q, pos, neg, sam_q, sam_pos, sam_neg

    def __len__(self):
        return len(self.img_q)






if __name__ == "__main__":
    dataset = KITTI()
    q, pos, neg, sam_q, sam_pos, sam_neg = dataset.__getitem__(1)
    print(len(dataset))
    print(q.shape)
    print(pos.shape)
    print(neg.shape)
    print(sam_q.shape)
    print(sam_pos.shape)
    print(sam_neg.shape)