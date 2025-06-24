"""
Demonstrating how to undistort images.

Reads in the given calibration file, parses it, and uses it to undistort the given
image. Then display both the original and undistorted images.

To use:

    python undistort.py image calibration_file
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import re
import os
from tqdm import tqdm

class Undistort(object):
    def __init__(self, fin, scale=1.0, fmask=None):
        self.fin = fin
        # read in distort
        with open(fin, 'r') as f:
            #chunks = f.readline().rstrip().split(' ')
            header = f.readline().rstrip()
            chunks = re.sub(r'[^0-9,]', '', header).split(',')
            self.mapu = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(' ')
                self.mapu[int(chunks[0]),int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]),int(chunks[1])] = float(chunks[2])
        # generate a mask
        self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
        self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        kernel = np.ones((30,30),np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

    """
    Optionally, define a mask
    """
    def set_mask(self,fmask):
        # add in the additional mask passed in as fmask
        if fmask:
            mask = cv2.cvtColor(cv2.imread(fmask), cv2.COLOR_BGR2GRAY)
            self.mask = self.mask & mask
        new_shape = (int(self.mask.shape[1]*1), int(self.mask.shape[0]*1))
        self.mask = cv2.resize(self.mask, new_shape,
                               interpolation=cv2.INTER_CUBIC)

    """
    Use OpenCV to undistorted the given image
    """
    def undistort(self, img):
        return cv2.resize(cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
                          (self.mask.shape[1], self.mask.shape[0]),
                          interpolation=cv2.INTER_CUBIC)

def main():
    Cams = [1,2,3,4,5]
    for cam in Cams:
        distort_img_path = "/mnt/data/nclt/data/image_data/2012-02-05/lb3/Cam%d"%(cam)
        calibration_file  = "/media/shorwin/Shorwin/nclt/data/U2D_ALL_1616X1232/U2D_Cam%d_1616X1232.txt"%(cam)
        undistort_img_path = "/home/shorwin/work/depth/2012-02-05/Cam%d"%(cam)

        if not os.path.exists(undistort_img_path):
            os.makedirs(undistort_img_path)

        distort_imgs = sorted(os.listdir(distort_img_path))
        undistort = Undistort(calibration_file)
        print ('Loaded camera calibration')

        for im in tqdm(distort_imgs):
            im_path = os.path.join(distort_img_path,im)
            img = cv2.imread(im_path)
            im_undistorted = undistort.undistort(img)
            save_path = os.path.join(undistort_img_path,im)
            cv2.imwrite(save_path,im_undistorted)

if __name__ == "__main__":
    main()