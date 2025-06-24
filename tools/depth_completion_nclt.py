import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import ip_basic
import os

def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_vel_hits(filename):
    with open(filename, "rb") as f_bin:
        hits = []

        while True:
            x_str = f_bin.read(2)
            if len(x_str) < 2:
                break

            y_str = f_bin.read(2)
            z_str = f_bin.read(2)
            i_str = f_bin.read(1)
            l_str = f_bin.read(1)

            if len(y_str) < 2 or len(z_str) < 2 or len(i_str) < 1 or len(l_str) < 1:
                break

            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', y_str)[0]
            z = struct.unpack('<H', z_str)[0]
            i = struct.unpack('B', i_str)[0]
            l = struct.unpack('B', l_str)[0]

            x, y, z = convert(x, y, z)

            hits.append([x, y, z, 1])

    hits = np.asarray(hits)

    return hits.transpose()

def ssc_to_homo(ssc):
    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def project_vel_to_cam(hits, cam_num):
    K = np.loadtxt('./cam_params/K_cam%d.csv' % (cam_num), delimiter=',')
    x_lb3_c = np.loadtxt('./cam_params/x_lb3_c%d.csv' % (cam_num), delimiter=',')

    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]

    T_lb3_c = ssc_to_homo(x_lb3_c)
    T_body_lb3 = ssc_to_homo(x_body_lb3)

    T_lb3_body = np.linalg.inv(T_body_lb3)
    T_c_lb3 = np.linalg.inv(T_lb3_c)

    T_c_body = np.matmul(T_c_lb3, T_lb3_body)

    hits_c = np.matmul(T_c_body, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    return hits_im

def save_depth_image(x_im, y_im, z_im, image_shape, output_filename_png):
    depth_image = np.zeros(image_shape)
    for x, y, z in zip(x_im, y_im, z_im):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[int(y), int(x)] = z

    final_depth_map, _ = ip_basic.fill_in_multiscale(depth_image)

    # 将深度图归一化到0-255的范围
    depth_image_normalized = (255 * (final_depth_map - np.min(final_depth_map)) / np.ptp(final_depth_map)).astype(np.float32)
    depth_image_pil = Image.fromarray(depth_image_normalized)
    depth_image_pil.save(output_filename_png)
    print(f"Depth map saved as {output_filename_png}")


def main():
    Cams = [1,2,3,4,5]
    for cam in Cams:
        undistort_img_path = "/media/shorwin/Shorwin/nclt/data/image_data/2012-01-08/lb3/Cam%d"%(cam)
        lidar_path = "/mnt/data/nclt/data/velodyne_data/2012-01-08_vel/velodyne_sync"
        save_depth_path = "/media/shorwin/Shorwin/nclt/data/image_data/2012-01-08/lb3/Cam%d"%(cam)
        lidars = sorted(os.listdir(lidar_path))
        print ('Loaded camera calibration')
        for lidar in lidars:
            lidar_file_path = os.path.join(lidar_path,lidar)
            img_name = lidar.split('.')[0]+".tiff"
            img_file_path = os.path.join(undistort_img_path,img_name)
            print(lidar_file_path,img_file_path)
            if not os.path.exists(img_file_path):
                continue
            hits_body = load_vel_hits(lidar_file_path)
            image = mpimg.imread(img_file_path)
            hits_image = project_vel_to_cam(hits_body, cam)

            x_im = hits_image[0, :] / hits_image[2, :]
            y_im = hits_image[1, :] / hits_image[2, :]
            z_im = hits_image[2, :]

            idx_infront = z_im > 0
            x_im = x_im[idx_infront]
            y_im = y_im[idx_infront]
            z_im = z_im[idx_infront]

            # output_filename_tiff = "depth_map.tiff"
            depth_name = lidar.split('.')[0]+".tiff"
            output_filename_png = os.path.join(save_depth_path, depth_name)
            save_depth_image(x_im, y_im, z_im, image.shape[:2], output_filename_png)

    return 0

if __name__ == '__main__':
    main()
