import cv2
import os
import sys
import time
import numpy as np
import ip_basic
import obj_utils
import matplotlib.pyplot as plt

fill_type = 'multiscale'


def main():

    dataset_path = "/home/shorwin/work/SemanticKitti/dataset"
    seq = ["00"]
    for s in seq:
        files = sorted(os.listdir(dataset_path+"/sequences/"+s+"/velodyne"))
        calib = dataset_path + "/sequences/" + s + "/calib.txt"
        depth_dir = dataset_path + "/sequences/" + s + "/lidar_D80t/"
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
        # print(files)
        for file in files:
            velodyne = dataset_path+"/sequences/"+s+"/velodyne/"+file
            # 读取校准信息
            calib_data = {}
            with open(calib, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    calib_data[key] = np.array([float(x) for x in value.split()])
            P2 = calib_data['P2'].reshape(3, 4)
            Tr_velo_to_cam = np.eye(4)
            Tr_velo_to_cam[:3, :4] = calib_data['Tr'].reshape(3, 4)

            # Load point cloud
            velo_points = np.fromfile(velodyne, dtype=np.float32).reshape((-1, 4))[:, 0:3]
            # 将点云转换到相机坐标系
            velo_pc_padded = np.hstack((velo_points[:, :3], np.ones((velo_points.shape[0], 1))))
            cam0_point_cloud = Tr_velo_to_cam @ velo_pc_padded.T
            # 投影点云到图像平面
            cam0_point_cloud, _ = obj_utils.filter_pc_to_area(
                cam0_point_cloud[0:3], area_extents=np.asarray([[-40, 40], [-3, 5], [0, 80]]))

            point_cloud = cam0_point_cloud.T
            pc_padded = np.append(point_cloud.T, np.ones((1, point_cloud.T.shape[1])), axis=0)

            # 投影点云到图像平面
            point_cloud_img = np.dot(P2, pc_padded)
            # point_cloud_img = point_cloud_img.T
            # point_cloud_img[:, :2] /= point_cloud_img[:, 2:3]
            point_cloud_img[0:2] = point_cloud_img[0:2] / point_cloud_img[2]

            point_cloud_img_2 = point_cloud_img[0:2]

            points_in_img_int = np.int32(np.round(point_cloud_img_2))

            # # Remove points outside image
            valid_indices = \
                (points_in_img_int[0] >= 0) & (points_in_img_int[0] < 1241) & \
                (points_in_img_int[1] >= 0) & (points_in_img_int[1] < 376)
            all_points = point_cloud[valid_indices]
            points_in_img_int = points_in_img_int[:, valid_indices]
            # Invert depths
            all_points[:, 2] = 80 - all_points[:, 2]
            # Only save valid pixels, keep closer points when overlapping
            projected_depths = np.zeros((376, 1241))
            valid_indices = [points_in_img_int[1], points_in_img_int[0]]
            projected_depths[valid_indices] = [
                max(projected_depths[
                        points_in_img_int[1, idx], points_in_img_int[0, idx]],
                    all_points[idx, 2])
                for idx in range(points_in_img_int.shape[1])]
            projected_depths[valid_indices] = \
                80 - projected_depths[valid_indices]

            # Fill depth map
            if fill_type == 'multiscale':

                final_depth_map, _ = ip_basic.fill_in_multiscale(projected_depths)

            else:
                raise ValueError('Invalid fill algorithm')

            # depth_map = (final_depth_map * 255 / np.max(final_depth_map)).astype(np.uint8)[165:]
            depth_map = (final_depth_map * 255 / np.max(final_depth_map)).astype(np.float32)[165:]
            depth_map = cv2.resize(depth_map,(800,128))
            # cv2.imwrite(depth_dir+file.split('.')[0]+'.png', depth_map)
            cv2.imwrite(depth_dir+file.split('.')[0]+'.tiff', depth_map)



if __name__ == "__main__":
    main()
