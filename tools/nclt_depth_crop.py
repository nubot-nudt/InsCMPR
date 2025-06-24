import open3d as o3d
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ip_basic
from PIL import Image
import os
from tqdm import tqdm


def convert(x_s, y_s, z_s):
    scaling = 0.005  # 5 mm
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
    sr = np.sin(np.pi / 180.0 * ssc[3])
    cr = np.cos(np.pi / 180.0 * ssc[3])
    sp = np.sin(np.pi / 180.0 * ssc[4])
    cp = np.cos(np.pi / 180.0 * ssc[4])
    sh = np.sin(np.pi / 180.0 * ssc[5])
    ch = np.cos(np.pi / 180.0 * ssc[5])

    H = np.zeros((4, 4))
    H[0, 0] = ch * cp
    H[0, 1] = -sh * cr + ch * sp * sr
    H[0, 2] = sh * sr + ch * sp * cr
    H[1, 0] = sh * cp
    H[1, 1] = ch * cr + sh * sp * sr
    H[1, 2] = -ch * sr + sh * sp * cr
    H[2, 0] = -sp
    H[2, 1] = cp * sr
    H[2, 2] = cp * cr
    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]
    H[3, 3] = 1
    return H


def interpolate_depth_image(depth_image):
    # 深度图像的副本
    board = np.copy(depth_image)

    height, width = board.shape

    # 第一轮插值
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if board[i - 1, j] > 0 and board[i + 1, j] > 0 and board[i, j] == 0:
                board[i, j] = min(board[i - 1, j], board[i + 1, j])
                continue
            if board[i, j - 1] > 0 and board[i, j + 1] > 0 and board[i, j] == 0:
                board[i, j] = min(board[i, j - 1], board[i, j + 1])

    # 第二轮插值
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            if board[i - 2, j] > 0 and board[i + 2, j] > 0 and board[i, j] == 0:
                board[i, j] = min(board[i - 2, j], board[i + 2, j])
            if board[i, j - 2] > 0 and board[i, j + 2] > 0 and board[i, j] == 0:
                board[i, j] = min(board[i, j - 2], board[i, j + 2])

    # # 再次进行第一轮插值
    # for i in range(1, height - 1):
    #     for j in range(1, width - 1):
    #         if board[i - 1, j] > 0 and board[i + 1, j] > 0 and board[i, j] == 0:
    #             board[i, j] = min(board[i - 1, j], board[i + 1, j])
    #             continue
    #         if board[i, j - 1] > 0 and board[i, j + 1] > 0 and board[i, j] == 0:
    #             board[i, j] = min(board[i, j - 1], board[i, j + 1])

    return board


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


def save_depth_image(x_im, y_im, z_im, image_shape, depth_image_path, depth_img_path, final_depth_map_path, crop_size):
    depth_image = np.zeros(image_shape)
    for x, y, z in zip(x_im, y_im, z_im):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_image[int(y), int(x)] = z
    # depth_img = interpolate_depth_image(depth_image)

    # final_depth_map= ip_basic.fill_in_fast(depth_image)
    final_depth_map, _ = ip_basic.fill_in_multiscale(depth_image)


    depth_image_normalized = (255 * (depth_image - np.min(depth_image)) / np.ptp(depth_image)).astype(np.float32)
    depth_image_pil = Image.fromarray(depth_image_normalized)
    cropped_depth_image = depth_image_pil.crop(crop_size)
    cropped_depth_image = cropped_depth_image.rotate(-90, expand=True)
    cropped_depth_image.save(depth_image_path)

    # # 将深度图归一化到0-255的范围
    # depth_img_normalized = (255 * (depth_img - np.min(depth_img)) / np.ptp(depth_img)).astype(np.float32)
    # depth_img_pil = Image.fromarray(depth_img_normalized)
    # cropped_depth_img = depth_img_pil.crop((700, 250, 850, 950))
    # cropped_depth_img = cropped_depth_img.rotate(-90, expand=True)
    # cropped_depth_img.save(depth_img_path)

    # 将深度图归一化到0-255的范围
    final_depth_map_normalized = (255 * (final_depth_map - np.min(final_depth_map)) / np.ptp(final_depth_map)).astype(
        np.float32)
    final_depth_map_pil = Image.fromarray(final_depth_map_normalized)
    cropped_final_depth_map = final_depth_map_pil.crop(crop_size)
    cropped_final_depth_map = cropped_final_depth_map.rotate(-90, expand=True)
    cropped_final_depth_map.save(final_depth_map_path)

    # # 创建一个包含两个子图的图形
    # fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # # 在第一个子图中显示第一张图片
    # axes[0].imshow(cropped_depth_image, cmap='viridis')
    # axes[0].axis('off')  # 隐藏坐标轴
    # axes[0].set_title('Depth')

    # # 在第二个子图中显示第二张图片
    # axes[1].imshow(cropped_depth_img, cmap='viridis')
    # axes[1].axis('off')  # 隐藏坐标轴
    # axes[1].set_title('Depth DC(ModaLink)')

    # # 在第三个子图中显示第三张图片
    # axes[2].imshow(cropped_final_depth_map, cmap='viridis')
    # axes[2].axis('off')  # 隐藏坐标轴
    # axes[2].set_title('Depth DC(Ours)')

    # plt.show()


def upsample_point_cloud(point_cloud, k=5, num_new_points=1):
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    new_points = []
    new_colors = []

    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], k)
        for j in range(1, len(idx)):
            point_1 = points[i]
            point_2 = points[idx[j]]
            color_1 = colors[i]
            color_2 = colors[idx[j]]
            for t in range(1, num_new_points + 1):
                new_point = point_1 + t / (num_new_points + 1) * (point_2 - point_1)
                new_points.append(new_point)
                new_color = color_1 + t / (num_new_points + 1) * (color_2 - color_1)
                new_colors.append(new_color)

    new_points = np.array(new_points)
    new_colors = np.array(new_colors)
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(np.vstack((points, new_points)))
    new_point_cloud.colors = o3d.utility.Vector3dVector(np.vstack((colors, new_colors)))
    return new_point_cloud


def convert_to_original_format(upsampled_pcd):
    points = np.asarray(upsampled_pcd.points)
    colors = np.asarray(upsampled_pcd.colors)[:, 0]  # Assuming intensity is stored in the first channel
    intensities = colors * 255  # Denormalize intensities to original scale
    hits = np.hstack((points, intensities.reshape(-1, 1)))
    return hits


def main():
    Cams = [5]
    for cam in Cams:
        if cam == 3:
            crop_size = (695, 300, 895, 900)
        elif cam == 1:
            crop_size = (697, 338, 897, 910)
        elif cam == 5:
            crop_size = (710, 340, 910, 920)
        elif cam == 2:
            crop_size = (679, 310, 879, 910)
        else:
            crop_size = (706, 300, 906, 910)

        undistort_img_path = f"/media/shorwin/My Passport/nclt/depth/2013-04-05/Cam{cam}"
        lidar_path = "/mnt/data/nclt/data/velodyne_data/2013-04-05_vel/velodyne_sync"
        save_depth_path = f"/home/shorwin/work/depth/2013-04-05/Caam{cam}/depth"
        save_depth_M_path = f"/home/shorwin/work/depth/2013-04-05/Caam{cam}/depth_M"
        save_depth_Ours_path = f"/home/shorwin/work/depth/2013-04-05/Caam{cam}/depth_Ours"

        if not os.path.exists(save_depth_path):
            os.makedirs(save_depth_path)

        if not os.path.exists(save_depth_M_path):
            os.makedirs(save_depth_M_path)
        if not os.path.exists(save_depth_Ours_path):
            os.makedirs(save_depth_Ours_path)

        lidars = sorted(os.listdir(lidar_path))
        print('Loaded camera calibration')

        for lidar in tqdm(lidars):
            lidar_file_path = os.path.join(lidar_path, lidar)
            img_name = lidar.split('.')[0] + ".tiff"
            img_file_path = os.path.join(undistort_img_path, img_name)
            # print(lidar_file_path, img_file_path)
            if not os.path.exists(img_file_path):
                print("image file does not exists!")
                continue

            hits_body = load_vel_hits(lidar_file_path)
            image = mpimg.imread(img_file_path)
            hits_image = project_vel_to_cam(hits_body, cam)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(hits_image[:3, :].T)
            intensities = np.ones((hits_image.shape[1],))  # Normalize intensities to [0, 1]
            point_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(intensities.reshape(-1, 1), (1, 3)))  # Use colors to store intensities

            upsampled_pcd = upsample_point_cloud(point_cloud)

            upsampled_hits = convert_to_original_format(upsampled_pcd)

            x_im = upsampled_hits[:, 0] / upsampled_hits[:, 2]
            y_im = upsampled_hits[:, 1] / upsampled_hits[:, 2]
            z_im = upsampled_hits[:, 2]

            idx_infront = z_im > 0
            x_im = x_im[idx_infront]
            y_im = y_im[idx_infront]
            z_im = z_im[idx_infront]
            depth_name = lidar.split('.')[0] + ".tiff"
            depth_image_path = os.path.join(save_depth_path, depth_name)
            depth_img_path = os.path.join(save_depth_M_path, depth_name)
            final_depth_map_path = os.path.join(save_depth_Ours_path, depth_name)
            save_depth_image(x_im, y_im, z_im, image.shape[:2], depth_image_path, depth_img_path, final_depth_map_path,
                             crop_size)

    return 0


if __name__ == '__main__':
    main()
