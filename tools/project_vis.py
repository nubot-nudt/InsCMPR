import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
img = f'/home/shorwin/work/SemanticKitti/dataset/sequences/00/image_2/000000.png'
binary = f'/home/shorwin/work/SemanticKitti/dataset/sequences/00/velodyne/000000.bin'

# 读取校准信息
calib_data = {}
with open(f'/home/shorwin/work/SemanticKitti/dataset/sequences/00/calib.txt','r') as f:
    for line in f.readlines():
        key, value = line.split(':', 1)
        calib_data[key] = np.array([float(x) for x in value.split()])
P2 = calib_data['P2'].reshape(3, 4)
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :4] = calib_data['Tr'].reshape(3, 4)

point_cloud = np.fromfile(binary, dtype=np.float32).reshape((-1,4))

# 将点云转换到相机坐标系
point_cloud_hom = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))
point_cloud_cam = Tr_velo_to_cam @ point_cloud_hom.T

# 投影点云到图像平面
point_cloud_img = P2 @ point_cloud_cam
point_cloud_img = point_cloud_img.T
point_cloud_img[:, :2] /= point_cloud_img[:, 2:3]

# 计算点云到激光雷达的欧几里得距离
distances = np.sqrt(np.sum(point_cloud[:, :3] ** 2, axis=1))

# 读取图像
image = cv2.imread(img)

# 在图像上绘制投影点，并使用欧几里得距离表示深度
for point, dist in zip(point_cloud_img, distances):
    u, v, depth = point
    if depth > 0 and 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        # 使用颜色映射表示深度
        depth_value = int(255 * (dist / 100))  # 假设最大深度为100米
        color = cv2.applyColorMap(np.array([[depth_value]], dtype=np.uint8), cv2.COLORMAP_JET).flatten()
        cv2.circle(image, (int(u), int(v)), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
# 显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Projection')
plt.show()
