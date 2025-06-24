import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取范围图像文件（假设范围图像是16位单通道图像）
range_image = cv2.imread('/home/shorwin/work/SemanticKitti/dataset/sequences/00/lidar/000000.tiff', cv2.IMREAD_ANYDEPTH)

# 检查图像是否成功读取
if range_image is None:
    print("无法读取范围图像文件")
    exit()

# 将范围图像转换为灰度图像
# 这里假设范围图像的值范围是0到65535（16位图像）
# 你可以根据实际情况调整最小值和最大值
min_range = np.min(range_image)
max_range = np.max(range_image)
range_image_normalized = (range_image - min_range) / (max_range - min_range)
range_image_gray = (range_image_normalized * 255).astype(np.uint8)

# 将灰度图像转换为伪彩色图像（可选）
range_image_color = cv2.applyColorMap(range_image_gray, cv2.COLORMAP_JET)

# 显示范围图像
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.title('Range Image (Grayscale)')
plt.imshow(range_image_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title('Range Image (Color Map)')
plt.imshow(cv2.cvtColor(range_image_color, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
