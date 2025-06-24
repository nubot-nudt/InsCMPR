from PIL import Image
import matplotlib.pyplot as plt
import ip_basic
import numpy as np
import os


def normalize_depth_map(depth_map):
    """ 归一化深度图 """
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth_map = (depth_map - min_val) / (max_val - min_val)
    return normalized_depth_map


def process_and_save_tiff_and_png(input_folder, tiff_output_folder, png_output_folder):
    """ 处理并保存 TIFF 和 PNG 文件 """
    if not os.path.exists(tiff_output_folder):
        os.makedirs(tiff_output_folder)
    if not os.path.exists(png_output_folder):
        os.makedirs(png_output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            img = np.array(Image.open(input_path)).astype(np.float32)
            final_depth_map, _ = ip_basic.fill_in_multiscale(img)
            normalized_depth_map = normalize_depth_map(final_depth_map)

            # 保存 TIFF 文件
            tiff_output_path = os.path.join(tiff_output_folder, filename)
            Image.fromarray(normalized_depth_map).save(tiff_output_path)
            print(f"Processed and saved {tiff_output_path}")

            # 保存 PNG 文件
            png_output_path = os.path.join(png_output_folder, os.path.splitext(filename)[0] + '.png')
            normalized_depth_map_uint8 = (normalized_depth_map * 255).astype(np.uint8)
            Image.fromarray(normalized_depth_map_uint8).save(png_output_path)
            print(f"Processed and saved {png_output_path}")


# 示例使用
input_folder = "/media/shorwin/Shorwin/HAOMO/lidar1"
tiff_output_folder = "/media/shorwin/Shorwin/HAOMO/lidar"
png_output_folder = "/media/shorwin/Shorwin/HAOMO/lidar2"
process_and_save_tiff_and_png(input_folder, tiff_output_folder, png_output_folder)