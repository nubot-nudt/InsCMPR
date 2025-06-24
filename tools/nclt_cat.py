import cv2
import os
import numpy as np

specified_folders = ['Caam2', 'Caam1', 'Caam5', 'Caam4', 'Caam3']  # 替换为你的文件夹顺序

root_path = "/home/shorwin/work/crop/out1"
folders = [os.path.join(root_path, folder) for folder in specified_folders]
print(folders)

image_filenames = sorted(os.listdir(folders[0]))
print(image_filenames)

for filename in image_filenames:
    images = []
    for folder in folders:
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Warning: Unable to read image {image_path}")

    if len(images) == len(folders):
        stitched_image = np.hstack(images)

        output_filename = "/home/shorwin/work/crop/cat/" + filename
        cv2.imwrite(output_filename, stitched_image)
        print(f"Saved {output_filename}")
    else:
        print(f"Warning: Not all images are available for {filename}")
