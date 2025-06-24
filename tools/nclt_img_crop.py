import os
from PIL import Image
from tqdm import tqdm


def batch_crop_images(input_dir, output_dir, min_width, max_width, min_height, max_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in tqdm(os.listdir(input_dir)):
        # print(file)
        if file.endswith('.tiff'):
            image_path = os.path.join(input_dir, file)
            image = Image.open(image_path)

            width, height = image.size
            cropped_image = image.crop((min_width, min_height, max_width, max_height))

            rotated_image = cropped_image.rotate(-90, expand=True)

            output_path = os.path.join(output_dir, file)
            rotated_image.save(output_path)
            # print(f"Cropped image saved: {output_path}")


Cams = [1, 2, 3, 4, 5]
for cam in Cams:
    input_dir = "/media/shorwin/My Passport/nclt/depth/2013-04-05/Cam%d" % (cam)
    output_dir = "/home/shorwin/work/crop/out1/Caam%d" % (cam)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if cam == 3:
        min_width = 695
        max_width = 895
        min_height = 300
        max_height = 900
    elif cam == 1:
        min_width = 697
        max_width = 897
        min_height = 338
        max_height = 910
    elif cam == 5:
        min_width = 710
        max_width = 910
        min_height = 340
        max_height = 920
    elif cam == 2:
        min_width = 679
        max_width = 879
        min_height = 310
        max_height = 910
    else:
        min_width = 706
        max_width = 906
        min_height = 300
        max_height = 910

    batch_crop_images(input_dir, output_dir, min_width, max_width, min_height, max_height)
