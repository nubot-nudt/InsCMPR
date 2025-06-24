import os
from tqdm import tqdm
def list_files(directory, output_file):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append((filename, file_path))
    
    file_paths.sort(key=lambda x: x[0])  # 按文件名排序
    
    with open(output_file, 'w') as file:
        for _, file_path in file_paths:
            file.write(file_path + '\n')



if __name__ == "__main__":
    sequences_list = ["00","02","05","06","08"]
    dataset_root = "/media/nubot/data/SemanticKitti/dataset"
    sub_dir = ["image_2","lidar"]
    for seq in tqdm(sequences_list):
        for dir in sub_dir:
            directory_path = dataset_root + '/sequences/' + seq + '/'+ dir  # 替换为你的目录路径
            output_txt_file =dataset_root + '/sequences/' + seq + '/'+ dir + '.txt'  # 输出文件名
            list_files(directory_path, output_txt_file)
        