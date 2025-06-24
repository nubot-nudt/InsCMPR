#include <fstream>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include "osList.h"
#include <unistd.h>
#include <omp.h>


using namespace std;

// 加载calib文件
std::vector<Eigen::Matrix<float, 3, 4>> readCalib(string path)
{
    vector<Eigen::Matrix<float, 3, 4>> calib(5); // P0 P1 P2 P3 Tr

    std::ifstream file(path);
    string temp_str;
    for (int n = 0; n < 5; n++)
    {
        file >> temp_str;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                file >> temp_str;
                calib[n](i, j) = std::stof(temp_str);
            }
        }
    }
    file.close();

    return calib;
}

// 获取VELODYNE BIN文件中所包含的点数
int getBinSize(string path)
{
    int size = 0;
    FILE *fp = fopen(path.c_str(), "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    size = size / (int)sizeof(float) / 4;
    return size;
}

// 读取KITTI VELODYNE点云
Eigen::MatrixXf readBin(string path, int size)
{
    Eigen::MatrixXf pc(size, 4);
    std::ifstream velodyne_bin(path, std::ios::binary);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            float data;
            velodyne_bin.read((char *)&data, sizeof(float));
            pc(i, j) = data;
        }
    }
    velodyne_bin.close();
    return pc;
}

template <typename T>
float pointDepth(T p)
{
    float depth = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return depth;
}

// Eigen::MatrixXi genRIM(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int height = 64, int width = 950, float fov_up = 5, float fov_down = -25)
// {
//     fov_up = fov_up / 180.0 * M_PI;
//     fov_down = fov_down / 180.0 * M_PI;
//     float fov = abs(fov_down) + abs(fov_up);

//     int n = cloud->size();
//     std::vector<float> depth(n);
// #pragma omp parallel for num_threads(8)
//     for (int i = 0; i < n; i++)
//     {
//         depth[i] = pointDepth(cloud->points[i]);
//     }
//     Eigen::MatrixXf rangeMap(height, width);
//     Eigen::MatrixXi rangeIndMap(height, width);
//     float mm = 0;
//     for (int i = 0; i < n; i++)
//     {
//         float yaw = -atan2(cloud->points[i].y, cloud->points[i].x);
//         float pitch = asin(cloud->points[i].z / depth[i]);

//         float proj_x = 0.5 * (yaw / M_PI + 1.0);
//         float proj_y = 1.0 - (pitch + abs(fov_down)) / fov;

//         proj_x *= width;
//         proj_y *= height;

//         // if(rangeMap(proj_x, proj_y) == 0.0 || rangeMap(proj_x, proj_y) < depth[i])
//         // {
//         int x = min(max((int)proj_x, 0), width-1);
//         int y = min(max((int)proj_y, 0), height-1);
//         // cout << x<< endl;
//         // cout << y << endl;
//         rangeMap(y, x) = depth[i];
//         rangeIndMap(y, x) = i;
//         // }
        
//         // cout << rangeMap((int)proj_y, (int)proj_x) << endl;
//     }
//     cv::Mat board = cv::Mat::zeros(height, width, CV_32FC1);
//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             rangeMap(i, j) = min(rangeMap(i, j), (float)60.0);
//             board.at<float>(i, j) = rangeMap(i, j)/60*255;
//             // cout<<board.at<float>(i,j)<<endl;
//         }
//     }
//     cv::imwrite("board.png", board);
//     return rangeIndMap;
// }

cv::Mat align(string calib_file, int camera_index, string image_path, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    // 读取calib文件
    std::vector<Eigen::Matrix<float, 3, 4>> calib = readCalib(calib_file);
    // 读取图片
    cv::Mat img = cv::imread(image_path, 0);
    // 裁切
    img = img(cv::Rect(0, 165, img.cols, img.rows - 165));
    // 白板
    cv::Mat board = cv::Mat::zeros(img.rows / 3, img.cols / 2, CV_32FC1);
    // Eigen::Matrix<uint, board.rows, board.cols> pointIndMap;
    // 点云转矩阵
    Eigen::MatrixXf pc(cloud->size(), 4);
    for (int i = 0; i < cloud->size(); i++)
    {
        pc(i, 0) = cloud->points[i].x;
        pc(i, 1) = cloud->points[i].y;
        pc(i, 2) = cloud->points[i].z;
        pc(i, 3) = 1;
    }
    // 从lidar到相机的变换矩阵
    Eigen::Matrix<float, 4, 4> calib_lidar;
    calib_lidar.block<3, 4>(0, 0) = calib[4];
    calib_lidar.block<1, 4>(3, 0) = Eigen::Matrix<float, 1, 4>{0, 0, 0, 1};
    Eigen::Matrix<float, 3, 4> P_velo_to_img = calib[camera_index] * calib_lidar;
    Eigen::MatrixXf pc_align = (P_velo_to_img * pc.transpose()).transpose();
    // 投影
    for (int i = 0; i < cloud->size(); i++)
    {
        pc_align(i, 0) = pc_align(i, 0) / pc_align(i, 2);
        pc_align(i, 1) = pc_align(i, 1) / pc_align(i, 2);

        int x = (pc_align(i, 1) - 165) / img.size[0] * board.size[0];
        int y = pc_align(i, 0) / img.size[1] * board.size[1];

        if (x < 0 || x >= board.size[0] || y < 0 || y >= board.size[1])
        {
            continue;
        }
        float depth = pointDepth(cloud->points[i]);
        if (board.at<float>(x, y) == 0.0)
        {
            board.at<float>(x, y) = min(depth, (float)60.0) / (float)60.0;
        }
        else
        {
            if (depth < board.at<float>(x, y))
            {
                board.at<float>(x, y) = min(depth, (float)60.0);
            }
            else
            {
                continue;
            }
        }
    }
    // 补充
    for (int i = 1; i < board.size[0] - 1; i++)
    {
        for (int j = 1; j < board.size[1] - 1; j++)
        {
            if (board.at<float>(i - 1, j) > 0 && board.at<float>(i + 1, j) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i - 1, j), board.at<float>(i + 1, j));
                continue;
            }
            if (board.at<float>(i, j - 1) > 0 && board.at<float>(i, j + 1) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i, j - 1), board.at<float>(i, j + 1));
            }
        }
    }
    for (int i = 2; i < board.size[0] - 2; i++)
    {
        for (int j = 2; j < board.size[1] - 2; j++)
        {
            if (board.at<float>(i - 2, j) > 0 && board.at<float>(i + 2, j) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i - 2, j), board.at<float>(i + 2, j));
            }
            if (board.at<float>(i, j - 2) > 0 && board.at<float>(i, j + 2) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i, j - 2), board.at<float>(i, j + 2));
            }
        }
    }
    for (int i = 1; i < board.size[0] - 1; i++)
    {
        for (int j = 1; j < board.size[1] - 1; j++)
        {
            if (board.at<float>(i - 1, j) > 0 && board.at<float>(i + 1, j) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i - 1, j), board.at<float>(i + 1, j));
                continue;
            }
            if (board.at<float>(i, j - 1) > 0 && board.at<float>(i, j + 1) > 0 && board.at<float>(i, j) == 0)
            {
                board.at<float>(i, j) = min(board.at<float>(i, j - 1), board.at<float>(i, j + 1));
            }
        }
    }
    return board;
}

void upsample(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>);
    *cloud_copy = *cloud;
    int n = 50;
    int inn = 6;
    int c = 0;
    for (int i = 0; i < cloud->size(); i++)
    {
        std::vector<int> kdtree_index;
        std::vector<float> kdtree_sq_distance;
        kdtree.nearestKSearch(cloud->points[i], n, kdtree_index, kdtree_sq_distance);
        for (int j = 5; j < n; j++)
        {
            if (cloud->points[i].intensity == cloud->points[kdtree_index[j]].intensity)
            {
                continue;
            }
            for (int k = 1; k < inn; k++)
            {
                pcl::PointXYZI temp;
                temp.x = (cloud->points[i].x / inn * k + cloud->points[kdtree_index[j]].x / inn * (inn - k));
                temp.y = (cloud->points[i].y / inn * k + cloud->points[kdtree_index[j]].y / inn * (inn - k));
                temp.z = (cloud->points[i].z / inn * k + cloud->points[kdtree_index[j]].z / inn * (inn - k));
                temp.intensity = cloud->points[i].intensity;
                cloud_copy->push_back(temp);
            }
            c++;
            if (c > 2)
            {
                break;
            }
        }
    }
    *cloud = *cloud_copy;
    kdtree.setInputCloud(cloud_copy);
    for (int i = 0; i < cloud_copy->size(); i++)
    {
        std::vector<int> kdtree_index;
        std::vector<float> kdtree_sq_distance;
        kdtree.nearestKSearch(cloud->points[i], 5, kdtree_index, kdtree_sq_distance);
        for (int j = 0; j < 5; j++)
        {
            pcl::PointXYZI temp;
            temp.x = (cloud_copy->points[i].x + cloud_copy->points[kdtree_index[j]].x) / 2;
            temp.y = (cloud_copy->points[i].y + cloud_copy->points[kdtree_index[j]].y) / 2;
            temp.z = (cloud_copy->points[i].z + cloud_copy->points[kdtree_index[j]].z) / 2;
            temp.intensity = cloud->points[i].intensity;
            cloud->push_back(temp);
        }
    }

    return;
}


int main()
{
    int mm=0;
    // 数据集路径
    string dataset_path = "/home/shorwin/work/SemanticKitti/dataset";
    std::vector<std::string> seq = { "02", "05", "06", "08"};
    for(auto seqn: seq)
    {
        std::vector<std::string> files = pathList(dataset_path+"/sequences/"+seqn+"/velodyne");
        // #pragma omp parallel for num_threads(4)
        for(int i=0; i<files.size(); i++)
        {
            string velodyne = dataset_path+"/sequences/"+seqn+"/velodyne/"+files[i];
            string camera = dataset_path+"/sequences/"+seqn+"/image_2/"+files[i].substr(0,6)+".png";
            string calib = dataset_path+"/sequences/"+seqn+"/calib.txt";
            // 处理后数据存放路径
            string depth_dir = dataset_path+"/sequences/"+seqn+"/lidar/"; 
            if (access(depth_dir.c_str(),F_OK) != 0)
            {
                string cmd = "mkdir " + depth_dir;
                system(cmd.c_str());
            }
            // if(access(("/media/xwd/XWDSF306/data_odometry_velodyne/dataset/sequences/"+seqn+"/lidar6/"+files[i].substr(0,6)+".tiff").c_str(), F_OK) == 0)
            // {
            //     continue;
            // }
            cout<<velodyne<<endl;
            cout<<camera<<endl;

            // read point cloud
            Eigen::MatrixXf pc = readBin(velodyne, getBinSize(velodyne));
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            for(int i=0;i<pc.rows();i++)
            {
                if(pc(i,0) > 1.0)
                {
                    pcl::PointXYZI temp;
                    temp.x = pc(i,0);
                    temp.y = pc(i,1);
                    temp.z = pc(i,2);
                    temp.intensity = i/(pc.rows()/32);
                    cloud->push_back(temp);
                }
            }
            mm = max(mm, (int)cloud->size());

            // pcl::visualization::CloudViewer viewer2("cloud view");
            // viewer2.showCloud(cloud);
            // while (!viewer2.wasStopped())
            // {

            // }
            upsample(cloud);
            cv::Mat img = align(calib, 2, camera, cloud);

            cv::imwrite(depth_dir+files[i].substr(0,6)+".tiff", img);
        }

    }


    return 0;
}