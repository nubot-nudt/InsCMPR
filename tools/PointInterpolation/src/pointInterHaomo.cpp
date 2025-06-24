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
#include "osList.h"
#include <unistd.h>
#include <omp.h>
#include "utils.h"
using namespace std;

vector<vector<double>> load_pose(string path)
{
    ifstream ifs;
    ifs.open(path, ios::in);
    string buff;
    assert(ifs.is_open());
    vector<vector<double>> time_pose;
    while (getline(ifs, buff))
    {
        time_pose.push_back(segment(buff));
    }
    ifs.close();
    for (int i = 0; i < time_pose.size(); i++)
    {
        time_pose[i][0] *= 1e6;
    }
    return time_pose;
}

vector<pair<string, string>> load_pair(string path)
{
    ifstream ifs;
    ifs.open(path, ios::in);
    string buff;
    assert(ifs.is_open());
    vector<pair<string, string>> cam_lid;
    while (getline(ifs, buff))
    {
        pair<string, string> temp;
        temp.first = buff.substr(0, 20);
        temp.second = buff.substr(21, 9);
        cam_lid.push_back(temp);
    }
    ifs.close();
    return cam_lid;
}

Eigen::Affine3d quaternionToRotationMatrix(double w, double x, double y, double z)
{
    Eigen::Affine3d rotationMatrix;

    double sqw = w * w;
    double sqx = x * x;
    double sqy = y * y;
    double sqz = z * z;

    rotationMatrix(0, 0) = sqx - sqy - sqz + sqw;
    rotationMatrix(1, 0) = 2.0 * (x * y + z * w);
    rotationMatrix(2, 0) = 2.0 * (x * z - y * w);

    rotationMatrix(0, 1) = 2.0 * (x * y - z * w);
    rotationMatrix(1, 1) = -sqx + sqy - sqz + sqw;
    rotationMatrix(2, 1) = 2.0 * (y * z + x * w);

    rotationMatrix(0, 2) = 2.0 * (x * z + y * w);
    rotationMatrix(1, 2) = 2.0 * (y * z - x * w);
    rotationMatrix(2, 2) = -sqx - sqy + sqz + sqw;

    return rotationMatrix;
}

void transform_point_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < cloud->size(); i++)
    {
        if (cloud->points[i].x > 0 && cloud->points[i].intensity<40)
        {
            temp->push_back(cloud->points[i]);
        }
    }
    *cloud = *temp;

    Eigen::Affine3d transform_1 = Eigen::Affine3d::Identity();
    transform_1.translation() << 0.029999999329447746, -1.6729999780654907, -1.149999976158142;
    pcl::transformPointCloud(*cloud, *cloud, transform_1);

    Eigen::Affine3d R = quaternionToRotationMatrix(0.5026944875717163, -0.4975827634334564, 0.49832797050476074, -0.5013769865036011);
    pcl::transformPointCloud(*cloud, *cloud, R.inverse());

    Eigen::Affine3d transform_2 = Eigen::Affine3d::Identity();
    transform_2.translation() << -1.9454224109649658, 0.07745788991451263, -1.1428420543670654;
    pcl::transformPointCloud(*cloud, *cloud, transform_2);
}

void upsample(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>);
    *cloud_copy = *cloud;
    int n = 50;
    int inn = 6;
    for (int i = 0; i < cloud->size(); i++)
    {
        std::vector<int> kdtree_index;
        std::vector<float> kdtree_sq_distance;
        kdtree.nearestKSearch(cloud->points[i], n, kdtree_index, kdtree_sq_distance);
        int c = 0;
        for (int j = 5; j < n; j++)
        {
            if (cloud->points[i].intensity - cloud->points[kdtree_index[j]].intensity < 0.2)
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
    // kdtree.setInputCloud(cloud_copy);
    // for (int i = 0; i < cloud_copy->size(); i++)
    // {
    //     std::vector<int> kdtree_index;
    //     std::vector<float> kdtree_sq_distance;
    //     kdtree.nearestKSearch(cloud->points[i], 5, kdtree_index, kdtree_sq_distance);
    //     for (int j = 0; j < 5; j++)
    //     {
    //         pcl::PointXYZI temp;
    //         temp.x = (cloud_copy->points[i].x + cloud_copy->points[kdtree_index[j]].x) / 2;
    //         temp.y = (cloud_copy->points[i].y + cloud_copy->points[kdtree_index[j]].y) / 2;
    //         temp.z = (cloud_copy->points[i].z + cloud_copy->points[kdtree_index[j]].z) / 2;
    //         temp.intensity = cloud->points[i].intensity;
    //         cloud->push_back(temp);
    //     }
    // }

    return;
}


void pad_line(cv::Mat &img)
{
    int t=3;
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            if(img.at<float>(i,j)==0)
            {
                int hp = 0;
                for(int h=1; h<t; h++)
                {
                    if(img.at<float>(i+h,j) > 0)
                    {
                        hp = h;
                        break;
                    }
                }
                int hm = 0;
                for(int h=1; h<t; h++)
                {
                    if(img.at<float>(i-h,j) > 0)
                    {
                        hm = h;
                        break;
                    }
                }
                if(hp==0 | hm==0)
                {
                    continue;
                }
                img.at<float>(i,j) = (hm*img.at<float>(i+hp,j) + hp*img.at<float>(i-hm,j))/(hp+hm);
            }
        }
    }
}
void pad_line2(cv::Mat &img)
{
    int t=3;
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            if(img.at<float>(i,j)==0)
            {
                int hp = 0;
                for(int h=1; h<t; h++)
                {
                    if(img.at<float>(i,j+h) > 0)
                    {
                        hp = h;
                        break;
                    }
                }
                int hm = 0;
                for(int h=1; h<t; h++)
                {
                    if(img.at<float>(i,j-h) > 0)
                    {
                        hm = h;
                        break;
                    }
                }
                if(hp==0 | hm==0)
                {
                    continue;
                }
                img.at<float>(i,j) = (hm*img.at<float>(i,j+hp) + hp*img.at<float>(i,j-hm))/(hp+hm);
            }
        }
    }
}


cv::Mat align(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    Eigen::MatrixXd pc(cloud->size(), 4);
    for (int i = 0; i < cloud->size(); i++)
    {
        pc(i, 0) = cloud->points[i].x;
        pc(i, 1) = cloud->points[i].y;
        pc(i, 2) = cloud->points[i].z;
        pc(i, 3) = 1;
    }
    Eigen::Matrix3d intrinsic;
    intrinsic << 3719.18, 0., 1915.66,
        0., 3719.83, 1072.77,
        0., 0., 1.;
    Eigen::MatrixXd pc_align = (intrinsic * pc.transpose()).transpose();
    int width = 320;
    int height = 102;
    float maxdepth = 100;
    cv::Mat board = cv::Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < cloud->size(); i++)
    {
        pc_align(i, 0) = pc_align(i, 0) / pc_align(i, 2);
        pc_align(i, 1) = pc_align(i, 1) / pc_align(i, 2);

        int x = pc_align(i, 1) / 2160 * height - 30;
        int y = pc_align(i, 0) / 3840 * width;

        if (x < 0 || x >= height || y < 0 || y >= width)
        {
            continue;
        }
        float depth = min(pointDepth(cloud->points[i]), maxdepth);
        if (board.at<float>(x, y) == 0.0)
        {
            board.at<float>(x, y) = depth / maxdepth;
        }
        else
        {
            if (depth/maxdepth < board.at<float>(x, y))
            {
                board.at<float>(x, y) = depth/maxdepth;
            }
            else
            {
                continue;
            }
        }
    }
    cv::Rect select = cv::Rect(0, 0, 320, 45);
	cv::Mat board2 = board(select);

    return board2;
}

int main()
{
    // cout << fixed;

    string pcd_path = "/media/shorwin/Shorwin/HAOMO/pcd";
    string img_path = "/media/shorwin/Shorwin/HAOMO/images";
    string root = "/media/shorwin/Shorwin/HAOMO";

    vector<vector<double>> time_pose = load_pose(root + "/pose_kitti_fmt.json");

    pcl::PCDReader reader;
    vector<pair<string, string>> pair = load_pair(root + "/pair.txt");
    for (int i = 0; i < pair.size(); i++)
    {
        // cout << pcd_path + "/" + pair[i].second + " -> " + img_path + "/" + pair[i].first << endl;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        reader.read(pcd_path + "/" + pair[i].second, *cloud);

        // float mmm = 0;
        for(int j=0; j<cloud->size(); j++)
        {
            float angle = atan2(cloud->points[j].z, sqrt(cloud->points[j].x*cloud->points[j].x+cloud->points[j].y*cloud->points[j].y)) * 200;
            // mmm = max(mmm, cloud->points[j].intensity);
            cloud->points[j].intensity = angle;
        }
        // cout << mmm << endl;

        upsample(cloud);
        string name_pcd = "/media/shorwin/Shorwin/HAOMO/uppcd/" + pair[i].second.substr(0, 5) + ".pcd";
        cout << name_pcd << endl;
        pcl::io::savePCDFileASCII(name_pcd, *cloud);
        continue;
        // pcl::visualization::CloudViewer viewer("cloud view");
        // viewer.showCloud(cloud);
        // while (!viewer.wasStopped())
        // {
        // }
        transform_point_cloud(cloud);
        cv::Mat board = align(cloud);
        // pad_line(board);
        float maxValue = *max_element(board.begin<float>(), board.end<float>());
        cout << maxValue << endl;
        // pad_line2(board);
        // cv::imshow("BEV", board);
        // if (cv::waitKey(0) == 27)
        // {
        //     break;
        // }
        string name = "/media/shorwin/Shorwin/HAOMO/lidar/" + pair[i].second.substr(0, 5) + ".tiff";
        cout << name << " " << i << "/" << pair.size() << endl;
        cv::imwrite(name, board);
    }
    return 0;
}