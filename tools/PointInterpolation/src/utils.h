#include <opencv2/opencv.hpp>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <chrono>

class TimeCost
{
public:
    TimeCost():startTime_(std::chrono::steady_clock::now()), endTime_(std::chrono::steady_clock::now()), secToMillSec(1000.0)
    {
        Tic();
    }
    ~TimeCost(){}
    void Tic()
    {
        startTime_ = std::chrono::steady_clock::now();
    }
    double Toc()
    {
        endTime_ = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = endTime_ - startTime_;
        return elapsedSeconds.count() * secToMillSec;
    }
private:
    std::chrono::time_point<std::chrono::steady_clock> startTime_;
    std::chrono::time_point<std::chrono::steady_clock> endTime_;
    const double secToMillSec;
private:
    TimeCost(const TimeCost&) = delete;
    TimeCost(TimeCost&&) = delete;
    TimeCost& operator=(const TimeCost&) = delete;
    TimeCost& operator=(TimeCost&&) = delete;
};

template<typename T>
cv::Mat cloudTopView(T pc, int width, float scale, bool show=false)
{
    cv::Mat board = cv::Mat::zeros(width, width, CV_32FC1);
    int n = pc->size();
    for(int i=0; i<n; i++)
    {
        int x = std::min(std::max(0, int(pc->points[i].x*scale+width/2)), width-1);
        int y = std::min(std::max(0, int(pc->points[i].y*scale+width/2)), width-1);
        board.at<float>(x,y) = 1;
    }
    if(show)
    {
        cv::imshow("BEV", board);
        cv::waitKey(0);
    }
    return board;
}

template<typename T>
float pointDepth(T p)
{
    float depth = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    return depth;
}