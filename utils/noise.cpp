
#include "noise.hpp"
#include <random>

using namespace cv;

// Gaussian Noise
cv::Mat addGaussianNoise(const Mat &src, double mean, double stddev)
{
    Mat noise = Mat(src.size(), src.type());
    randn(noise, mean, stddev);
    Mat dst;
    addWeighted(src, 1.0, noise, 1.0, 0.0, dst);
    return dst;
}

// Salt and Pepper Noise
cv::Mat addSaltAndPepperNoise(const Mat &src, double amount)
{
    Mat dst = src.clone();
    int num_salt = static_cast<int>(amount * src.total() / 2);
    int num_pepper = num_salt;

    for (int i = 0; i < num_salt; i++)
    {
        int x = rand() % dst.cols;
        int y = rand() % dst.rows;
        dst.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
    }

    for (int i = 0; i < num_pepper; i++)
    {
        int x = rand() % dst.cols;
        int y = rand() % dst.rows;
        dst.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
    }

    return dst;
}

// Speckle Noise
cv::Mat addSpeckleNoise(const Mat &src, double variance)
{
    Mat noise(src.size(), CV_32F);
    randn(noise, 0, sqrt(variance));

    Mat srcFloat, dst;
    src.convertTo(srcFloat, CV_32F, 1.0 / 255.0);
    Mat speckle = srcFloat + srcFloat.mul(noise);
    speckle = max(speckle, 0.0f);
    speckle = min(speckle, 1.0f);
    speckle.convertTo(dst, CV_8U, 255.0);

    return dst;
}
