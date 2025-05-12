#ifndef NOISE_UTILS_HPP
#define NOISE_UTILS_HPP

#include <opencv2/opencv.hpp>

cv::Mat addGaussianNoise(const cv::Mat &src, double mean = 0.9, double stddev = 15.0);
cv::Mat addSaltAndPepperNoise(const cv::Mat &src, double amount = 0.01);
// void addSaltAndPepperNoise(cv::Mat &src, double pa = 0.00000000001, double pb = 0.00000000001);
cv::Mat addSpeckleNoise(const cv::Mat &src, double variance = 0.01);

#endif // NOISE_UTILS_HPP
