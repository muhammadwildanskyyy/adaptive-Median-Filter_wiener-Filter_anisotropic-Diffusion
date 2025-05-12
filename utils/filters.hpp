#pragma once
#include <opencv2/opencv.hpp>

// cv::Mat adaptiveMedianFilter(const cv::Mat &src, int maxWindowSize = 7);
cv::Mat adaptiveMedianFilter(const cv::Mat src);

cv::Mat wienerFilter(const cv::Mat &input, int ksize = 5, double noiseVariance = 400.0);

cv::Mat anisotropicDiffusion(const cv::Mat &inputImage, int iterations = 15, float lambda = 0.25, float k = 15.0);
