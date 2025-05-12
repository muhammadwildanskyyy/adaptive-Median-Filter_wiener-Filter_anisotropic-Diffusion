#include "filters.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
using namespace std;
using namespace cv;
// 1. Adaptive Median Filter
uchar adaptiveProcess(const Mat &im, int row, int col, int kernelSize, int maxSize)
{
    vector<uchar> pixels;
    for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
    {
        for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        if (zxy > min && zxy < max)
        {
            return zxy;
        }
        else
        {
            return med;
        }
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
        else
            return med;
    }
}

Mat adaptiveMedianFilter(Mat src)
{
    Mat dst;
    int minSize = 3; // 滤波器窗口的起始大小
    int maxSize = 11; // 滤波器窗口的最大尺寸
    copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BORDER_REFLECT);
    int rows = dst.rows;
    int cols = dst.cols;
    for (int j = maxSize / 2; j < rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < cols * dst.channels() - maxSize / 2; i++)
        {
            dst.at<uchar>(j, i) = adaptiveProcess(dst, j, i, minSize, maxSize);
        }
    }
    return dst;
}

// Fungsi Wiener Filter
Mat wienerFilter(const Mat &input, int ksize, double noiseVariance)
{
    Mat inputF;
    input.convertTo(inputF, CV_32F);

    // Hitung rata-rata lokal
    Mat mean;
    blur(inputF, mean, Size(ksize, ksize));

    // Hitung rata-rata kuadrat
    Mat sqr;
    sqr = inputF.mul(inputF);

    Mat meanSqr;
    blur(sqr, meanSqr, Size(ksize, ksize));

    // Varians lokal
    Mat variance = meanSqr - mean.mul(mean);

    // Hindari nilai negatif (bila noiseVariance terlalu besar)
    Mat noiseMat = Mat::ones(variance.size(), CV_32F) * noiseVariance;
    Mat gain;
    divide(variance - noiseMat, variance, gain);

    // Ganti nilai NaN/inf menjadi 0
    gain.setTo(0, variance <= noiseVariance);

    Mat output = mean + gain.mul(inputF - mean);

    Mat result;
    output.convertTo(result, input.type());
    return result;
}

// Fungsi anisotropic diffusion
Mat anisotropicDiffusion(const Mat &inputImage, int iterations, float lambda, float k)
{
    // Konversi gambar ke float
    Mat img;
    inputImage.convertTo(img, CV_32F);

    Mat dst = img.clone();

    for (int i = 0; i < iterations; i++)
    {
        Mat temp = dst.clone();

        // Gradien arah (N, S, E, W)
        Mat north = Mat::zeros(dst.size(), CV_32F);
        Mat south = Mat::zeros(dst.size(), CV_32F);
        Mat east = Mat::zeros(dst.size(), CV_32F);
        Mat west = Mat::zeros(dst.size(), CV_32F);

        // Hitung gradien
        north.rowRange(0, dst.rows - 1) = temp.rowRange(1, dst.rows) - temp.rowRange(0, dst.rows - 1);
        south.rowRange(1, dst.rows) = temp.rowRange(0, dst.rows - 1) - temp.rowRange(1, dst.rows);
        east.colRange(1, dst.cols) = temp.colRange(0, dst.cols - 1) - temp.colRange(1, dst.cols);
        west.colRange(0, dst.cols - 1) = temp.colRange(1, dst.cols) - temp.colRange(0, dst.cols - 1);

        // Fungsi konduktivitas c = exp(-(grad^2 / k^2))
        Mat cN, cS, cE, cW;
        cv::exp(-(north.mul(north)) / (k * k), cN);
        cv::exp(-(south.mul(south)) / (k * k), cS);
        cv::exp(-(east.mul(east)) / (k * k), cE);
        cv::exp(-(west.mul(west)) / (k * k), cW);

        // Update citra
        dst += lambda * (cN.mul(north) + cS.mul(south) + cE.mul(east) + cW.mul(west));
    }

    // Konversi kembali ke tipe 8-bit untuk ditampilkan
    Mat outputImage;
    dst.convertTo(outputImage, CV_8U);
    return outputImage;
}
