

#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils/noise.hpp"
#include "utils/filters.hpp"

using namespace cv;
using namespace std;

int main()
{
    // Membaca gambar
    Mat image = imread("output/image.png");
    if (image.empty())
    {
        cout << "Gagal membuka gambar!" << endl;
        return -1;
    }

    // Mengonversi gambar ke grayscale
    Mat imgGray;
    cvtColor(image, imgGray, COLOR_BGR2GRAY);

    // Tambahkan Gaussian Noise
    Mat gaussian = addGaussianNoise(imgGray);

    // Tambahkan Salt-and-Pepper Noise pada gambar yang sudah diberi Gaussian Noise
    Mat saltpepper = addSaltAndPepperNoise(imgGray);
    // addSaltAndPepperNoise(imgGray, 0.1, 0.1);

    // Tambahkan Speckle Noise pada gambar yang sudah diberi Salt-and-Pepper Noise
    Mat speckle = addSpeckleNoise(imgGray);

    // Terapkan filter untuk noise gaussian
    Mat medianFilteredGaussian = adaptiveMedianFilter(gaussian);
    Mat wienerFilteredGaussian = wienerFilter(gaussian);
    Mat anisotropicDiffusionGaussian = anisotropicDiffusion(gaussian);

    // // Terapkan filter untuk noise saltpepper
    Mat medianFilteredSaltpepper = adaptiveMedianFilter(saltpepper);
    Mat wienerFilteredSaltpepper = wienerFilter(saltpepper);
    Mat diffusionFilteredSaltpepper = anisotropicDiffusion(saltpepper);

    Mat medianFilteredSpeckle = adaptiveMedianFilter(speckle);
    Mat wienerFilteredSpeckle = wienerFilter(speckle);
    Mat diffusionFilteredSpeckle = anisotropicDiffusion(speckle);
    // Menampilkan gambar dengan ketiga noise

    imshow("Original", image);
    imshow("gaussian", gaussian);
    imshow("saltpepper", saltpepper);
    imshow("speckle", speckle);

    imshow("median Filtered gaussian", medianFilteredGaussian);
    imshow("wiener Filtered gaussian", wienerFilteredGaussian);
    imshow("diffusion Filtered gaussian", anisotropicDiffusionGaussian);

    imshow("median Filtered saltpepper", medianFilteredSaltpepper);
    imshow("wiener Filtered saltpepper", wienerFilteredSaltpepper);
    imshow("diffusion Filtered saltpepper", diffusionFilteredSaltpepper);

    imshow("median Filtered speckle", medianFilteredSpeckle);
    imshow("wiener Filtered speckle", wienerFilteredSpeckle);
    imshow("diffusion Filtered speckle", diffusionFilteredSpeckle);

    imwrite("output/original.png", image);
    imwrite("output/gaussian.png", gaussian);
    imwrite("output/saltpepper.png", saltpepper);
    imwrite("output/speckle.png", speckle);

    imwrite("output/median_gaussian.png", medianFilteredGaussian);
    imwrite("output/wiener_gaussian.png", wienerFilteredGaussian);
    imwrite("output/diffusion_gaussian.png", anisotropicDiffusionGaussian);

    imwrite("output/median_saltpepper.png", medianFilteredSaltpepper);
    imwrite("output/wiener_saltpepper.png", wienerFilteredSaltpepper);
    imwrite("output/diffusion_saltpepper.png", diffusionFilteredSaltpepper);

    imwrite("output/median_speckle.png", medianFilteredSpeckle);
    imwrite("output/wiener_speckle.png", wienerFilteredSpeckle);
    imwrite("output/diffusion_speckle.png", diffusionFilteredSpeckle);
    waitKey(0);

    return 0;
}
