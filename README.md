

## Overview

Create adaptive filters (adaptive Median Filter, Wiener Filter, and Anisotropic Diffusion) which will be implemented on three noises (Gaussian Noise, Salt-and-Pepper Noise, and Speckle Noise).

## Key Features

- **mage Noise Simulation**: Implemented three different types of noise—Gaussian, Salt-and-Pepper, and Speckle—on grayscale images to simulate real-world image degradation, allowing robust testing of denoising algorithms.
- **Noise Filtering System**: Applied three denoising techniques—Adaptive Median Filter, Wiener Filter, and Anisotropic Diffusion—for each noise type, enabling a comparative analysis of filtering performance on different noise characteristics.
- **Modular Utilities Integration**: Utilized custom utility modules (noise.hpp, filters.hpp) to separate concerns, making the codebase cleaner and more maintainable. This modularity also supports easy experimentation with different filtering strategies.
- **Result Visualization and Export**: Leveraged OpenCV's imshow() and imwrite() functions to both preview and export the results of each filtering step, ensuring easy verification and further processing of the output images.
- **Performance and Scalability**: Designed to work with image resizing for better performance during testing and to maintain consistency in comparison across noise and filter combinations.
- **Grayscale Conversion Pipeline**: Ensured consistency in noise application by converting images to grayscale before processing, creating a standardized baseline for noise and filtering operations.

## Technologies Used

- **C++**: Core programming language used to implement the image processing pipeline with high performance and fine-grained control over memory and computation.
- **OpenCV**: Open-source computer vision library used for image manipulation, grayscale conversion, resizing, noise addition, filtering, visualization, and exporting images.
- **Custom Utility Modules (noise.hpp, filters.hpp)**: Encapsulated logic for various noise generation methods and filtering techniques, promoting code modularity and reusability.


## Challenges and Learnings

During the development of this program, the main challenges faced were understanding the characteristics of the different types of noise—Gaussian, Salt-and-Pepper, and Speckle—and selecting the appropriate filtering method for each type. This process provides a deep understanding of the behavior of filters such as Adaptive Median, Wiener, and Anisotropic Diffusion. In addition, designing a modular code structure using header files such as noise.hpp and filters.hpp is an important lesson in applying clean and structured programming principles. Using OpenCV also enriches understanding in image manipulation, from conversion to grayscale, adding noise, to visualization and saving results. Another challenge emerged in the form of image artifacts due to incorrect settings of filter parameters and data types, which prompted further exploration of matrix operations in OpenCV. This entire process reinforces the understanding of the importance of preprocessing, modularity, and controlled experimentation in digital image processing.

## Project Repository
You can find the complete source code for this project on [GitHub](https://github.com/muhammadwildanskyyy/adaptive-Median-Filter_wiener-Filter_anisotropic-Diffusion.git).
