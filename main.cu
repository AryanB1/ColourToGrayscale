#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

__global__ void rgbToGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = 0.3f * r + 0.6f * g + 0.1f * b;
    }
}

int main(std::string inputFile) {
    // Load image using OpenCV
    cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cout << "Image not found" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    size_t colorBytes = width * height * channels * sizeof(unsigned char);
    size_t grayBytes = width * height * sizeof(unsigned char);

    unsigned char *h_input = inputImage.data;
    unsigned char *h_output = new unsigned char[width * height];

    // Allocate memory on device
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, colorBytes);
    cudaMalloc(&d_output, grayBytes);

    // Copy input to device
    cudaMemcpy(d_input, h_input, colorBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Launch kernel
    rgbToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, grayBytes, cudaMemcpyDeviceToHost);

    // Save the grayscale image
    cv::Mat outputImage(height, width, CV_8UC1, h_output);
    cv::imwrite("output_gpu.jpg", outputImage);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    std::cout << "Grayscale conversion complete! Saved as output_gpu.jpg" << std::endl;
    return 0;
}
