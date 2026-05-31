#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>

namespace cuda_filter
{

// CUDA error checking
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    // CUDA kernel for 2D convolution
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            // Clamp the result to [0, 255]
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel
        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        // Convert kernel to float array for faster access
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    // Apply kernel
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    // Clamp the result to [0, 255]
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }

    void applyHDRToneMappingCPU(const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation) {
      output.create(input.size(), input.type());
      for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {
          const cv::Vec3b &in = input.at<cv::Vec3b>(y, x);
            
          float b = in[0] / 255.0f * exposure;
          float g = in[1] / 255.0f * exposure;
          float r = in[2] / 255.0f * exposure;
            
          r = r / (1.0f + r);
          g = g / (1.0f + g);
          b = b / (1.0f + b);

          float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
          r = luminance + saturation * (r - luminance);
          g = luminance + saturation * (g - luminance);
          b = luminance + saturation * (b - luminance);

          r = powf(r, 1.0f / gamma);
          g = powf(g, 1.0f / gamma);
          b = powf(b, 1.0f / gamma);

          output.at<cv::Vec3b>(y, x)[0] = static_cast<unsigned char>(std::min(std::max(b * 255.0f, 0.0f), 255.0f));
          output.at<cv::Vec3b>(y, x)[1] = static_cast<unsigned char>(std::min(std::max(g * 255.0f, 0.0f), 255.0f));
          output.at<cv::Vec3b>(y, x)[2] = static_cast<unsigned char>(std::min(std::max(r * 255.0f, 0.0f), 255.0f));
        }
      }
    }

    __global__ void hdrToneMappingKernel(const unsigned char *input, unsigned char *output,
                                     int width, int height, int channels,
                                     float exposure, float gamma, float saturation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    float b = input[idx + 0] / 255.0f;
    float g = input[idx + 1] / 255.0f;
    float r = input[idx + 2] / 255.0f;

    r *= exposure; g *= exposure; b *= exposure;

    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);

    float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
    r = luminance + saturation * (r - luminance);
    g = luminance + saturation * (g - luminance);
    b = luminance + saturation * (b - luminance);

    r = powf(r, 1.0f / gamma);
    g = powf(g, 1.0f / gamma);
    b = powf(b, 1.0f / gamma);

    output[idx + 0] = static_cast<unsigned char>(min(max(b * 255.0f, 0.0f), 255.0f));
    output[idx + 1] = static_cast<unsigned char>(min(max(g * 255.0f, 0.0f), 255.0f));
    output[idx + 2] = static_cast<unsigned char>(min(max(r * 255.0f, 0.0f), 255.0f));
  }

  void applyHDRToneMappingGPU(HDRProcessor &proc, const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation)
  {
    output.create(input.size(), input.type());
    size_t imageSize = input.total() * input.elemSize();

    if (imageSize != proc.lastSize) {
        if (proc.d_input) cudaFree(proc.d_input);
        if (proc.d_output) cudaFree(proc.d_output);
        cudaMalloc(&proc.d_input, imageSize);
        cudaMalloc(&proc.d_output, imageSize);
        proc.lastSize = imageSize;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(proc.d_input, input.data, imageSize, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((input.cols + blockDim.x - 1) / blockDim.x, (input.rows + blockDim.y - 1) / blockDim.y);

    hdrToneMappingKernel<<<gridDim, blockDim>>>(proc.d_input, proc.d_output, input.cols, input.rows, input.channels(), exposure, gamma, saturation);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, proc.d_output, imageSize, cudaMemcpyDeviceToHost));
  }

} // namespace cuda_filter
