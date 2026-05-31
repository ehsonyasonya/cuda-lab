#pragma once

#include <opencv2/opencv.hpp>

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyHDRToneMappingCPU(const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation);

    class HDRProcessor {
    public:
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        size_t lastSize = 0;

        ~HDRProcessor() {
            if (d_input) cudaFree(d_input);
            if (d_output) cudaFree(d_output);
        }
    };

    void applyHDRToneMappingGPU(HDRProcessor &proc, const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter