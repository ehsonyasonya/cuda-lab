#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace cuda_filter {

    // Manages GPU memory and execution streams for the filter pipeline.
    //Handles buffer allocation/resizing and lifecycle of CUDA resources.
    class PipelineManager {
    public:
        unsigned char *d_buf1 = nullptr;
        unsigned char *d_buf2 = nullptr;
        size_t bufferSize = 0;
        cudaStream_t stream; // Stream for asynchronous GPU operations

        PipelineManager() {
            cudaStreamCreate(&stream);
        }

        ~PipelineManager() {
            if (d_buf1) cudaFree(d_buf1);
            if (d_buf2) cudaFree(d_buf2);
            cudaStreamDestroy(stream);
        }

        //Ensures GPU buffers are allocated for the given image size.
        void ensureBufferSize(size_t size) {
            if (size != bufferSize) {
                if (d_buf1) cudaFree(d_buf1);
                if (d_buf2) cudaFree(d_buf2);
                cudaMalloc(&d_buf1, size);
                cudaMalloc(&d_buf2, size);
                bufferSize = size;
            }
        }
    };

    //GPU-accelerated filter functions.
    void applyFilterGPU(PipelineManager &pm, const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyHDRToneMappingGPU(PipelineManager &pm, const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation);
    
    // Part 2: Wipe Transition
    void applyWipeTransitionGPU(PipelineManager &pm, const cv::Mat &frame1, const cv::Mat &frame2, cv::Mat &output, float progress);

    //Part 3: Instrumentation helper to measure execution time.
    struct ScopedTimer {
        cudaEvent_t start, stop;
        const char* name;
        ScopedTimer(const char* n, cudaStream_t stream = 0) : name(n) {
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start, stream);
        }
        ~ScopedTimer() {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            // Assuming plog is configured globally
            printf("Filter [%s] execution time: %.3f ms\n", name, ms);
            cudaEventDestroy(start); cudaEventDestroy(stop);
        }
    };

} // namespace cuda_filter