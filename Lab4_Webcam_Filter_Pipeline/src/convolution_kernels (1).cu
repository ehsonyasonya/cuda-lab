#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <algorithm>

namespace cuda_filter
{

#define CHECK_CUDA_ERROR(call)                                                 \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                            \
        }                                                                      \
    }

    // Existing convolution kernel
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        int radius = kernelSize / 2;
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];
                    sum += pixelValue * kernelValue;
                }
            }
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    // Existing wipe transition kernel
    __global__ void wipeTransitionKernel(const unsigned char *src1, const unsigned char *src2, 
                                        unsigned char *output, float progress, int width, int height, int channels) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        
        int idx = (y * width + x) * channels;
        float splitPoint = progress * width;
        
        for(int c = 0; c < channels; c++) {
            output[idx + c] = (x < splitPoint) ? src2[idx + c] : src1[idx + c];
        }
    }

    // GPU kernel for HDR Tone Mapping
    __global__ void hdrToneMappingKernel(const unsigned char *input, unsigned char *output,
                                        int width, int height, int channels, 
                                        float exposure, float gamma, float saturation) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;

        for(int c = 0; c < channels; c++) {
            float val = input[idx + c] / 255.0f;
            val = powf(val * exposure, 1.0f / gamma) * saturation;
            output[idx + c] = (unsigned char)(min(max(val * 255.0f, 0.0f), 255.0f));
        }
    }

    // GPU function for standard convolution filters using async stream
    void applyFilterGPU(PipelineManager &pm, const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel) {
        ScopedTimer timer("Convolution", pm.stream);
        size_t imageSize = input.total() * input.elemSize();
        pm.ensureBufferSize(imageSize);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(pm.d_buf1, input.data, imageSize, cudaMemcpyHostToDevice, pm.stream));

        float *d_kernel;
        size_t kSize = kernel.rows * kernel.cols * sizeof(float);
        cudaMalloc(&d_kernel, kSize);
        cudaMemcpyAsync(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice, pm.stream);

        dim3 block(16, 16);
        dim3 grid((input.cols + 15) / 16, (input.rows + 15) / 16);
        
        convolutionKernel<<<grid, block, 0, pm.stream>>>(pm.d_buf1, pm.d_buf2, d_kernel, input.cols, input.rows, input.channels(), kernel.rows);
        
        cudaFreeAsync(d_kernel, pm.stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data, pm.d_buf2, imageSize, cudaMemcpyDeviceToHost, pm.stream));
    }

    // GPU function for Wipe Transition using async stream
    void applyWipeTransitionGPU(PipelineManager &pm, const cv::Mat &frame1, const cv::Mat &frame2, cv::Mat &output, float progress) {
        ScopedTimer timer("WipeTransition", pm.stream);
        size_t imageSize = frame1.total() * frame1.elemSize();
        pm.ensureBufferSize(imageSize);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(pm.d_buf1, frame1.data, imageSize, cudaMemcpyHostToDevice, pm.stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(pm.d_buf2, frame2.data, imageSize, cudaMemcpyHostToDevice, pm.stream));

        dim3 block(16, 16);
        dim3 grid((frame1.cols + 15) / 16, (frame1.rows + 15) / 16);

        wipeTransitionKernel<<<grid, block, 0, pm.stream>>>(pm.d_buf1, pm.d_buf2, pm.d_buf1, progress, frame1.cols, frame1.rows, frame1.channels());

        CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data, pm.d_buf1, imageSize, cudaMemcpyDeviceToHost, pm.stream));
    }

    // GPU function for HDR Tone Mapping using async stream
    void applyHDRToneMappingGPU(PipelineManager &pm, const cv::Mat &input, cv::Mat &output, float exposure, float gamma, float saturation) {
        ScopedTimer timer("HDRToneMapping", pm.stream);
        size_t imageSize = input.total() * input.elemSize();
        pm.ensureBufferSize(imageSize);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(pm.d_buf1, input.data, imageSize, cudaMemcpyHostToDevice, pm.stream));

        dim3 blockDim(16, 16);
        dim3 gridDim((input.cols + blockDim.x - 1) / blockDim.x, (input.rows + blockDim.y - 1) / blockDim.y);

        hdrToneMappingKernel<<<gridDim, blockDim, 0, pm.stream>>>(pm.d_buf1, pm.d_buf2, input.cols, input.rows, input.channels(), exposure, gamma, saturation);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data, pm.d_buf2, imageSize, cudaMemcpyDeviceToHost, pm.stream));
    }

} // namespace cuda_filter