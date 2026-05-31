#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Pre-defined filters for the assignment
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1};

float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1};

float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

// Macro to automatically catch and report CUDA errors
#define CHECK_CUDA_ERROR(call)                                                 \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n",          \
                    cudaGetErrorString(err), __LINE__, __FILE__);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Structure to store image dimensions and pixel data
typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels; 
} Image;


// CPU SEQUENTIAL IMPLEMENTATION
void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    int width = input->width;
    int height = input->height;
    int radius = filterWidth / 2; // Distance from center to the edge of the filter

    // Loop through every single pixel of the image (Row by Row)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;

            // Loop through the convolution filter window
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    
                    // Get coordinates of the neighboring pixel
                    int pixelY = y + ky;
                    int pixelX = x + kx;

                    // BOUNDARY HANDLING (Clamping): If out of bounds, use the edge pixel
                    if (pixelY < 0) pixelY = 0;
                    if (pixelY >= height) pixelY = height - 1;
                    if (pixelX < 0) pixelX = 0;
                    if (pixelX >= width) pixelX = width - 1;

                    // Convert 2D coordinates to 1D array indices
                    int imageIndex = pixelY * width + pixelX;
                    int filterIndex = (ky + radius) * filterWidth + (kx + radius);

                    // Multiply pixel value by filter weight and add to total sum
                    sum += (float)input->data[imageIndex] * filter[filterIndex];
                }
            }

            // Clamping the final value to valid pixel range [0, 255]
            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;
            
            // Save the calculated pixel to the output image
            output->data[y * width + x] = (unsigned char)sum;
        }
    }
}
//NAIVE GPU IMPLEMENTATION
__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       float *filter, int filterWidth,
                                       int width, int height, int channels)
{
    // Calculate global 2D pixel coordinates (x, y) for this specific GPU thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard: ensure the thread doesn't process outside the actual image
    if (x < width && y < height) {
        int radius = filterWidth / 2;
        float sum = 0.0f;

        // Slide the filter window over the current pixel
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                
                int pixelY = y + ky;
                int pixelX = x + kx;

                // Boundary handling (Clamping)
                if (pixelY < 0) pixelY = 0;
                if (pixelY >= height) pixelY = height - 1;
                if (pixelX < 0) pixelX = 0;
                if (pixelX >= width) pixelX = width - 1;

                // 1D indexing calculation taking channels into account
                int imageIndex = (pixelY * width + pixelX) * channels;
                int filterIndex = (ky + radius) * filterWidth + (kx + radius);

                // Accumulate the sum
                sum += (float)input[imageIndex] * filter[filterIndex];
            }
        }

        // Clamp the result to [0, 255]
        if (sum < 0.0f) sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;

        // Write the final pixel directly into GPU global memory
        output[(y * width + x) * channels] = (unsigned char)sum;
    }
}

__constant__ float d_filter[81]; 

//CONSTANT MEMORY OPTIMIZATION
__global__ void convolutionKernelConstant(unsigned char *input, unsigned char *output,
                                         int filterWidth, int width, int height, int channels)
{
    // Calculate global 2D pixel coordinates (x, y) for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard
    if (x < width && y < height) {
        int radius = filterWidth / 2;
        float sum = 0.0f;

        // Slide the filter window
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                
                int pixelY = y + ky;
                int pixelX = x + kx;

                // Boundary handling (Clamping)
                if (pixelY < 0) pixelY = 0;
                if (pixelY >= height) pixelY = height - 1;
                if (pixelX < 0) pixelX = 0;
                if (pixelX >= width) pixelX = width - 1;

                int imageIndex = (pixelY * width + pixelX) * channels;
                int filterIndex = (ky + radius) * filterWidth + (kx + radius);

                // OPTIMIZATION: Reading from fast __constant__ d_filter instead of global memory
                sum += (float)input[imageIndex] * d_filter[filterIndex];
            }
        }

        // Clamp the result to [0, 255]
        if (sum < 0.0f) sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;

        // Write to output
        output[(y * width + x) * channels] = (unsigned char)sum;
    }
}

size_t getSharedMemorySize(int blockDimX, int blockDimY, int filterWidth) 
{
    int radius = filterWidth / 2;
    int sharedWidth = blockDimX + 2 * radius;
    int sharedHeight = blockDimY + 2 * radius;
    
    // Returns total bytes needed for the tile
    return sharedWidth * sharedHeight * sizeof(unsigned char);
}

// Placeholder for future assignment parts
__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        float *filter, int filterWidth,
                                        int width, int height, int channels)
{
  // GPU takes the exact dynamic size calculated by getSharedMemorySize()
    extern __shared__ unsigned char sharedTile[];

    int radius = filterWidth / 2;
    
    // Dynamic dimensions for traversing the shared array
    int sharedWidth = blockDim.x + 2 * radius;
    int sharedHeight = blockDim.y + 2 * radius;

    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x + radius;
    int localY = threadIdx.y + radius;

    // Cooperative loading of the tile and its halo (apron) pixels
    for (int chY = threadIdx.y; chY < sharedHeight; chY += blockDim.y) {
        for (int chX = threadIdx.x; chX < sharedWidth; chX += blockDim.x) {
            
            int gX = blockIdx.x * blockDim.x + chX - radius;
            int gY = blockIdx.y * blockDim.y + chY - radius;

            // Boundary handling (Clamping)
            if (gY < 0) gY = 0;
            if (gY >= height) gY = height - 1;
            if (gX < 0) gX = 0;
            if (gX >= width) gX = width - 1;

            int sharedIndex = chY * sharedWidth + chX;
            sharedTile[sharedIndex] = input[(gY * width + gX) * channels];
        }
    }

    // Wait until the entire block finishes loading its shared tile
    __syncthreads();

    // Compute convolution using ultra-fast Shared Memory
    if (globalX < width && globalY < height) {
        float sum = 0.0f;

        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                
                int sharedIndex = (localY + ky) * sharedWidth + (localX + kx);
                unsigned char pixelValue = sharedTile[sharedIndex];
                
                int filterIndex = (ky + radius) * filterWidth + (kx + radius);
                // Combining Shared Memory with Constant Memory (d_filter)
                sum += (float)pixelValue * d_filter[filterIndex];
            }
        }

        if (sum < 0.0f) sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;

        output[(globalY * width + globalX) * channels] = (unsigned char)sum;
    }
}

int main(int argc, char **argv)
{
  int width = 2048;
  int height = 2048;
  int channels = 1; 
  int filterWidth = 5; 

  printf("==================================================\n");
  printf("   CUDA Convolution Lab - Performance Evaluation  \n");
  printf("==================================================\n");
  printf("Image Size: %dx%d | Filter Width: %d\n\n", width, height, filterWidth);

  // Allocate Host (CPU) memory for images
  Image hostInput, hostOutputCPU, hostOutputGPU;
  hostInput.width = width; hostInput.height = height; hostInput.channels = channels;
  hostInput.data = (unsigned char*)malloc(width * height * channels);
  hostOutputCPU.width = width; hostOutputCPU.height = height; hostOutputCPU.channels = channels;
  hostOutputCPU.data = (unsigned char*)malloc(width * height * channels);
  hostOutputGPU.width = width; hostOutputGPU.height = height; hostOutputGPU.channels = channels;
  hostOutputGPU.data = (unsigned char*)malloc(width * height * channels);

  // Fill the input image with random pixel values [0-255]
  for (int i = 0; i < width * height; i++) {
    hostInput.data[i] = rand() % 256;
  }

  // Variables for timing
  float timeCPU = 0.0f;
  float timeNaive = 0.0f;
  float timeConstant = 0.0f;
  float timeShared = 0.0f;

  // CUDA event timing setup
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  //CPU Execution
  cudaEventRecord(start, 0);
  convolutionCPU(&hostInput, &hostOutputCPU, gaussianBlur5x5, filterWidth);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeCPU, start, stop);
  printf("[-] CPU sequential execution completed.\n");

  // GPU Memory Setup
  unsigned char *deviceInput, *deviceOutput;
  float *deviceFilter;
  size_t imgSize = width * height * channels * sizeof(unsigned char);
  size_t filterSize = filterWidth * filterWidth * sizeof(float);

  // Allocate Device (GPU Global Memory) arrays
  CHECK_CUDA_ERROR(cudaMalloc((void**)&deviceInput, imgSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&deviceOutput, imgSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&deviceFilter, filterSize));

  // Copy data from Host (CPU) to Device (GPU)
  CHECK_CUDA_ERROR(cudaMemcpy(deviceInput, hostInput.data, imgSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(deviceFilter, gaussianBlur5x5, filterSize, cudaMemcpyHostToDevice));

  // Define CUDA thread block configuration (16x16 threads per block)
  dim3 blockSize(16, 16);
  // Calculate grid size to cover the entire image
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                (height + blockSize.y - 1) / blockSize.y);

  //GPU Naive Execution
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
  convolutionKernelNaive<<<gridSize, blockSize>>>(deviceInput, deviceOutput, deviceFilter, filterWidth, width, height, channels);
  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeNaive, start, stop));
  
  // Check for configuration errors and wait for GPU to finish execution
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaMemcpy(hostOutputGPU.data, deviceOutput, imgSize, cudaMemcpyDeviceToHost));
  printf("[-] GPU Naive execution completed.\n");

  //Verify correctness by comparing CPU and GPU results
  bool success = true;
  for (int i = 0; i < width * height; i++) {
    if (abs(hostOutputCPU.data[i] - hostOutputGPU.data[i]) > 1) {
      printf("Mismatch at index %d! CPU: %d, GPU: %d\n", i, hostOutputCPU.data[i], hostOutputGPU.data[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("SUCCESS! CPU and GPU results match using gaussianBlur5x5!\n");
  } else {
    printf("ERROR: Results do not match!\n");
  }

  // Allocate host memory to store Constant GPU results for verification
  Image hostOutputGPU_Const;
  hostOutputGPU_Const.width = width; hostOutputGPU_Const.height = height; hostOutputGPU_Const.channels = channels;
  hostOutputGPU_Const.data = (unsigned char*)malloc(imgSize);

  // OPTIMIZATION: Copy filter weights from Host to GPU Constant Memory
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter, gaussianBlur5x5, filterSize));

  //GPU Constant Memory Execution
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
  convolutionKernelConstant<<<gridSize, blockSize>>>(deviceInput, deviceOutput, filterWidth, width, height, channels);
  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeConstant, start, stop));
  CHECK_CUDA_ERROR(cudaGetLastError());

  // Copy result back to CPU
  CHECK_CUDA_ERROR(cudaMemcpy(hostOutputGPU_Const.data, deviceOutput, imgSize, cudaMemcpyDeviceToHost));
  printf("[-] GPU Constant Memory execution completed.\n");

  // Verify correctness of Constant Memory implementation
  bool constSuccess = true;
  for (int i = 0; i < width * height; i++) {
    if (abs(hostOutputCPU.data[i] - hostOutputGPU_Const.data[i]) > 1) {
      printf("Constant Memory Mismatch at index %d! CPU: %d, GPU: %d\n", i, hostOutputCPU.data[i], hostOutputGPU_Const.data[i]);
      constSuccess = false;
      break;
    }
  }

  if (constSuccess) {
    printf("SUCCESS! Constant Memory results match using gaussianBlur5x5!\n");
  } else {
    printf("ERROR: Constant Memory results do not match!\n");
  }

  //GPU Shared Memory version (Clean Dynamic Setup)
  Image hostOutputGPU_Shared;
  hostOutputGPU_Shared.width = width; hostOutputGPU_Shared.height = height; hostOutputGPU_Shared.channels = channels;
  hostOutputGPU_Shared.data = (unsigned char*)malloc(imgSize);

  // 1. CALCULATE TILE SIZE using our clean utility function
  size_t sharedMemSize = getSharedMemorySize(blockSize.x, blockSize.y, filterWidth);

  //GPU Shared Memory Execution
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
  convolutionKernelShared<<<gridSize, blockSize, sharedMemSize>>>(deviceInput, deviceOutput, deviceFilter, filterWidth, width, height, channels);
  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeShared, start, stop));
    
  CHECK_CUDA_ERROR(cudaGetLastError());

  // Copy result back to CPU
  CHECK_CUDA_ERROR(cudaMemcpy(hostOutputGPU_Shared.data, deviceOutput, imgSize, cudaMemcpyDeviceToHost));
  printf("[-] GPU Shared Memory execution completed.\n");

  // Verify correctness
  bool sharedSuccess = true;
  for (int i = 0; i < width * height; i++) {
    if (abs(hostOutputCPU.data[i] - hostOutputGPU_Shared.data[i]) > 1) {
      printf("Shared Memory Mismatch at index %d! CPU: %d, GPU: %d\n", i, hostOutputCPU.data[i], hostOutputGPU_Shared.data[i]);
      sharedSuccess = false;
      break;
    }
  }

  if (sharedSuccess) {
    printf("SUCCESS! Shared Memory results match using gaussianBlur5x5!\n");
  } else {
    printf("ERROR: Results do not match!\n");
  }

  printf("==================================================\n");
  printf("   PERFORMANCE SUMMARY (Time in Milliseconds)     \n");
  printf("==================================================\n");
  printf(" CPU Sequential      : %10.4f ms\n", timeCPU);
  printf(" GPU Naive           : %10.4f ms  (Speedup: x%.2f)\n", timeNaive, timeCPU / timeNaive);
  printf(" GPU Constant Memory : %10.4f ms  (Speedup: x%.2f)\n", timeConstant, timeCPU / timeConstant);
  printf(" GPU Shared Memory   : %10.4f ms  (Speedup: x%.2f)\n", timeShared, timeCPU / timeShared);
  printf("==================================================\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceFilter);
  free(hostInput.data);
  free(hostOutputCPU.data);
  free(hostOutputGPU.data);

  return 0;
}
