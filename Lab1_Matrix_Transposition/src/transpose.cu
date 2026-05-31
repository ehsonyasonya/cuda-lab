#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>



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

// CPU Sequential Matrix Transposition
void transposeCPU(float *in, float *out, int width, int height) {
  // Standard nested loops translating row-major indices to column-major indices
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out[x * height + y] = in[y * width + x];
        }
    }
}

// PART 1: Naive GPU Matrix Transposition
__global__ void transposeKernelNaive(float *in, float *out, int width, int height) {
    // Calculate global 2D coordinates for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to prevent threads from processing out-of-bounds padding
    if (x < width && y < height) {
      // Coalesced memory read (sequential), but uncoalesced memory write (strided)
        out[x * height + y] = in[y * width + x];
    }
}

// PART 2: Optimized GPU Matrix Transposition (Unified Memory Optimizations)
__global__ void transposeKernelOptimized(float *in, float *out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Core transposition arithmetic remains the same; optimized via Unified Memory Prefetching
        out[x * height + y] = in[y * width + x];
    }
}

// Verification function to compare CPU and GPU outputs for floating-point safety
bool verifyResults(float *cpuOut, float *gpuOut, int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(cpuOut[i] - gpuOut[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

// Main benchmarking orchestration function
void runExperiment(int width, int height, int blockX, int blockY) {
    int size = width * height;
    size_t bytes = size * sizeof(float);
    printf("==================================================\n");
    printf(" Experiment: %d x %d | Block Size: [%d, %d]\n", width, height, blockX, blockY);
    printf("==================================================\n");
    // Allocate standard host-side (CPU) staging buffers
    float *h_in = (float*)malloc(bytes);
    float *h_outCPU = (float*)malloc(bytes);
    float *h_outGPU = (float*)malloc(bytes);

    // Initialize the source matrix with random floating-point values
    for (int i = 0; i < size; ++i) {
        h_in[i] = (float)(rand() % 100) / 7.0f;
    }

    // Initialize CUDA events for high-precision GPU timing
    float timeCPU = 0.0f, timeNaive = 0.0f, timeOptimized = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 1. CPU Sequential Execution
    clock_t cpuStart = clock();
    transposeCPU(h_in, h_outCPU, width, height);
    clock_t cpuEnd = clock();
    timeCPU = (float)(cpuEnd - cpuStart) * 1000.0f / CLOCKS_PER_SEC;

    // --- STANDARD DEVICE ALLOCATION (For Naive GPU) ---
    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Calculate execution configurations (Grid and Block sizes)
    dim3 blockSize(blockX, blockY);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 2. GPU Naive Kernel Execution
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    transposeKernelNaive<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeNaive, start, stop));

    // Explicit copy operation back from Device to Host
    CHECK_CUDA_ERROR(cudaMemcpy(h_outGPU, d_out, bytes, cudaMemcpyDeviceToHost));
    // Validate Naive kernel mathematical accuracy
    bool naiveCorrect = verifyResults(h_outCPU, h_outGPU, size);

    // --- UNIFIED MEMORY ALLOCATION (For Optimized GPU) ---
    float *um_in, *um_out;
    CHECK_CUDA_ERROR(cudaMallocManaged(&um_in, bytes));
    CHECK_CUDA_ERROR(cudaMallocManaged(&um_out, bytes));

    // Populate Unified Memory buffer from host
    memcpy(um_in, h_in, bytes);
    // Advanced Optimization: Asymmetric device asynchronous prefetching hint
    int device = 0;
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(um_in, bytes, device, NULL));
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(um_out, bytes, device, NULL));
  
    // 3. GPU Optimized Kernel Execution
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    transposeKernelOptimized<<<gridSize, blockSize>>>(um_in, um_out, width, height);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeOptimized, start, stop));

    // Synchronize global pipeline before reading Unified Memory arrays on the host side
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    bool optCorrect = verifyResults(h_outCPU, um_out, size);
    printf(" CPU Time       : %10.4f ms\n", timeCPU);
    printf(" GPU Naive      : %10.4f ms  (Correct: %s)\n", timeNaive, naiveCorrect ? "YES" : "NO");
    printf(" GPU Optimized  : %10.4f ms  (Correct: %s)\n", timeOptimized, optCorrect ? "YES" : "NO");
    printf(" Speedup (Naive vs Opt): x%.2f\n\n", timeNaive / timeOptimized);
  
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(um_in);
    cudaFree(um_out);
    free(h_in);
    free(h_outCPU);
    free(h_outGPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



int main() {
    // Seed pseudo-random number generator for reproducible verification tests
    srand(42);
    runExperiment(2048, 1024, 16, 16);
    runExperiment(4096, 2048, 16, 16);
    runExperiment(4096, 2048, 32, 32);
    runExperiment(4096, 2048, 32, 8);

    return 0;
}
