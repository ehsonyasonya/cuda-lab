# CUDA University Labs

Repository for laboratory works on CUDA parallel computing.

---

## Labs List

### Lab 1: Matrix Transposition Optimization
* **Description:** Implementation of matrix transposition on CPU and GPU. The study analyzes the impact of Unified Memory (UM) with asynchronous prefetching on data migration latency and explores the limitations of global memory access patterns.
* **Source Code and PDF Report:** [Go to Lab1_Matrix_Transposition folder](./Lab1_Matrix_Transposition/)

### Lab 2: Image Convolution Filter Implementation
* **Description:** 2D image convolution implemented on CPU and GPU with memory optimizations (Constant Memory and Dynamic Shared Memory).
* **Source Code and PDF Report:** [Go to Lab2_Convolution folder](./Lab2_Convolution/)

### Lab 3: Real-Time HDR Tone Mapping
* **Description:** HDR tone mapping implementation with GPU acceleration. Adapted for static images to ensure accurate performance benchmarking in Google Colab.
* **Source Code and PDF Report:** [Go to Lab3_Tone_Mapping folder](./Lab3_Tone_Mapping/)

### Lab 4: CUDA Webcam Filter Pipeline
* **Description:** Design and implementation of a flexible, high-performance CUDA filter pipeline. Features include dynamic sequential filter chaining, custom Wipe Transitions, and asynchronous processing using CUDA streams. Includes a detailed performance analysis demonstrating L2 cache efficiency and effective GPU resource management.
* **Source Code and PDF Report:** [Go to Lab4_Webcam_Filter_Pipeline folder](./Lab4_Webcam_Filter_Pipeline/)

---

## Environment & How to Run
All projects are designed and tested within **Google Colab** using NVIDIA GPU architectures (T4 / A100).

To run any laboratory work from this repository:
1. Open the corresponding lab folder listed above.
2. Click on the `.ipynb` notebook file.
3. Click the **"Open in Colab"** button that automatically appears at the top of the file view on GitHub.

**Important Note for Code Access:**
If you wish to examine or modify the implementation details, please navigate to the **`src/`** folder located inside each laboratory directory. There you will find the modified CUDA kernels (`.cu`), headers (`.h`), and source files (`.cpp`) that contain the custom logic for each task.
