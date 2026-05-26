# CUDA University Labs

Repository for laboratory works on CUDA parallel computing.

---

## Labs List

### Lab 1: Matrix Transposition Optimization
* **Description:** Implementation of matrix transposition on CPU and GPU. The study analyzes the impact of Unified Memory (UM) with asynchronous prefetching on data migration latency and explores the limitations of global memory access patterns.
* **Source Code and PDF Report:** [Go to Lab1_Transpose folder](./Lab1_Matrix_Transposition/)

### Lab 2: Image Convolution Filter Implementation
* **Description:** 2D image convolution implemented on CPU and GPU with memory optimizations (Constant Memory and Dynamic Shared Memory).
* **Source Code and PDF Report:** [Go to Lab2_Convolution folder](./Lab2_Convolution/)

---

## Environment & How to Run
All projects are designed and tested within **Google Colab** using NVIDIA GPU architectures (T4 / A100).

To run any laboratory work from this repository:
1. Open the corresponding lab folder listed above.
2. Click on the `.ipynb` notebook file.
3. Click the **"Open in Colab"** button that automatically appears at the top of the file view on GitHub.
