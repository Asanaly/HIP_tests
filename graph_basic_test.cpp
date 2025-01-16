#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

#define N 1024  // Matrix size

// Kernel to perform element-wise addition
__global__ void add_kernel(float* A, float* B, float* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel to perform element-wise multiplication
__global__ void mul_kernel(float* A, float* B, float* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize matrices with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    // Copy data to device
    hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);

    // Create a stream and graph
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipGraph_t graph;
    hipGraphCreate(&graph, 0);

    // Kernel nodes for addition and multiplication
    hipKernelNodeParams addParams = {};
    addParams.func = (void*)add_kernel;
    addParams.gridDim = dim3(N / 256);
    addParams.blockDim = dim3(256);
    addParams.kernelParams = (void**)&d_A;

    hipGraphAddKernelNode(nullptr, graph, nullptr, 0, &addParams);

    hipKernelNodeParams mulParams = {};
    mulParams.func = (void*)mul_kernel;
    mulParams.gridDim = dim3(N / 256);
    mulParams.blockDim = dim3(256);
    mulParams.kernelParams = (void**)&d_B;

    hipGraphAddKernelNode(nullptr, graph, nullptr, 0, &mulParams);

    // Execute the graph and measure time
    auto start = std::chrono::high_resolution_clock::now();
    hipGraphLaunch(graph, stream);
    hipStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> exec_time = end - start;
    std::cout << "Graph execution time: " << exec_time.count() << " seconds" << std::endl;

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);

    return 0;
}
