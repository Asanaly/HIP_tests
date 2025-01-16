#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Kernel for element-wise addition
__global__ void addKernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel for element-wise scaling
__global__ void scaleKernel(float* C, float* D, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = C[idx] * scale;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements per batch
    const int numBatches = 5; // Number of batches
    const size_t dataSize = N * sizeof(float);

    // Host allocations
    std::vector<float> A_host(N * numBatches, 1.0f); // Fill with 1.0
    std::vector<float> B_host(N * numBatches, 2.0f); // Fill with 2.0
    std::vector<float> D_host(N * numBatches, 0.0f); // Output buffer

    // Device allocations
    float *A_device, *B_device, *C_device, *D_device;
    hipMalloc(&A_device, dataSize);
    hipMalloc(&B_device, dataSize);
    hipMalloc(&C_device, dataSize);
    hipMalloc(&D_device, dataSize);

    // HIP streams and events
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipGraph_t graph;
    hipGraphExec_t graphExec;

    // Begin capturing the graph
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

    // Transfer A and B to the device
    hipMemcpyAsync(A_device, A_host.data(), dataSize, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(B_device, B_host.data(), dataSize, hipMemcpyHostToDevice, stream);

    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    hipLaunchKernelGGL(addKernel, gridSize, blockSize, 0, stream, A_device, B_device, C_device, N);
    hipLaunchKernelGGL(scaleKernel, gridSize, blockSize, 0, stream, C_device, D_device, 2.0f, N);

    // Transfer D back to the host
    hipMemcpyAsync(D_host.data(), D_device, dataSize, hipMemcpyDeviceToHost, stream);

    // End capturing and instantiate the graph
    hipStreamEndCapture(stream, &graph);
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Record the start time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the graph multiple times
    for (int i = 0; i < numBatches; ++i) {
        hipGraphLaunch(graphExec, stream);
    }

    // Synchronize
    hipStreamSynchronize(stream);

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output performance metrics
    double throughput = (numBatches * N * 2) / elapsed.count() / 1e9; // 2 ops per element
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Throughput: " << throughput << " GFLOP/s\n";
    std::cout << "Memory processed: " << (numBatches * dataSize * 3) / 1e9 << " GB\n";

    // Cleanup
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    hipFree(A_device);
    hipFree(B_device);
    hipFree(C_device);
    hipFree(D_device);

    return 0;
}
