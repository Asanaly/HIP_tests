#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define M 1024  // Rows in matrix A
#define K 1024  // Columns in matrix A and rows in matrix B
#define P 1024  // Columns in matrix B
#define N 8     // Number of matrices in the batch

// Kernel for Batched Matrix Multiplication: C = A * B
__global__ void batchedMatMulKernel(float* A, float* B, float* C, int M, int K, int P, int N) {
    int batchIdx = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < P) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[batchIdx * M * K + x * K + i] * B[batchIdx * K * P + i * P + y];
        }
        C[batchIdx * M * P + x * P + y] = sum;
    }
}

int main() {
    // Allocate host memory for matrices A, B, and C
    std::vector<float> A(M * K * N, 1.0f);  // All values set to 1.0
    std::vector<float> B(K * P * N, 1.0f);  // All values set to 1.0
    std::vector<float> C(M * P * N, 0.0f);  // Initialize with 0.0

    // Device memory pointers
    float *A_d, *B_d, *C_d;
    (void)hipMalloc(&A_d, M * K * N * sizeof(float));
    (void)hipMalloc(&B_d, K * P * N * sizeof(float));
    (void)hipMalloc(&C_d, M * P * N * sizeof(float));

    // HIP streams and graphs
    hipStream_t stream;
    (void)hipStreamCreate(&stream);

    hipGraph_t graph;
    hipGraphExec_t graphExec;

    // Record graph build start time
    auto buildStart = std::chrono::high_resolution_clock::now();

    // Begin graph capture
    (void)hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

    // Transfer data to device
    (void)hipMemcpyAsync(A_d, A.data(), M * K * N * sizeof(float), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(B_d, B.data(), K * P * N * sizeof(float), hipMemcpyHostToDevice, stream);

    // Set up kernel dimensions
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, 
                  (P + blockSize.y - 1) / blockSize.y, 
                  N);  // N matrices in the batch

    // Launch the kernel
    hipLaunchKernelGGL(batchedMatMulKernel, gridSize, blockSize, 0, stream, A_d, B_d, C_d, M, K, P, N);

    // Transfer result back to host
    (void)hipMemcpyAsync(C.data(), C_d, M * P * N * sizeof(float), hipMemcpyDeviceToHost, stream);

    // End graph capture
    (void)hipStreamEndCapture(stream, &graph);

    // Instantiate the graph
    (void)hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Record graph build end time
    auto buildEnd = std::chrono::high_resolution_clock::now();

    // Launch the graph and measure execution time
    auto execStart = std::chrono::high_resolution_clock::now();
    (void)hipGraphLaunch(graphExec, stream);
    (void)hipStreamSynchronize(stream);
    auto execEnd = std::chrono::high_resolution_clock::now();

    // Calculate times
    std::chrono::duration<double> buildTime = buildEnd - buildStart;
    std::chrono::duration<double> execTime = execEnd - execStart;

    // Print performance results
    std::cout << "Graph build time: " << buildTime.count() << " seconds\n";
    std::cout << "Graph execution time: " << execTime.count() << " seconds\n";

    // Throughput (operations per second)
    long long totalOps = (long long)M * P * N * K;  // Operations for matrix multiplication
    std::cout << "Throughput: " << totalOps / execTime.count() / 1e9 << " GOPS\n";

    // Memory usage
    std::cout << "Memory usage: "
              << (M * K * N + K * P * N + M * P * N) * sizeof(float) / (1024.0 * 1024.0 * 1024.0)  // GB
              << " GB\n";

    // Clean up
    (void)hipGraphExecDestroy(graphExec);
    (void)hipGraphDestroy(graph);
    (void)hipStreamDestroy(stream);
    (void)hipFree(A_d);
    (void)hipFree(B_d);
    (void)hipFree(C_d);

    return 0;
}
