#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define CHECK_HIP_ERROR(val) { if(val != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(val) << std::endl; exit(-1); } }

// Matrix dimensions
constexpr int M = 512;  // Rows of A and C
constexpr int N = 512;  // Columns of B and C
constexpr int K = 512;  // Columns of A, Rows of B

// Block size for the kernel
constexpr int BLOCK_SIZE = 16;

// GPU kernel for matrix multiplication
__global__ void matMul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
        if (row < M && tileIdx * BLOCK_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * BLOCK_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tileIdx * BLOCK_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(tileIdx * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to validate GPU results against CPU results
bool validate(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, int M, int N, int K) {
    std::vector<float> C_cpu(M * N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C_cpu[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(C[i] - C_cpu[i]) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": GPU = " << C[i] << ", CPU = " << C_cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Output GPU info
    int device;
    hipDeviceProp_t deviceProp;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, device));
    std::cout << "Using GPU: " << deviceProp.name << std::endl;

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    // Allocate host memory
    std::vector<float> h_A(sizeA, 1.0f);  // Initialize A with 1.0
    std::vector<float> h_B(sizeB, 2.0f);  // Initialize B with 2.0
    std::vector<float> h_C(sizeC, 0.0f);  // Output C

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeA * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeB * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeC * sizeof(float)));

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeA * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), sizeB * sizeof(float), hipMemcpyHostToDevice));

    // Create HIP graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

    // Define kernel parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    void* kernelArgs[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    hipKernelNodeParams kernelNodeParams = {0};
    kernelNodeParams.func = reinterpret_cast<void*>(matMul);
    kernelNodeParams.gridDim = gridDim;
    kernelNodeParams.blockDim = blockDim;
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = nullptr;

    hipGraphNode_t kernelNode;
    CHECK_HIP_ERROR(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));

    // Instantiate graph
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "MatMul completed in " << duration.count() << " ms" << std::endl;

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost));

    // Validate results
    if (validate(h_A, h_B, h_C, M, N, K)) {
        std::cout << "Validation Passed!" << std::endl;
    } else {
        std::cout << "Validation Failed!" << std::endl;
    }

    // Memory usage
    size_t freeMem, totalMem;
    CHECK_HIP_ERROR(hipMemGetInfo(&freeMem, &totalMem));
    std::cout << "Memory: Free = " << freeMem / (1024.0 * 1024) << " MB, Total = " << totalMem / (1024.0 * 1024) << " MB" << std::endl;

    // Clean up
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    return 0;
}
