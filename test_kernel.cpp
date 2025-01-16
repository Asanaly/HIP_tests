#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_HIP_ERROR(val) { if(val != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(val) << std::endl; exit(-1); } }

constexpr size_t N = 1 << 20; // Vector size (1M elements)

__global__ void vectorAdd(const float* A, const float* B, float* C, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Allocate host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CHECK_HIP_ERROR(hipMalloc(&d_A, N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, N * sizeof(float)));

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Create HIP graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

    // Define kernel parameters
    void* kernelArgs[] = {&d_A, &d_B, &d_C, (void*)&N};
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Add kernel to graph
    hipKernelNodeParams kernelNodeParams = {0};
    kernelNodeParams.func = reinterpret_cast<void*>(vectorAdd);
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

    std::cout << "Graph execution time: " << duration.count() << " ms" << std::endl;

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "Results are correct!" << std::endl;
    }

    // Measure memory usage
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
