#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#define CHECK_HIP_ERROR(val) { if(val != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(val) << std::endl; exit(-1); } }

constexpr size_t N = 1 << 20; // Vector size (1M elements)

// Kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel for vector multiplication
__global__ void vectorMul(const float* A, const float* B, float* C, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

// Function to run a test
void runTest(const std::string& testName, hipGraph_t graph, hipGraphExec_t graphExec, hipStream_t stream) {
    std::cout << "Running Test: " << testName << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << "Test: " << testName << " completed in " << duration.count() << " ms" << std::endl;
}

int main() {
    // Output GPU info
    int device;
    hipDeviceProp_t deviceProp;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, device));
    std::cout << "Using GPU: " << deviceProp.name << std::endl;

    // Allocate host memory
    std::vector<float> h_A(N, 1.5f);  // Initialize with 1.5
    std::vector<float> h_B(N, 2.0f);  // Initialize with 2.0
    std::vector<float> h_C(N, 0.0f);

    float *d_A, *d_B, *d_C, *d_D;

    // Allocate device memory
    CHECK_HIP_ERROR(hipMalloc(&d_A, N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, N * sizeof(float)));  // For addition result
    CHECK_HIP_ERROR(hipMalloc(&d_D, N * sizeof(float)));  // For multiplication result

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
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    // Add vector addition kernel to graph
    void* addKernelArgs[] = {&d_A, &d_B, &d_C, (void*)&N};
    hipKernelNodeParams addKernelParams = {0};
    addKernelParams.func = reinterpret_cast<void*>(vectorAdd);
    addKernelParams.gridDim = gridDim;
    addKernelParams.blockDim = blockDim;
    addKernelParams.sharedMemBytes = 0;
    addKernelParams.kernelParams = addKernelArgs;
    addKernelParams.extra = nullptr;

    hipGraphNode_t addKernelNode;
    CHECK_HIP_ERROR(hipGraphAddKernelNode(&addKernelNode, graph, nullptr, 0, &addKernelParams));

    // Add vector multiplication kernel to graph
    void* mulKernelArgs[] = {&d_A, &d_B, &d_D, (void*)&N};
    hipKernelNodeParams mulKernelParams = {0};
    mulKernelParams.func = reinterpret_cast<void*>(vectorMul);
    mulKernelParams.gridDim = gridDim;
    mulKernelParams.blockDim = blockDim;
    mulKernelParams.sharedMemBytes = 0;
    mulKernelParams.kernelParams = mulKernelArgs;
    mulKernelParams.extra = nullptr;

    hipGraphNode_t mulKernelNode;
    CHECK_HIP_ERROR(hipGraphAddKernelNode(&mulKernelNode, graph, nullptr, 0, &mulKernelParams));

    // Instantiate graph
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Run and measure tests
    runTest("Vector Addition", graph, graphExec, stream);

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_D, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify multiplication results
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] * h_B[i];
        if (h_C[i] != expected) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": " << h_C[i] << " != " << expected << std::endl;
            break;
        }
    }
    std::cout << "Multiplication Results: " << (correct ? "Correct!" : "Incorrect!") << std::endl;

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
    CHECK_HIP_ERROR(hipFree(d_D));

    return 0;
}
