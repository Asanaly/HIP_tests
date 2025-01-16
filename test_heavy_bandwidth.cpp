#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

// Helper macros
#define CHECK_HIP_ERROR(err) if (err != hipSuccess) { \
    std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
    exit(-1); \
}

#define PRINT_BANDWIDTH(bytes, ms) std::cout << "Bandwidth: " << (bytes / (ms * 1e6)) << " GB/s" << std::endl;

// Kernel for heavy computation
__global__ void heavy_computation(const float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = input[idx];
        for (int i = 0; i < 100; ++i) {  // Simulate heavy computation
            value = value * 0.99f + 0.01f;
        }
        output[idx] = value;
    }
}

// Measure time helper
double getElapsedTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// Memory usage helper
void printMemoryUsage() {
    size_t freeMem, totalMem;
    CHECK_HIP_ERROR(hipMemGetInfo(&freeMem, &totalMem));
    std::cout << "Memory Usage: Free = " << (freeMem / 1024.0 / 1024.0) << " MB, Total = " << (totalMem / 1024.0 / 1024.0) << " MB\n";
}

void executeGraph(hipGraphExec_t graphExec, hipStream_t stream, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
    }
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
}

int main() {
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Scaling parameters
    const int sizes[] = {512, 1024, 2048};
    const int iterations = 10;

    for (int size : sizes) {
        const int width = size;
        const int height = size;
        const int dataSize = width * height;

        // Allocate memory
        float *d_input, *d_output;
        CHECK_HIP_ERROR(hipMalloc(&d_input, dataSize * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_output, dataSize * sizeof(float)));

        // Initialize data
        std::vector<float> h_input(dataSize, 1.0f);
        CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));

        // Record the graph
        hipGraph_t graph;
        CHECK_HIP_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        hipLaunchKernelGGL(heavy_computation, numBlocks, threadsPerBlock, 0, stream, d_input, d_output, width, height);

        CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));

        // Dynamic modification: Add another kernel
        hipGraphNode_t newNode;
        hipKernelNodeParams params = {};
        params.func = reinterpret_cast<void *>(heavy_computation);
        params.gridDim = numBlocks;
        params.blockDim = threadsPerBlock;
        params.sharedMemBytes = 0;
        params.kernelParams = reinterpret_cast<void **>(&d_output);
        params.extra = nullptr;
        CHECK_HIP_ERROR(hipGraphAddKernelNode(&newNode, graph, nullptr, 0, &params));

        // Instantiate graph
        hipGraphExec_t graphExec;
        CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        executeGraph(graphExec, stream, iterations);
        auto end = std::chrono::high_resolution_clock::now();

        // Output performance metrics
        double execTimeMs = getElapsedTime(start, end);
        std::cout << "Size: " << size << "x" << size << ", Execution Time: " << execTimeMs << " ms\n";
        PRINT_BANDWIDTH(dataSize * sizeof(float) * 2, execTimeMs);

        // Print memory usage
        printMemoryUsage();

        // Clean up
        CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
        CHECK_HIP_ERROR(hipGraphDestroy(graph));
        CHECK_HIP_ERROR(hipFree(d_input));
        CHECK_HIP_ERROR(hipFree(d_output));
    }

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
