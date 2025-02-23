#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_HIP_ERROR(err)                                                  \
    if (err != hipSuccess) {                                                  \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line "   \
                  << __LINE__ << std::endl;                                   \
        exit(-1);                                                             \
    }

#define DEBUG_PRINT(msg) std::cout << msg << std::endl;

__global__ void heavy_computation(const float *input1, const float *input2, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = input1[idx] + input2[idx];
        for (int i = 0; i < 500; ++i) {  // Simulate heavy computation
            value = value * 0.98f + 0.02f;
        }
        output[idx] = value;
    }
}

void allocateAndValidate(void **ptr, size_t size) {
    hipError_t err = hipMalloc(ptr, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory: " << hipGetErrorString(err) << std::endl;
        exit(-1);
    }
}

void measureExecutionTime(hipGraphExec_t graphExec, hipStream_t stream, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        double execTime = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Iteration " << i + 1 << " execution time: " << execTime << " ms" << std::endl;
    }
}

int main() {
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    const int width = 1024, height = 1024, dataSize = width * height;
    float *d_input1 = nullptr, *d_input2 = nullptr, *d_output = nullptr;

    allocateAndValidate(reinterpret_cast<void **>(&d_input1), dataSize * sizeof(float));
    allocateAndValidate(reinterpret_cast<void **>(&d_input2), dataSize * sizeof(float));
    allocateAndValidate(reinterpret_cast<void **>(&d_output), dataSize * sizeof(float));

    std::vector<float> h_input1(dataSize, 1.0f);
    std::vector<float> h_input2(dataSize, 2.0f);
    CHECK_HIP_ERROR(hipMemcpy(d_input1, h_input1.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_input2, h_input2.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));

    // Record the graph
    hipGraph_t graph;
    CHECK_HIP_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hipLaunchKernelGGL(heavy_computation, numBlocks, threadsPerBlock, 0, stream, d_input1, d_input2, d_output, width, height);

    CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));

    // Instantiate and execute the graph
    hipGraphExec_t graphExec;
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    measureExecutionTime(graphExec, stream, 5);

    // Clean up
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipFree(d_input1));
    CHECK_HIP_ERROR(hipFree(d_input2));
    CHECK_HIP_ERROR(hipFree(d_output));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
