#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void conv2d(const float* input, const float* kernel, float* output, 
                       int inputWidth, int inputHeight, 
                       int kernelWidth, int kernelHeight, 
                       int outputWidth, int outputHeight) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX < outputWidth && outputY < outputHeight) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int inputX = outputX + kx;
                int inputY = outputY + ky;
                sum += input[inputY * inputWidth + inputX] * kernel[ky * kernelWidth + kx];
            }
        }
        output[outputY * outputWidth + outputX] = sum;
    }
}

void runConvolutionComparison() {
    const int inputWidth = 64 * 2, inputHeight = 64 * 2;
    const int kernelWidth = 3 * 2, kernelHeight = 3 * 2;
    const int outputWidth = inputWidth - kernelWidth + 1;
    const int outputHeight = inputHeight - kernelHeight + 1;
    const int numKernels = 50;

    const size_t inputSize = inputWidth * inputHeight * sizeof(float);
    const size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    const size_t outputSize = outputWidth * outputHeight * sizeof(float);

    std::vector<float> h_input(inputWidth * inputHeight, 1.0f);
    std::vector<float> h_kernel(kernelWidth * kernelHeight, 1.0f);
    std::vector<std::vector<float>> h_outputs(numKernels, std::vector<float>(outputWidth * outputHeight, 0.0f));

    float *d_input, *d_kernel;
    std::vector<float*> d_outputs(numKernels);

    // Allocate memory
    CHECK_HIP_ERROR(hipMalloc(&d_input, inputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_kernel, kernelSize));
    for (int i = 0; i < numKernels; i++) {
        CHECK_HIP_ERROR(hipMalloc(&d_outputs[i], outputSize));
    }

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputSize, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_kernel, h_kernel.data(), kernelSize, hipMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, 
                 (outputHeight + blockDim.y - 1) / blockDim.y);

    // Non-graph execution
    float nonGraphTotalTime = 0.0f;
    auto nonGraphStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numKernels; i++) {
            hipLaunchKernelGGL(conv2d, gridDim, blockDim, 0, 0, 
                               d_input, d_kernel, d_outputs[i], 
                               inputWidth, inputHeight, kernelWidth, kernelHeight, 
                               outputWidth, outputHeight);
        }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        nonGraphTotalTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    auto nonGraphEnd = std::chrono::high_resolution_clock::now();
    float nonGraphAverageTime = nonGraphTotalTime / 10.0f;

    std::cout << "Non-graph total execution time: " << nonGraphTotalTime << " ms" << std::endl;
    std::cout << "Non-graph average execution time: " << nonGraphAverageTime << " microseconds" << std::endl;

    // Graph execution
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

    std::vector<hipGraphNode_t> kernelNodes(numKernels);
    for (int i = 0; i < numKernels; i++) {
        hipKernelNodeParams kernelParams = {0};
        void* kernelArgs[] = {reinterpret_cast<void*>(&d_input), 
                              reinterpret_cast<void*>(&d_kernel), 
                              reinterpret_cast<void*>(&d_outputs[i]), 
                              const_cast<void*>(reinterpret_cast<const void*>(&inputWidth)), 
                              const_cast<void*>(reinterpret_cast<const void*>(&inputHeight)), 
                              const_cast<void*>(reinterpret_cast<const void*>(&kernelWidth)), 
                              const_cast<void*>(reinterpret_cast<const void*>(&kernelHeight)), 
                              const_cast<void*>(reinterpret_cast<const void*>(&outputWidth)), 
                              const_cast<void*>(reinterpret_cast<const void*>(&outputHeight))};
        kernelParams.func = (void*)conv2d;
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;

        CHECK_HIP_ERROR(hipGraphAddKernelNode(&kernelNodes[i], graph, nullptr, 0, &kernelParams));
    }

    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    float graphTotalTime = 0.0f;
    auto graphStart = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        graphTotalTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    auto graphEnd = std::chrono::high_resolution_clock::now();
    float graphAverageTime = graphTotalTime / 10.0f;

    std::cout << "Graph total execution time: " << graphTotalTime << " ms" << std::endl;
    std::cout << "Graph average execution time: " << graphAverageTime << " microseconds" << std::endl;

    // Percentage improvement
    float improvement = ((nonGraphAverageTime - graphAverageTime) / nonGraphAverageTime) * 100.0f;
    std::cout << "Percentage improvement with graph: " << improvement << "%" << std::endl;

    // Cleanup
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_kernel));
    for (int i = 0; i < numKernels; i++) {
        CHECK_HIP_ERROR(hipFree(d_outputs[i]));
    }
}

int main() {
    runConvolutionComparison();
    return 0;
}
