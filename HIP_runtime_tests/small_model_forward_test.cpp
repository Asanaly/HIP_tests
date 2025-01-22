#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Function to print memory usage
void printMemoryUsage(const std::string& label) {
    size_t freeMem, totalMem;
    CHECK_HIP_ERROR(hipMemGetInfo(&freeMem, &totalMem));
    std::cout << label << " - Free memory: " << freeMem / (1024.0 * 1024.0) << " MB, Total memory: " 
              << totalMem / (1024.0 * 1024.0) << " MB" << std::endl;
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

void runConvolutionTest() {
    const int inputWidth = 64, inputHeight = 64;
    const int kernelWidth = 3, kernelHeight = 3;
    const int outputWidth = inputWidth - kernelWidth + 1;
    const int outputHeight = inputHeight - kernelHeight + 1;

    const size_t inputSize = inputWidth * inputHeight * sizeof(float);
    const size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    const size_t outputSize = outputWidth * outputHeight * sizeof(float);

    std::vector<float> h_input(inputWidth * inputHeight, 1.0f);
    std::vector<float> h_kernel(kernelWidth * kernelHeight, 1.0f);
    std::vector<float> h_output(outputWidth * outputHeight, 0.0f);

    float *d_input, *d_kernel, *d_output;
    float *d_intermediate1, *d_intermediate2, *d_intermediate3, *d_intermediate4;

    // Report initial memory usage
    printMemoryUsage("Before memory allocation");

    // Allocate memory
    CHECK_HIP_ERROR(hipMalloc(&d_input, inputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_kernel, kernelSize));
    CHECK_HIP_ERROR(hipMalloc(&d_intermediate1, outputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_intermediate2, outputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_intermediate3, outputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_intermediate4, outputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_output, outputSize));

    // Report memory usage after allocation
    printMemoryUsage("After memory allocation");

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputSize, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_kernel, h_kernel.data(), kernelSize, hipMemcpyHostToDevice));

    // Create HIP graph
    hipGraph_t graph;
    hipGraphNode_t kernelNode[5];
    hipKernelNodeParams kernelParams[5] = {0};
    hipStream_t stream;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, 
                 (outputHeight + blockDim.y - 1) / blockDim.y);

    void* kernelArgs[] = {reinterpret_cast<void*>(&d_input), 
                          reinterpret_cast<void*>(&d_kernel), 
                          reinterpret_cast<void*>(&d_output), 
                          const_cast<void*>(reinterpret_cast<const void*>(&inputWidth)), 
                          const_cast<void*>(reinterpret_cast<const void*>(&inputHeight)), 
                          const_cast<void*>(reinterpret_cast<const void*>(&kernelWidth)), 
                          const_cast<void*>(reinterpret_cast<const void*>(&kernelHeight)), 
                          const_cast<void*>(reinterpret_cast<const void*>(&outputWidth)), 
                          const_cast<void*>(reinterpret_cast<const void*>(&outputHeight))};

    kernelParams[0].func = (void*)conv2d;
    kernelParams[0].gridDim = gridDim;
    kernelParams[0].blockDim = blockDim;
    kernelParams[0].sharedMemBytes = 0;
    kernelParams[0].kernelParams = kernelArgs;
    kernelParams[0].extra = nullptr;

    CHECK_HIP_ERROR(hipGraphAddKernelNode(&kernelNode[0], graph, nullptr, 0, &kernelParams[0]));

    hipGraphExec_t graphExec;
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Report memory usage before execution
    printMemoryUsage("Before graph execution");

    float total_time = 0.0f;
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    float average_time = total_time / 100.0f;
    std::cout << "Average execution time: " << average_time << " microseconds" << std::endl;

    // Report memory usage after execution
    printMemoryUsage("After graph execution");

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_output.data(), d_output, outputSize, hipMemcpyDeviceToHost));

    // Free resources
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_kernel));
    CHECK_HIP_ERROR(hipFree(d_intermediate1));
    CHECK_HIP_ERROR(hipFree(d_intermediate2));
    CHECK_HIP_ERROR(hipFree(d_intermediate3));
    CHECK_HIP_ERROR(hipFree(d_intermediate4));
    CHECK_HIP_ERROR(hipFree(d_output));

    // Report final memory usage
    printMemoryUsage("After memory deallocation");

    std::cout << "Test completed successfully." << std::endl;
}

int main() {
    runConvolutionTest();
    return 0;
}
