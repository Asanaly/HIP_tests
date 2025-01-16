#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel for convolution
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

    // Allocate memory
    CHECK_HIP_ERROR(hipMalloc(&d_input, inputSize));
    CHECK_HIP_ERROR(hipMalloc(&d_kernel, kernelSize));
    CHECK_HIP_ERROR(hipMalloc(&d_output, outputSize));

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), inputSize, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_kernel, h_kernel.data(), kernelSize, hipMemcpyHostToDevice));

    // Create HIP graph
    hipGraph_t graph;
    hipGraphNode_t kernelNode;
    hipKernelNodeParams kernelParams = {0};
    hipStream_t stream;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, 
                 (outputHeight + blockDim.y - 1) / blockDim.y);

    void* kernelArgs[] = {&d_input, &d_kernel, &d_output, 
                          &inputWidth, &inputHeight, 
                          &kernelWidth, &kernelHeight, 
                          &outputWidth, &outputHeight};

    kernelParams.func = (void*)conv2d;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    CHECK_HIP_ERROR(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams));

    // Instantiate and launch graph
    hipGraphExec_t graphExec;
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time to build graph: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time to execute graph: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " microseconds" << std::endl;

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_output.data(), d_output, outputSize, hipMemcpyDeviceToHost));

    // Free resources
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);

    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_output);

    std::cout << "Test completed successfully." << std::endl;
}

int main() {
    runConvolutionTest();
    return 0;
}