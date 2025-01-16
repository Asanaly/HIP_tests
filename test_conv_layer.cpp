#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Convolution kernel (no padding, stride = 1)
__global__ void conv2dKernel(float* input, float* filter, float* output, int H, int W, int KH, int KW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < H - KH + 1 && y < W - KW + 1) {
        float sum = 0.0f;
        for (int i = 0; i < KH; ++i) {
            for (int j = 0; j < KW; ++j) {
                sum += input[(x + i) * W + (y + j)] * filter[i * KW + j];
            }
        }
        output[x * (W - KW + 1) + y] = sum;
    }
}

int main() {
    const int H = 64;      // Input height
    const int W = 64;      // Input width
    const int KH = 3;      // Kernel height
    const int KW = 3;      // Kernel width
    const size_t inputSize = H * W * sizeof(float);
    const size_t filterSize = KH * KW * sizeof(float);
    const size_t outputSize = (H - KH + 1) * (W - KW + 1) * sizeof(float);

    // Host allocations
    std::vector<float> inputHost(H * W, 1.0f);  // Fill with 1.0
    std::vector<float> filterHost(KH * KW, 0.5f);  // Fill with 0.5
    std::vector<float> outputHost((H - KH + 1) * (W - KW + 1), 0.0f);

    // Device allocations
    float *inputDevice, *filterDevice, *outputDevice;
    hipMalloc(&inputDevice, inputSize);
    hipMalloc(&filterDevice, filterSize);
    hipMalloc(&outputDevice, outputSize);

    // HIP streams and graphs
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipGraph_t graph;
    hipGraphExec_t graphExec;

    // Record graph build start time
    auto buildStart = std::chrono::high_resolution_clock::now();

    // Begin graph capture
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

    // Transfer input and filter to the device
    hipMemcpyAsync(inputDevice, inputHost.data(), inputSize, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(filterDevice, filterHost.data(), filterSize, hipMemcpyHostToDevice, stream);

    // Launch the convolution kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((H - KH + 1 + blockSize.x - 1) / blockSize.x, 
                  (W - KW + 1 + blockSize.y - 1) / blockSize.y);
    hipLaunchKernelGGL(conv2dKernel, gridSize, blockSize, 0, stream, inputDevice, filterDevice, outputDevice, H, W, KH, KW);

    // Transfer output back to the host
    hipMemcpyAsync(outputHost.data(), outputDevice, outputSize, hipMemcpyDeviceToHost, stream);

    // End graph capture
    hipStreamEndCapture(stream, &graph);

    // Instantiate the graph
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Record graph build end time
    auto buildEnd = std::chrono::high_resolution_clock::now();

    // Launch the graph and measure execution time
    auto execStart = std::chrono::high_resolution_clock::now();
    hipGraphLaunch(graphExec, stream);
    hipStreamSynchronize(stream);
    auto execEnd = std::chrono::high_resolution_clock::now();

    // Calculate times
    std::chrono::duration<double> buildTime = buildEnd - buildStart;
    std::chrono::duration<double> execTime = execEnd - execStart;

    // Print results
    std::cout << "Graph build time: " << buildTime.count() << " seconds\n";
    std::cout << "Graph execution time: " << execTime.count() << " seconds\n";
    std::cout << "Throughput: " << (H - KH + 1) * (W - KW + 1) / execTime.count() / 1e6 << " MPixels/s\n";

    // Cleanup
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    hipFree(inputDevice);
    hipFree(filterDevice);
    hipFree(outputDevice);

    return 0;
}
