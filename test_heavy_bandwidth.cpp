// Debugging Helper
#define DEBUG_PRINT(msg) std::cout << msg << std::endl;

// Kernel for heavy computation with bounds check
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

// Modified ExecuteGraph with Validation
void executeGraphWithValidation(hipGraphExec_t graphExec, hipStream_t stream, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        hipError_t err = hipGraphLaunch(graphExec, stream);
        if (err != hipSuccess) {
            std::cerr << "Error during graph execution: " << hipGetErrorString(err) << std::endl;
            exit(-1);
        }
    }
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
}

// Memory Allocation Validation
void allocateAndValidate(void **ptr, size_t size) {
    hipError_t err = hipMalloc(ptr, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory: " << hipGetErrorString(err) << std::endl;
        exit(-1);
    }
}

int main() {
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    const int width = 1024, height = 1024, dataSize = width * height;
    float *d_input = nullptr, *d_output = nullptr;

    allocateAndValidate(reinterpret_cast<void **>(&d_input), dataSize * sizeof(float));
    allocateAndValidate(reinterpret_cast<void **>(&d_output), dataSize * sizeof(float));

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

    // Instantiate and execute the graph
    hipGraphExec_t graphExec;
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    executeGraphWithValidation(graphExec, stream, 5);

    // Clean up
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph));
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_output));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
