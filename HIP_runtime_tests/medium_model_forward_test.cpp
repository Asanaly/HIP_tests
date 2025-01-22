#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>

#define INPUT_WIDTH 4
#define INPUT_HEIGHT 4
#define INPUT_CHANNELS 1
#define KERNEL_SIZE 3
#define OUTPUT_CHANNELS 1
#define POOL_SIZE 2
#define DENSE_SIZE 4
#define OUTPUT_SIZE 2

// Activation function (ReLU)
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Convolution operation
__global__ void convolution(float* input, float* kernel, float* output, int input_width, int input_height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel = kernel_size / 2;

    if (x < input_width && y < input_height) {
        float sum = 0.0f;
        for (int i = -half_kernel; i <= half_kernel; i++) {
            for (int j = -half_kernel; j <= half_kernel; j++) {
                int xi = x + i;
                int yj = y + j;
                if (xi >= 0 && xi < input_width && yj >= 0 && yj < input_height) {
                    sum += input[yj * input_width + xi] * kernel[(i + half_kernel) * kernel_size + (j + half_kernel)];
                }
            }
        }
        output[y * input_width + x] = relu(sum);
    }
}

// Max Pooling operation
__global__ void max_pooling(float* input, float* output, int input_width, int input_height, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pool_width = input_width / pool_size;

    if (x < pool_width && y < pool_width) {
        float max_val = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int xi = x * pool_size + i;
                int yj = y * pool_size + j;
                max_val = fmaxf(max_val, input[yj * input_width + xi]);
            }
        }
        output[y * pool_width + x] = max_val;
    }
}

// Fully Connected Layer
__global__ void dense(float* input, float* weights, float* biases, float* output, int input_size, int output_size) {
    int idx = threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[idx * input_size + j];
        }
        output[idx] = relu(sum + biases[idx]);
    }
}

int main() {
    // Host data
    float input[INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS] = {
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        0.2f, 0.5f, 0.2f,
        0.5f, 1.0f, 0.5f,
        0.2f, 0.5f, 0.2f
    };
    float dense_weights[INPUT_WIDTH * INPUT_HEIGHT * DENSE_SIZE] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f, 0.7f
    };
    float dense_biases[DENSE_SIZE] = {0.1f, 0.2f, 0.3f, 0.4f};
    float output_weights[DENSE_SIZE * OUTPUT_SIZE] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f,
        0.7f, 0.8f
    };
    float output_biases[OUTPUT_SIZE] = {0.1f, 0.2f};
    float conv_output[INPUT_WIDTH * INPUT_HEIGHT * OUTPUT_CHANNELS];
    float pool_output[INPUT_WIDTH / POOL_SIZE * INPUT_HEIGHT / POOL_SIZE * OUTPUT_CHANNELS];
    float dense_output[DENSE_SIZE];
    float final_output[OUTPUT_SIZE];

    // Device data
    float *d_input, *d_kernel, *d_conv_output, *d_pool_output;
    float *d_dense_weights, *d_dense_biases, *d_dense_output;
    float *d_output_weights, *d_output_biases, *d_final_output;

    hipMalloc(&d_input, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    hipMalloc(&d_conv_output, INPUT_WIDTH * INPUT_HEIGHT * OUTPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_pool_output, INPUT_WIDTH / POOL_SIZE * INPUT_HEIGHT / POOL_SIZE * OUTPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_dense_weights, INPUT_WIDTH * INPUT_HEIGHT * DENSE_SIZE * sizeof(float));
    hipMalloc(&d_dense_biases, DENSE_SIZE * sizeof(float));
    hipMalloc(&d_dense_output, DENSE_SIZE * sizeof(float));
    hipMalloc(&d_output_weights, DENSE_SIZE * OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_output_biases, OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    hipMemcpy(d_input, input, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_dense_weights, dense_weights, INPUT_WIDTH * INPUT_HEIGHT * DENSE_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_dense_biases, dense_biases, DENSE_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_weights, output_weights, DENSE_SIZE * OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_biases, output_biases, OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // Define HIP graph
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipGraphCreate(&graph, 0);

    // Convolution layer
    dim3 blockSize(16, 16);
    dim3 gridSize((INPUT_WIDTH + blockSize.x - 1) / blockSize.x, (INPUT_HEIGHT + blockSize.y - 1) / blockSize.y);
    hipGraphNode_t convNode;
    hipKernelNodeParams convParams = {0};
    void* convArgs[] = {&d_input, &d_kernel, &d_conv_output, &INPUT_WIDTH, &INPUT_HEIGHT, &KERNEL_SIZE};
    convParams.func = reinterpret_cast<void*>(convolution);
    convParams.gridDim = gridSize;
    convParams.blockDim = blockSize;
    convParams.sharedMemBytes = 0;
    convParams.kernelParams = convArgs;
    convParams.extra = nullptr;
    hipGraphAddKernelNode(&convNode, graph, nullptr, 0, &convParams);

    // Max Pooling layer
    hipGraphNode_t poolNode;
    hipKernelNodeParams poolParams = {0};
    void* poolArgs[] = {&d_conv_output, &d_pool_output, &INPUT_WIDTH, &INPUT_HEIGHT, &POOL_SIZE};
    poolParams.func = reinterpret_cast<void*>(max_pooling);
    poolParams.gridDim = dim3(1);
    poolParams.blockDim = dim3(INPUT_WIDTH / POOL_SIZE, INPUT_HEIGHT / POOL_SIZE);
    poolParams.sharedMemBytes = 0;
    poolParams.kernelParams = poolArgs;
    poolParams.extra = nullptr;
    hipGraphAddKernelNode(&poolNode, graph, nullptr, 0, &poolParams);

    // Dense layer
    hipGraphNode_t denseNode;
    hipKernelNodeParams denseParams = {0};
    void* denseArgs[] = {&d_pool_output, &d_dense_weights, &d_dense_biases, &d_dense_output, &DENSE_SIZE};
    denseParams.func = reinterpret_cast<void*>(dense);
    denseParams.gridDim = dim3(1);
    denseParams.blockDim = dim3(DENSE_SIZE);
    denseParams.sharedMemBytes = 0;
    denseParams.kernelParams = denseArgs;
    denseParams.extra = nullptr;
    hipGraphAddKernelNode(&denseNode, graph, nullptr, 0, &denseParams);

    // Output layer
    hipGraphNode_t outputNode;
    hipKernelNodeParams outputParams = {0};
    void* outputArgs[] = {&d_dense_output, &d_output_weights, &d_output_biases, &d_final_output, &DENSE_SIZE, &OUTPUT_SIZE};
    outputParams.func = reinterpret_cast<void*>(dense);
    outputParams.gridDim = dim3(1);
    outputParams.blockDim = dim3(OUTPUT_SIZE);
    outputParams.sharedMemBytes = 0;
    outputParams.kernelParams = outputArgs;
    outputParams.extra = nullptr;
    hipGraphAddKernelNode(&outputNode, graph, nullptr, 0, &outputParams);

    // Instantiate the graph
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Measure execution time and run the graph 20 times
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++) {
        hipGraphLaunch(graphExec, stream);
        hipStreamSynchronize(stream);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Copy results back to host
    hipMemcpy(final_output, d_final_output, OUTPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Output performance data
    std::cout << "Execution Time: " << duration.count() / 20 << " seconds (average)" << std::endl;

    size_t free_mem, total_mem;
    hipMemGetInfo(&free_mem, &total_mem);
    std::cout << "Memory Usage: " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_kernel);
    hipFree(d_conv_output);
    hipFree(d_pool_output);
    hipFree(d_dense_weights);
    hipFree(d_dense_biases);
    hipFree(d_dense_output);
    hipFree(d_output_weights);
    hipFree(d_output_biases);
    hipFree(d_final_output);

    hipStreamDestroy(stream);
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);

    return 0;
}