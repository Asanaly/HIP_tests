#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2
#define LAYERS 3
#define REPEATS 50

// Activation function (Sigmoid)
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Feedforward operation for a single layer
__global__ void feedforward(float* input, float* weights, float* biases, float* output, int input_size, int output_size) {
    int idx = threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[idx * input_size + j];
        }
        output[idx] = sigmoid(sum + biases[idx]);
    }
}

int main() {
    // Host data
    float input[INPUT_SIZE] = {1.0f, 0.5f, 0.25f, 0.75f};
    float hidden_weights[LAYERS][INPUT_SIZE * HIDDEN_SIZE] = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.0f, 0.1f, 0.2f},
        {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.1f, 0.2f, 0.3f, 0.4f},
        {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f}
    };
    float hidden_biases[LAYERS][HIDDEN_SIZE] = {
        {0.1f, 0.2f, 0.3f},
        {0.2f, 0.3f, 0.4f},
        {0.3f, 0.4f, 0.5f}
    };
    float final_output[OUTPUT_SIZE];

    // Device data
    float *d_input, *d_hidden_weights[LAYERS], *d_hidden_biases[LAYERS], *d_hidden_output[LAYERS];
    float *d_final_output;

    hipMalloc(&d_input, INPUT_SIZE * sizeof(float));
    hipMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < LAYERS; i++) {
        hipMalloc(&d_hidden_weights[i], INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
        hipMalloc(&d_hidden_biases[i], HIDDEN_SIZE * sizeof(float));
        hipMalloc(&d_hidden_output[i], HIDDEN_SIZE * sizeof(float));

        hipMemcpy(d_hidden_weights[i], hidden_weights[i], INPUT_SIZE * HIDDEN_SIZE * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_hidden_biases[i], hidden_biases[i], HIDDEN_SIZE * sizeof(float), hipMemcpyHostToDevice);
    }

    hipMemcpy(d_input, input, INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // HIP Graph setup
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipGraphCreate(&graph, 0);

    for (int i = 0; i < LAYERS; i++) {
        void* kernelArgs[] = {
            (void*)&d_input,
            (void*)&d_hidden_weights[i],
            (void*)&d_hidden_biases[i],
            (void*)&d_hidden_output[i],
            (void*)&INPUT_SIZE,
            (void*)&HIDDEN_SIZE
        };

        hipKernelNodeParams kernelNodeParams = {};
        kernelNodeParams.func = (void*)feedforward;
        kernelNodeParams.gridDim = dim3(1);
        kernelNodeParams.blockDim = dim3(HIDDEN_SIZE);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = kernelArgs;
        kernelNodeParams.extra = nullptr;

        hipGraphNode_t kernelNode;
        hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams);
    }

    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Timing and execution
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPEATS; i++) {
        hipGraphLaunch(graphExec, 0);
        hipDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Copy results back to host
    hipMemcpy(final_output, d_final_output, OUTPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Output results
    std::cout << "Final Output: ";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average Time per Graph Execution: " << (elapsed.count() / REPEATS) << " seconds" << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_final_output);
    for (int i = 0; i < LAYERS; i++) {
        hipFree(d_hidden_weights[i]);
        hipFree(d_hidden_biases[i]);
        hipFree(d_hidden_output[i]);
    }

    return 0;
}
