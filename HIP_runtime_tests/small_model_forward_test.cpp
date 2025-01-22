#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <chrono>

#define INPUT_SIZE 6
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 3
#define RUNS 20

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

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
    float input[INPUT_SIZE] = {1.0f, 0.5f, 0.25f, 0.75f, 0.9f, 0.1f};
    float hidden_weights[INPUT_SIZE * HIDDEN_SIZE] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f, 0.0f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f, 0.0f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f, 0.0f
    };
    float hidden_biases[HIDDEN_SIZE] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float output_weights[HIDDEN_SIZE * OUTPUT_SIZE] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        0.0f, 0.1f, 0.2f,
        0.3f, 0.4f, 0.5f
    };
    float output_biases[OUTPUT_SIZE] = {0.1f, 0.2f, 0.3f};
    float hidden_output[HIDDEN_SIZE];
    float final_output[OUTPUT_SIZE];

    float *d_input, *d_hidden_weights, *d_hidden_biases, *d_hidden_output;
    float *d_output_weights, *d_output_biases, *d_final_output;

    hipMalloc(&d_input, INPUT_SIZE * sizeof(float));
    hipMalloc(&d_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    hipMalloc(&d_hidden_biases, HIDDEN_SIZE * sizeof(float));
    hipMalloc(&d_hidden_output, HIDDEN_SIZE * sizeof(float));
    hipMalloc(&d_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_output_biases, OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    hipMemcpy(d_input, input, INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_hidden_weights, hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_hidden_biases, hidden_biases, HIDDEN_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_weights, output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_biases, output_biases, OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);

    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;

    hipStreamCreate(&stream);
    hipGraphCreate(&graph, 0);

    hipGraphNode_t nodes[4];
    hipKernelNodeParams kernelNodeParams[2] = {0};

    kernelNodeParams[0].func = (void*)feedforward;
    kernelNodeParams[0].gridDim = dim3(1);
    kernelNodeParams[0].blockDim = dim3(HIDDEN_SIZE);
    kernelNodeParams[0].sharedMemBytes = 0;
    kernelNodeParams[0].kernelParams = (void**)&d_input;
    kernelNodeParams[0].extra = nullptr;

    hipGraphAddKernelNode(&nodes[0], graph, nullptr, 0, &kernelNodeParams[0]);

    kernelNodeParams[1].func = (void*)feedforward;
    kernelNodeParams[1].gridDim = dim3(1);
    kernelNodeParams[1].blockDim = dim3(OUTPUT_SIZE);
    kernelNodeParams[1].sharedMemBytes = 0;
    kernelNodeParams[1].kernelParams = (void**)&d_hidden_output;
    kernelNodeParams[1].extra = nullptr;

    hipGraphAddKernelNode(&nodes[1], graph, nullptr, 0, &kernelNodeParams[1]);

    auto start = std::chrono::high_resolution_clock::now();
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto graph_building_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    float total_time = 0.0f;
    for (int i = 0; i < RUNS; i++) {
        start = std::chrono::high_resolution_clock::now();
        hipGraphLaunch(graphExec, stream);
        hipStreamSynchronize(stream);
        end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    float average_time = total_time / RUNS;

    hipMemcpy(hidden_output, d_hidden_output, HIDDEN_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(final_output, d_final_output, OUTPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Average Execution Time: " << average_time << " ms" << std::endl;
    std::cout << "Total Execution Time: " << total_time << " ms" << std::endl;
    std::cout << "Graph Building Time: " << graph_building_time << " ms" << std::endl;

    hipFree(d_input);
    hipFree(d_hidden_weights);
    hipFree(d_hidden_biases);
    hipFree(d_hidden_output);
    hipFree(d_output_weights);
    hipFree(d_output_biases);
    hipFree(d_final_output);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);

    return 0;
}
