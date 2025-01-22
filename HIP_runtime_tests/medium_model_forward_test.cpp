#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <chrono>

#define INPUT_SIZE 224*224*3  // Example input size for an image
#define CONV1_SIZE 64
#define CONV2_SIZE 128
#define FC1_SIZE 512
#define FC2_SIZE 256
#define RUNS 20
#define OUTPUT_SIZE 10  // Example output size for classification

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
    // Allocate host data
    std::vector<float> input(INPUT_SIZE, 1.0f); // Example input filled with 1.0f
    std::vector<float> conv1_weights(INPUT_SIZE * CONV1_SIZE, 0.1f);
    std::vector<float> conv1_biases(CONV1_SIZE, 0.1f);
    std::vector<float> conv2_weights(CONV1_SIZE * CONV2_SIZE, 0.1f);
    std::vector<float> conv2_biases(CONV2_SIZE, 0.1f);
    std::vector<float> fc1_weights(CONV2_SIZE * FC1_SIZE, 0.1f);
    std::vector<float> fc1_biases(FC1_SIZE, 0.1f);
    std::vector<float> fc2_weights(FC1_SIZE * FC2_SIZE, 0.1f);
    std::vector<float> fc2_biases(FC2_SIZE, 0.1f);
    std::vector<float> output_weights(FC2_SIZE * OUTPUT_SIZE, 0.1f);
    std::vector<float> output_biases(OUTPUT_SIZE, 0.1f);

    std::vector<float> conv1_output(CONV1_SIZE);
    std::vector<float> conv2_output(CONV2_SIZE);
    std::vector<float> fc1_output(FC1_SIZE);
    std::vector<float> fc2_output(FC2_SIZE);
    std::vector<float> final_output(OUTPUT_SIZE);

    // Allocate device data
    float *d_input, *d_conv1_weights, *d_conv1_biases, *d_conv1_output;
    float *d_conv2_weights, *d_conv2_biases, *d_conv2_output;
    float *d_fc1_weights, *d_fc1_biases, *d_fc1_output;
    float *d_fc2_weights, *d_fc2_biases, *d_fc2_output;
    float *d_output_weights, *d_output_biases, *d_final_output;

    hipMalloc(&d_input, INPUT_SIZE * sizeof(float));
    hipMalloc(&d_conv1_weights, INPUT_SIZE * CONV1_SIZE * sizeof(float));
    hipMalloc(&d_conv1_biases, CONV1_SIZE * sizeof(float));
    hipMalloc(&d_conv1_output, CONV1_SIZE * sizeof(float));
    hipMalloc(&d_conv2_weights, CONV1_SIZE * CONV2_SIZE * sizeof(float));
    hipMalloc(&d_conv2_biases, CONV2_SIZE * sizeof(float));
    hipMalloc(&d_conv2_output, CONV2_SIZE * sizeof(float));
    hipMalloc(&d_fc1_weights, CONV2_SIZE * FC1_SIZE * sizeof(float));
    hipMalloc(&d_fc1_biases, FC1_SIZE * sizeof(float));
    hipMalloc(&d_fc1_output, FC1_SIZE * sizeof(float));
    hipMalloc(&d_fc2_weights, FC1_SIZE * FC2_SIZE * sizeof(float));
    hipMalloc(&d_fc2_biases, FC2_SIZE * sizeof(float));
    hipMalloc(&d_fc2_output, FC2_SIZE * sizeof(float));
    hipMalloc(&d_output_weights, FC2_SIZE * OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_output_biases, OUTPUT_SIZE * sizeof(float));
    hipMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    hipMemcpy(d_input, input.data(), INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_conv1_weights, conv1_weights.data(), INPUT_SIZE * CONV1_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_conv1_biases, conv1_biases.data(), CONV1_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_conv2_weights, conv2_weights.data(), CONV1_SIZE * CONV2_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_conv2_biases, conv2_biases.data(), CONV2_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_fc1_weights, fc1_weights.data(), CONV2_SIZE * FC1_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_fc1_biases, fc1_biases.data(), FC1_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_fc2_weights, fc2_weights.data(), FC1_SIZE * FC2_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_fc2_biases, fc2_biases.data(), FC2_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_weights, output_weights.data(), FC2_SIZE * OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output_biases, output_biases.data(), OUTPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);

    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;

    hipStreamCreate(&stream);
    hipGraphCreate(&graph, 0);

    hipGraphNode_t nodes[10];
    hipKernelNodeParams kernelNodeParams[5] = {0};

    kernelNodeParams[0].func = (void*)feedforward;
    kernelNodeParams[0].gridDim = dim3(1);
    kernelNodeParams[0].blockDim = dim3(CONV1_SIZE);
    kernelNodeParams[0].sharedMemBytes = 0;
    kernelNodeParams[0].kernelParams = (void**)&d_input;
    kernelNodeParams[0].extra = nullptr;

    hipGraphAddKernelNode(&nodes[0], graph, nullptr, 0, &kernelNodeParams[0]);

    kernelNodeParams[1].func = (void*)feedforward;
    kernelNodeParams[1].gridDim = dim3(1);
    kernelNodeParams[1].blockDim = dim3(CONV2_SIZE);
    kernelNodeParams[1].sharedMemBytes = 0;
    kernelNodeParams[1].kernelParams = (void**)&d_conv1_output;
    kernelNodeParams[1].extra = nullptr;

    hipGraphAddKernelNode(&nodes[1], graph, nullptr, 0, &kernelNodeParams[1]);

    kernelNodeParams[2].func = (void*)feedforward;
    kernelNodeParams[2].gridDim = dim3(1);
    kernelNodeParams[2].blockDim = dim3(FC1_SIZE);
    kernelNodeParams[2].sharedMemBytes = 0;
    kernelNodeParams[2].kernelParams = (void**)&d_conv2_output;
    kernelNodeParams[2].extra = nullptr;

    hipGraphAddKernelNode(&nodes[2], graph, nullptr, 0, &kernelNodeParams[2]);

    kernelNodeParams[3].func = (void*)feedforward;
    kernelNodeParams[3].gridDim = dim3(1);
    kernelNodeParams[3].blockDim = dim3(FC2_SIZE);
    kernelNodeParams[3].sharedMemBytes = 0;
    kernelNodeParams[3].kernelParams = (void**)&d_fc1_output;
    kernelNodeParams[3].extra = nullptr;

    hipGraphAddKernelNode(&nodes[3], graph, nullptr, 0, &kernelNodeParams[3]);

    kernelNodeParams[4].func = (void*)feedforward;
    kernelNodeParams[4].gridDim = dim3(1);
    kernelNodeParams[4].blockDim = dim3(OUTPUT_SIZE);
    kernelNodeParams[4].sharedMemBytes = 0;
    kernelNodeParams[4].kernelParams = (void**)&d_fc2_output;
    kernelNodeParams[4].extra = nullptr;

    hipGraphAddKernelNode(&nodes[4], graph, nullptr, 0, &kernelNodeParams[4]);

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

    hipMemcpy(conv1_output.data(), d_conv1_output, CONV1_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(conv2_output.data(), d_conv2_output, CONV2_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(fc1_output.data(), d_fc1_output, FC1_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(fc2_output.data(), d_fc2_output, FC2_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(final_output.data(), d_final_output, OUTPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Average Execution Time: " << average_time << " ms" << std::endl;
    std::cout << "Total Execution Time: " << total_time << " ms" << std::endl;
    std::cout << "Graph Building Time: " << graph_building_time << " ms" << std::endl;

    hipFree(d_input);
    hipFree(d_conv1_weights);
    hipFree(d_conv1_biases);
    hipFree(d_conv1_output);
    hipFree(d_conv2_weights);
    hipFree(d_conv2_biases);
    hipFree(d_conv2_output);
    hipFree(d_fc1_weights);
    hipFree(d_fc1_biases);
    hipFree(d_fc1_output);
    hipFree(d_fc2_weights);
    hipFree(d_fc2_biases);
    hipFree(d_fc2_output);
    hipFree(d_output_weights);
    hipFree(d_output_biases);
    hipFree(d_final_output);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);

    return 0;
}
