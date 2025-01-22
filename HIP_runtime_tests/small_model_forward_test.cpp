#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2

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
    float hidden_weights[INPUT_SIZE * HIDDEN_SIZE] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        0.0f, 0.1f, 0.2f
    };
    float hidden_biases[HIDDEN_SIZE] = {0.1f, 0.2f, 0.3f};
    float output_weights[HIDDEN_SIZE * OUTPUT_SIZE] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    };
    float output_biases[OUTPUT_SIZE] = {0.1f, 0.2f};
    float hidden_output[HIDDEN_SIZE];
    float final_output[OUTPUT_SIZE];

    // Device data
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

    // Feedforward for hidden layer
    hipLaunchKernelGGL(feedforward, dim3(1), dim3(HIDDEN_SIZE), 0, 0, d_input, d_hidden_weights, d_hidden_biases, d_hidden_output, INPUT_SIZE, HIDDEN_SIZE);
    hipDeviceSynchronize();

    // Feedforward for output layer
    hipLaunchKernelGGL(feedforward, dim3(1), dim3(OUTPUT_SIZE), 0, 0, d_hidden_output, d_output_weights, d_output_biases, d_final_output, HIDDEN_SIZE, OUTPUT_SIZE);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(hidden_output, d_hidden_output, HIDDEN_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(final_output, d_final_output, OUTPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Output final results
    std::cout << "Output: ";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_input);
    hipFree(d_hidden_weights);
    hipFree(d_hidden_biases);
    hipFree(d_hidden_output);
    hipFree(d_output_weights);
    hipFree(d_output_biases);
    hipFree(d_final_output);

    return 0;
}
