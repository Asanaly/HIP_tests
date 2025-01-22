#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#define CHECK_HIP_ERROR(val) { \
    if (val != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(val) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
}

double getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

class Layer {
public:
    virtual void forward(hipStream_t stream) = 0;
    virtual void setupGraphNode(hipGraph_t& graph, hipGraphNode_t& lastNode) = 0;
    virtual ~Layer() = default;
};

class DenseLayer : public Layer {
private:
    int inputSize, outputSize;
    float *d_input, *d_weights, *d_output;

    static __global__ void denseForward(const float* input, const float* weights, float* output, int inSize, int outSize) {
        int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (outIdx < outSize) {
            float sum = 0.0f;
            for (int inIdx = 0; inIdx < inSize; ++inIdx) {
                sum += input[inIdx] * weights[outIdx * inSize + inIdx];
            }
            output[outIdx] = sum;
        }
    }

public:
    DenseLayer(int inSize, int outSize) : inputSize(inSize), outputSize(outSize) {
        CHECK_HIP_ERROR(hipMalloc(&d_weights, inSize * outSize * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_output, outSize * sizeof(float)));
        
        // Initialize weights
        std::vector<float> h_weights(inSize * outSize, 0.01f);
        CHECK_HIP_ERROR(hipMemcpy(d_weights, h_weights.data(), inSize * outSize * sizeof(float), hipMemcpyHostToDevice));
    }

    void setInput(float* input) { d_input = input; }
    float* getOutput() { return d_output; }

    void forward(hipStream_t stream) override {
        dim3 blockDim(256);
        dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x);
        hipLaunchKernelGGL(denseForward, gridDim, blockDim, 0, stream, d_input, d_weights, d_output, inputSize, outputSize);
    }

    void setupGraphNode(hipGraph_t& graph, hipGraphNode_t& lastNode) override {
        hipKernelNodeParams nodeParams = {};
        dim3 blockDim(256);
        dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x);

        void* kernelArgs[] = {&d_input, &d_weights, &d_output, &inputSize, &outputSize};
        nodeParams.func = reinterpret_cast<void*>(denseForward);
        nodeParams.gridDim = gridDim;
        nodeParams.blockDim = blockDim;
        nodeParams.sharedMemBytes = 0;
        nodeParams.kernelParams = kernelArgs;
        nodeParams.extra = nullptr;

        hipGraphNode_t node;
        CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, graph, lastNode ? &lastNode : nullptr, lastNode ? 1 : 0, &nodeParams));
        lastNode = node;
    }

    ~DenseLayer() {
        hipFree(d_weights);
        hipFree(d_output);
    }
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    float *d_input;

public:
    NeuralNetwork() {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
    }

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    void compile(int inputSize) {
        // Allocate device memory for input
        CHECK_HIP_ERROR(hipMalloc(&d_input, inputSize * sizeof(float)));

        // Create a graph
        CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

        // Set up graph nodes
        hipGraphNode_t lastNode = nullptr;
        for (auto& layer : layers) {
            layer->setupGraphNode(graph, lastNode);
        }

        // Instantiate the graph
        CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    void forward(const std::vector<float>& input) {
        // Copy input to device
        CHECK_HIP_ERROR(hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
        layers.front()->setInput(d_input);

        // Execute the graph
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    ~NeuralNetwork() {
        hipStreamDestroy(stream);
        hipGraphExecDestroy(graphExec);
        hipGraphDestroy(graph);
        hipFree(d_input);
    }
};

int main() {
    const int inputSize = 512;
    const int hiddenSize = 256;
    const int outputSize = 128;

    // Create network
    NeuralNetwork net;
    net.addLayer(std::make_unique<DenseLayer>(inputSize, hiddenSize));
    net.addLayer(std::make_unique<DenseLayer>(hiddenSize, outputSize));

    // Compile the network
    double compileStart = getCurrentTime();
    net.compile(inputSize);
    double compileEnd = getCurrentTime();
    std::cout << "Graph compilation time: " << (compileEnd - compileStart) << " ms\n";

    // Test forward pass
    std::vector<float> input(inputSize, 1.0f);
    double forwardStart = getCurrentTime();
    net.forward(input);
    double forwardEnd = getCurrentTime();
    std::cout << "Forward pass time: " << (forwardEnd - forwardStart) << " ms\n";

    return 0;
}
