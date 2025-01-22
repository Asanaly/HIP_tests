#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <cmath>

#define CHECK_HIP_ERROR(val) { \
    if(val != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(val) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
}

// Layer base class
class Layer {
public:
    virtual void forward(hipStream_t stream) = 0;
    virtual void setupGraphNode(hipGraph_t& graph, hipGraphNode_t& lastNode) = 0;
    virtual ~Layer() = default;
};

// Convolution Layer
class ConvLayer : public Layer {
private:
    int inputChannels, outputChannels;
    int inputHeight, inputWidth;
    int kernelSize;
    int outputHeight, outputWidth;
    float *d_input, *d_weights, *d_output;
    float *d_bias;
    
    static __global__ void convForward(
        const float* input, const float* weights, const float* bias,
        float* output, int inC, int outC, int H, int W, int K) {
        int h = blockIdx.y * blockDim.y + threadIdx.y;
        int w = blockIdx.x * blockDim.x + threadIdx.x;
        int c = blockIdx.z;
        
        if (h < H - K + 1 && w < W - K + 1 && c < outC) {
            float sum = 0.0f;
            for (int ic = 0; ic < inC; ++ic) {
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        sum += input[(ic * H + h + kh) * W + w + kw] *
                               weights[(c * inC + ic) * K * K + kh * K + kw];
                    }
                }
            }
            output[(c * (H-K+1) + h) * (W-K+1) + w] = sum + bias[c];
        }
    }

public:
    ConvLayer(int inChannels, int outChannels, int inHeight, int inWidth, int kSize)
        : inputChannels(inChannels), outputChannels(outChannels),
          inputHeight(inHeight), inputWidth(inWidth), kernelSize(kSize) {
        outputHeight = inHeight - kSize + 1;
        outputWidth = inWidth - kSize + 1;
        
        // Allocate device memory
        size_t weightSize = outputChannels * inputChannels * kernelSize * kernelSize;
        CHECK_HIP_ERROR(hipMalloc(&d_weights, weightSize * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_bias, outputChannels * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&d_output, outputChannels * outputHeight * outputWidth * sizeof(float)));
        
        // Initialize weights with Xavier initialization
        std::vector<float> h_weights(weightSize);
        float scale = std::sqrt(2.0f / (inputChannels * kernelSize * kernelSize));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (size_t i = 0; i < weightSize; ++i) {
            h_weights[i] = dist(gen);
        }
        
        CHECK_HIP_ERROR(hipMemcpy(d_weights, h_weights.data(), weightSize * sizeof(float), 
                                hipMemcpyHostToDevice));
        
        // Initialize biases to zero
        std::vector<float> h_bias(outputChannels, 0.0f);
        CHECK_HIP_ERROR(hipMemcpy(d_bias, h_bias.data(), outputChannels * sizeof(float), 
                                hipMemcpyHostToDevice));
    }
    
    void setInput(float* input) { d_input = input; }
    float* getOutput() { return d_output; }
    
    void forward(hipStream_t stream) override {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            outputChannels
        );
        
        hipLaunchKernelGGL(convForward, gridDim, blockDim, 0, stream,
            d_input, d_weights, d_bias, d_output,
            inputChannels, outputChannels, inputHeight, inputWidth, kernelSize);
    }
    
    void setupGraphNode(hipGraph_t& graph, hipGraphNode_t& lastNode) override {
        hipKernelNodeParams nodeParams = {};
        
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (outputWidth + blockDim.x - 1) / blockDim.x,
            (outputHeight + blockDim.y - 1) / blockDim.y,
            outputChannels
        );
        
        void* kernelArgs[] = {
            &d_input, &d_weights, &d_bias, &d_output,
            &inputChannels, &outputChannels, &inputHeight, &inputWidth, &kernelSize
        };
        
        nodeParams.func = reinterpret_cast<void*>(convForward);
        nodeParams.gridDim = gridDim;
        nodeParams.blockDim = blockDim;
        nodeParams.sharedMemBytes = 0;
        nodeParams.kernelParams = kernelArgs;
        nodeParams.extra = nullptr;
        
        hipGraphNode_t node;
        CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, graph, lastNode ? &lastNode : nullptr, 
                                            lastNode ? 1 : 0, &nodeParams));
        lastNode = node;
    }
    
    ~ConvLayer() {
        hipFree(d_weights);
        hipFree(d_bias);
        hipFree(d_output);
    }
};

// ReLU Activation Layer
class ReLULayer : public Layer {
private:
    float *d_input, *d_output;
    int size;
    
    static __global__ void reluForward(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = input[idx] > 0 ? input[idx] : 0;
        }
    }

public:
    ReLULayer(int inputSize) : size(inputSize) {
        CHECK_HIP_ERROR(hipMalloc(&d_output, size * sizeof(float)));
    }
    
    void setInput(float* input) { d_input = input; }
    float* getOutput() { return d_output; }
    
    void forward(hipStream_t stream) override {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        hipLaunchKernelGGL(reluForward, dim3(numBlocks), dim3(blockSize), 0, stream,
            d_input, d_output, size);
    }
    
    void setupGraphNode(hipGraph_t& graph, hipGraphNode_t& lastNode) override {
        hipKernelNodeParams nodeParams = {};
        
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        void* kernelArgs[] = {&d_input, &d_output, &size};
        
        nodeParams.func = reinterpret_cast<void*>(reluForward);
        nodeParams.gridDim = dim3(numBlocks);
        nodeParams.blockDim = dim3(blockSize);
        nodeParams.sharedMemBytes = 0;
        nodeParams.kernelParams = kernelArgs;
        nodeParams.extra = nullptr;
        
        hipGraphNode_t node;
        CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, graph, lastNode ? &lastNode : nullptr, 
                                            lastNode ? 1 : 0, &nodeParams));
        lastNode = node;
    }
    
    ~ReLULayer() {
        hipFree(d_output);
    }
};

// Neural Network class
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
    
    void compile() {
        // Create graph
        CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));
        
        // Setup graph nodes
        hipGraphNode_t lastNode = nullptr;
        for (auto& layer : layers) {
            layer->setupGraphNode(graph, lastNode);
        }
        
        // Instantiate graph
        CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }
    
    void forward(float* input, int inputSize) {
        // Copy input to device
        CHECK_HIP_ERROR(hipMemcpy(d_input, input, inputSize * sizeof(float), hipMemcpyHostToDevice));
        
        // Launch graph
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

// Example usage
int main() {
    // Create a simple CNN
    const int inputChannels = 3;
    const int inputHeight = 32;
    const int inputWidth = 32;
    const int kernelSize = 3;
    
    NeuralNetwork net;
    
    // Add layers
    net.addLayer(std::make_unique<ConvLayer>(inputChannels, 64, inputHeight, inputWidth, kernelSize));
    net.addLayer(std::make_unique<ReLULayer>(64 * (inputHeight-kernelSize+1) * (inputWidth-kernelSize+1)));
    net.addLayer(std::make_unique<ConvLayer>(64, 128, inputHeight-kernelSize+1, inputWidth-kernelSize+1, kernelSize));
    net.addLayer(std::make_unique<ReLULayer>(128 * (inputHeight-2*kernelSize+2) * (inputWidth-2*kernelSize+2)));
    
    // Adding 20 additional layers to the network
    for (int i = 0; i < 10; ++i) {
        net.addLayer(std::make_unique<ConvLayer>(128, 128, inputHeight-2*kernelSize+2, inputWidth-2*kernelSize+2, kernelSize));
        net.addLayer(std::make_unique<ReLULayer>(128 * (inputHeight-3*kernelSize+3) * (inputWidth-3*kernelSize+3)));
    }

    // Compile network
    net.compile();
    
    // Create input data
    std::vector<float> input(inputChannels * inputHeight * inputWidth);
    std::generate(input.begin(), input.end(), std::rand);
    
    // Time the forward pass
    auto start = std::chrono::high_resolution_clock::now();
    net.forward(input.data(), input.size());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Forward pass time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;
    
    return 0;
}