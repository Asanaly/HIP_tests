#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Helper macro to check HIP errors
#define CHECK_HIP_ERROR(err) if (err != hipSuccess) { \
    std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
    exit(-1); \
}

// 3D stencil kernel
__global__ void stencil_3D(const float *input, const float *mask, float *output, 
                           int width, int height, int depth, int maskSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float sum = 0.0f;
    int halfMask = maskSize / 2;

    for (int dz = -halfMask; dz <= halfMask; ++dz) {
        for (int dy = -halfMask; dy <= halfMask; ++dy) {
            for (int dx = -halfMask; dx <= halfMask; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && ny >= 0 && nz >= 0 && nx < width && ny < height && nz < depth) {
                    int inputIdx = (nz * height + ny) * width + nx;
                    int maskIdx = ((dz + halfMask) * maskSize + (dy + halfMask)) * maskSize + (dx + halfMask);
                    sum += input[inputIdx] * mask[maskIdx];
                }
            }
        }
    }

    int outputIdx = (z * height + y) * width + x;
    output[outputIdx] = sum;
}

int main() {
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Define dimensions
    const int width = 128, height = 128, depth = 64;
    const int maskSize = 3; // 3x3x3 stencil
    const int dataSize = width * height * depth;

    // Allocate device memory
    float *d_input, *d_mask, *d_output;
    CHECK_HIP_ERROR(hipMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_mask, maskSize * maskSize * maskSize * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_output, dataSize * sizeof(float)));

    // Initialize data
    std::vector<float> h_input(dataSize, 1.0f);
    std::vector<float> h_mask(maskSize * maskSize * maskSize, 0.1f);
    CHECK_HIP_ERROR(hipMemcpy(d_input, h_input.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_mask, h_mask.data(), maskSize * maskSize * maskSize * sizeof(float), hipMemcpyHostToDevice));

    // Record the stencil computation graph
    hipGraph_t graph1;
    CHECK_HIP_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);
    hipLaunchKernelGGL(stencil_3D, numBlocks, threadsPerBlock, 0, stream,
                       d_input, d_mask, d_output, width, height, depth, maskSize);

    CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph1));

    // Record a second graph with additional processing
    hipGraph_t graph2;
    CHECK_HIP_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    hipLaunchKernelGGL(stencil_3D, numBlocks, threadsPerBlock, 0, stream,
                       d_output, d_mask, d_input, width, height, depth, maskSize);

    CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph2));

    // Combine the graphs
    hipGraph_t combinedGraph;
    CHECK_HIP_ERROR(hipGraphCreate(&combinedGraph, 0));

    hipGraphNode_t childNode1, childNode2;
    CHECK_HIP_ERROR(hipGraphAddChildGraphNode(&childNode1, combinedGraph, nullptr, 0, graph1));
    CHECK_HIP_ERROR(hipGraphAddChildGraphNode(&childNode2, combinedGraph, nullptr, 0, graph2));

    // Instantiate and execute combined graph
    hipGraphExec_t graphExec;
    CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, combinedGraph, nullptr, nullptr, 0));

    for (int i = 0; i < 10; ++i) {
        CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
    }

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    // Clean up
    CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
    CHECK_HIP_ERROR(hipGraphDestroy(graph1));
    CHECK_HIP_ERROR(hipGraphDestroy(graph2));
    CHECK_HIP_ERROR(hipGraphDestroy(combinedGraph));
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_mask));
    CHECK_HIP_ERROR(hipFree(d_output));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    std::cout << "Graphs combined and executed successfully!" << std::endl;
    return 0;
}
