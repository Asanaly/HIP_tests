test_batch_mul.cpp:12:71: error: expected ')'
   12 | __global__ void batchedMatMulKernel(float* A, float* B, float* C, int M, int K, int P, int N) {
      |                                                                       ^
test_batch_mul.cpp:6:11: note: expanded from macro 'M'
    6 | #define M 1024  // Rows in matrix A
      |           ^
test_batch_mul.cpp:12:36: note: to match this '('
   12 | __global__ void batchedMatMulKernel(float* A, float* B, float* C, int M, int K, int P, int N) {
      |                                    ^
test_batch_mul.cpp:34:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   34 |     hipMalloc(&A_d, M * K * N * sizeof(float));
      |     ^~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:35:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   35 |     hipMalloc(&B_d, K * P * N * sizeof(float));
      |     ^~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:36:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   36 |     hipMalloc(&C_d, M * P * N * sizeof(float));
      |     ^~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:40:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   40 |     hipStreamCreate(&stream);
      |     ^~~~~~~~~~~~~~~ ~~~~~~~
test_batch_mul.cpp:49:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   49 |     hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
      |     ^~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:52:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   52 |     hipMemcpyAsync(A_d, A.data(), M * K * N * sizeof(float), hipMemcpyHostToDevice, stream);
      |     ^~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:53:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   53 |     hipMemcpyAsync(B_d, B.data(), K * P * N * sizeof(float), hipMemcpyHostToDevice, stream);
      |     ^~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:62:95: error: too many arguments to function call, expected 4, have 7
   62 |     hipLaunchKernelGGL(batchedMatMulKernel, gridSize, blockSize, 0, stream, A_d, B_d, C_d, M, K, P, N);
      |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~
test_batch_mul.cpp:7:11: note: expanded from macro 'K'
    7 | #define K 1024  // Columns in matrix A and rows in matrix B
      |           ^
/opt/rocm-6.3.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:244:87: note: expanded from macro 'hipLaunchKernelGGL'
  244 | #define hipLaunchKernelGGL(kernelName, ...)  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)
      |                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~
/opt/rocm-6.3.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:241:78: note: expanded from macro 'hipLaunchKernelGGLInternal'
  241 |         kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);         \
      |         ~~~~~~~~~~                                                           ^~~~~~~~~~~
test_batch_mul.cpp:12:17: note: 'batchedMatMulKernel' declared here
   12 | __global__ void batchedMatMulKernel(float* A, float* B, float* C, int M, int K, int P, int N) {
      |                 ^                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:65:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   65 |     hipMemcpyAsync(C.data(), C_d, M * P * N * sizeof(float), hipMemcpyDeviceToHost, stream);
      |     ^~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:68:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   68 |     hipStreamEndCapture(stream, &graph);
      |     ^~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~
test_batch_mul.cpp:71:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   71 |     hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
      |     ^~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:78:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   78 |     hipGraphLaunch(graphExec, stream);
      |     ^~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~
test_batch_mul.cpp:79:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
   79 |     hipStreamSynchronize(stream);
      |     ^~~~~~~~~~~~~~~~~~~~ ~~~~~~
test_batch_mul.cpp:100:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  100 |     hipGraphExecDestroy(graphExec);
      |     ^~~~~~~~~~~~~~~~~~~ ~~~~~~~~~
test_batch_mul.cpp:101:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  101 |     hipGraphDestroy(graph);
      |     ^~~~~~~~~~~~~~~ ~~~~~
test_batch_mul.cpp:102:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  102 |     hipStreamDestroy(stream);
      |     ^~~~~~~~~~~~~~~~ ~~~~~~
test_batch_mul.cpp:103:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  103 |     hipFree(A_d);
      |     ^~~~~~~ ~~~
test_batch_mul.cpp:104:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  104 |     hipFree(B_d);
      |     ^~~~~~~ ~~~
test_batch_mul.cpp:105:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  105 |     hipFree(C_d);
      |     ^~~~~~~ ~~~
18 warnings and 2 errors generated when compiling for gfx1100.
failed to execute:/opt/rocm-6.3.1/lib/llvm/bin/clang++  --offload-arch=gfx1100 -O3 --driver-mode=g++ -O3 --hip-link  -x hip test_batch_mul.cpp -o "test_batch_mul"
