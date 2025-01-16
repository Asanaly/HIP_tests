ast-forward
 test_conv_layer.cpp | 163 +++++++++++++++++++++++++++++++++++++++++--------------------------------
 1 file changed, 91 insertions(+), 72 deletions(-)
heixiao@heixiao:~/Caiwo/HIP_tests$ hipcc test_conv_layer.cpp -o test_conv_layer
test_conv_layer.cpp:72:27: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   72 |                           &inputWidth, &inputHeight, 
      |                           ^~~~~~~~~~~
test_conv_layer.cpp:72:40: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   72 |                           &inputWidth, &inputHeight, 
      |                                        ^~~~~~~~~~~~
test_conv_layer.cpp:73:27: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   73 |                           &kernelWidth, &kernelHeight, 
      |                           ^~~~~~~~~~~~
test_conv_layer.cpp:73:41: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   73 |                           &kernelWidth, &kernelHeight, 
      |                                         ^~~~~~~~~~~~~
test_conv_layer.cpp:74:27: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   74 |                           &outputWidth, &outputHeight};
      |                           ^~~~~~~~~~~~
test_conv_layer.cpp:74:41: error: cannot initialize an array element of type 'void *' with an rvalue of type 'const int *'
   74 |                           &outputWidth, &outputHeight};
      |                                         ^~~~~~~~~~~~~
test_conv_layer.cpp:106:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  106 |     hipGraphExecDestroy(graphExec);
      |     ^~~~~~~~~~~~~~~~~~~ ~~~~~~~~~
test_conv_layer.cpp:107:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  107 |     hipGraphDestroy(graph);
      |     ^~~~~~~~~~~~~~~ ~~~~~
test_conv_layer.cpp:108:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  108 |     hipStreamDestroy(stream);
      |     ^~~~~~~~~~~~~~~~ ~~~~~~
test_conv_layer.cpp:110:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  110 |     hipFree(d_input);
      |     ^~~~~~~ ~~~~~~~
test_conv_layer.cpp:111:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  111 |     hipFree(d_kernel);
      |     ^~~~~~~ ~~~~~~~~
test_conv_layer.cpp:112:5: warning: ignoring return value of function declared with 'nodiscard' attribute [-Wunused-result]
  112 |     hipFree(d_output);
      |     ^~~~~~~ ~~~~~~~~
6 warnings and 6 errors generated when compiling for gfx1100.
failed to execute:/opt/rocm-6.3.1/lib/llvm/bin/clang++  --offload-arch=gfx1100 -O3 --driver-mode=g++ -O3 --hip-link  -x hip test_conv_layer.cpp -o "test_conv_layer"
