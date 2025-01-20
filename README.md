heixiao@heixiao:~/Caiwo/HIP_tests$ hipcc test_conv_layer.cpp -o test_conv_layer
test_conv_layer.cpp:74:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   74 |                       reinterpret_cast<void*>(&inputWidth), 
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_conv_layer.cpp:75:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   75 |                       reinterpret_cast<void*>(&inputHeight), 
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_conv_layer.cpp:76:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   76 |                       reinterpret_cast<void*>(&kernelWidth), 
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_conv_layer.cpp:77:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   77 |                       reinterpret_cast<void*>(&kernelHeight), 
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_conv_layer.cpp:78:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   78 |                       reinterpret_cast<void*>(&outputWidth), 
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_conv_layer.cpp:79:23: error: reinterpret_cast from 'const int *' to 'void *' casts away qualifiers
   79 |                       reinterpret_cast<void*>(&outputHeight)};
      |                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


sudo docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --privileged -u $(id -u):$(id -g) rocm/rocm-terminal:latest
