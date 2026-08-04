#! /bin.sh
# Intel Graphic Compute packages: intel-igc-core, intel-igc-opencl, libigc-dev,
#             libigc1, libsigc++-2.0-0v5
#  intel/clDNN: https://github.com/intel/clDNN


cmake -DCMAKE_BUILD_TYPE=Debug -DTHREADING=OMP -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=`which python3.8` -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_GPU=ON -DENABLE_OV_ONNX_FRONTEND=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DNGRAPH_PYTHON_BUILD_ENABLE=ON -DNGRAPH_ONNX_IMPORT_ENABLE=ON -DVERBOSE_BUILD=ON -DENABLE_WHEEL=ON ..
