#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#if __APPLE__
  #define CL_TARGET_OPENCL_VERSION 120
  #include <OpenCL/opencl.h>
#else
  #define CL_TARGET_OPENCL_VERSION 200
  #include <CL/cl.h>
#endif