#include <iostream>
#include <cuda_runtime.h>

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess)                                                    \
    {                                                                            \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "            \
                  << "code: " << error << ", reason: "                           \
                  << cudaGetErrorString(error) << std::endl;                     \
        exit(1);                                                                 \
    }                                                                            \
}

int main() {
    int dev = 0;  // 指定要使用的GPU设备ID，通常0表示第一张GPU
    cudaDeviceProp devProp;
    
    // 获取指定设备的属性
    CHECK(cudaGetDeviceProperties(&devProp, dev));

    // 输出设备名称
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    
    // 输出SM的数量
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    
    // 输出每个线程块的共享内存大小，单位为KB
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    
    // 输出每个线程块的最大线程数
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    
    // 输出每个SM（流式多处理器）的最大线程数
    std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    
    // 输出每个SM的最大线程束（warp）数，每个warp包含32个线程
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    return 0;
}
