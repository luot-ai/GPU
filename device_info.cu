#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << std::endl;
        std::cout << "-> " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA capable device(s)" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;

        // 推断共享内存的bank数（Volta及后续架构通常有32个内存银行）
        if (deviceProp.major >= 7) {  // Volta架构及以后 (Compute capability 7.x)
            std::cout << "  Shared memory bank count: 32 (Volta or newer architecture)" << std::endl;
        } else {
            std::cout << "  Shared memory bank count: Could vary depending on architecture" << std::endl;
        }
    }

    return 0;
}
