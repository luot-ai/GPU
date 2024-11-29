// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp
// nvprof ./test ./params/30epoch
// nvprof --profile-from-start off ./test ./params/30epoch
#include <random>
#include <iostream> 
#include <strings.h>
#include <vector>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cassert>



#define GEMMBLKMAX 128
#define ALIGN_DOWN(x, align) ((x) / (align) * (align))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define INDEX(row, col, width) ((row) * (width) + (col))
#define NPOINT 64
#define SAMPLE 1
#define USECONVMAX 0
#define CLASSNUM 10
#define DARKNETBLK 512
#define BLOCK 512
// #define USECONVMAX (SAMPLE == 0 ? 1 : (NPOINT >= 128 ? 1 : 0))

void checkCublasStatus(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            printf("CUBLAS_STATUS_SUCCESS\n");
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("CUBLAS_STATUS_NOT_INITIALIZED: The handle was not initialized properly.\n");
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed.\n");
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("CUBLAS_STATUS_INVALID_VALUE: Invalid parameters were passed to the function.\n");
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("CUBLAS_STATUS_ARCH_MISMATCH: The device architecture is not supported.\n");
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("CUBLAS_STATUS_EXECUTION_FAILED: Execution failed, possibly due to a previous CUDA error.\n");
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            printf("CUBLAS_STATUS_INTERNAL_ERROR: An internal operation failed.\n");
            break;
        default:
            printf("Unknown CUBLAS error.\n");
            break;
    }
}
dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / DARKNETBLK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*DARKNETBLK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}
void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}
void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}
void printVector_GPU(float* vec, int size) {
    printf("size:%d\n",size);
    // 在主机端创建一个标准向量以存储从设备复制的数据
    std::vector<float> vec_cpu(size);

    // 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(vec_cpu.data(), vec, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 打印向量内容
    for (const auto& value : vec_cpu) {
        std::cout << value << " ";
    }
    std::cout << std::endl; // 输出换行

}

/****************************************************************************************
 * 读取模型参数
 ****************************************************************************************/
// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::map<std::string, std::vector<float>> params;
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}
void read_params(std::string dir) {
    // std::string dir = "."; // 当前目录

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return ;
}

struct bn_layer {
    float* norm; 
    float* mean;   
    float* var;   
};

int cal_bn(int bn, int channel) {
    int norm = bn * channel;
    int mean = channel;
    int var  = channel;
    return norm + mean + var;
}

long long alloc_bn(bn_layer& l,float* device,long long offset,int channel,int bn){
    l.mean = device + offset;offset += channel;
    l.var  = device + offset;offset += channel;
    l.norm = device + offset;offset += channel*bn;
    return offset;
}

struct TNET {
    //stn3d
    float* input_trans;

    float* conv1_output_stn_cbr;
    float* conv2_output_stn_cbr;
    float* conv3_output_stn_cbr;

    bn_layer bn1_stn_cbr;
    bn_layer bn2_stn_cbr;
    bn_layer bn3_stn_cbr;
    float* relu1_output_stn_cbr;
    float* relu2_output_stn_cbr;
    float* CBR3_output;
    float* maxp_output;

    float* fc1_output_stn_cbr;
    float* fc2_output_stn_cbr;

    // bn_layer bn1_stn_fbr2f;
    // bn_layer bn2_stn_fbr2f;
    float* relu1_output_stn_fbr2f;
    float* relu2_output_stn_fbr2f;
    float* stn3d_out;
    //part2
    float* bmm1_res;
    float* bmm1_res_trans;
    float* fstn_input_conv;
    bn_layer fstn_input_bn;
    float* fstn_input;

    //stnkd
    float* conv1_output_fstn_cbr;
    float* conv2_output_fstn_cbr;
    float* conv3_output_fstn_cbr;

    bn_layer bn1_fstn_cbr;
    bn_layer bn2_fstn_cbr;
    bn_layer bn3_fstn_cbr;
    float* relu1_output_fstn_cbr;
    float* relu2_output_fstn_cbr;
    float* fstn_CBR3_output;
    float* fstn_maxp_output;

    float* fc1_output_fstn_fbr2f;
    float* fc2_output_fstn_fbr2f;

    // bn_layer bn1_fstn_fbr2f;
    // bn_layer bn2_fstn_fbr2f;
    float* relu1_output_fstn_fbr2f;
    float* relu2_output_fstn_fbr2f;
    float* stnkd_out;
    //part4
    float* fstn_input_trans;
    float* fstn_bmm1_res;
    float* fstn_bmm1_res_trans; // B C N
    float* cbr2_output;float* cbr2_output_conv;bn_layer cbr2_output_bn;
    float* feat_bn3;float* feat_bn3_conv;bn_layer feat_bn3_bn;
    float* encoder_output;
    //classify
    float* fc1_output_part5_fbr2f;float* relu1_output_part5_fbr2f;
    float* fc2_output_part5_fbr2f;float* relu2_output_part5_fbr2f;
    float* softmax_input;
    float* softmax_output;
};

long long cal_tnet_size(int batchSize, int numPoints, int inChannels){
    int ch = 1024;
    int ch_half = 512;
    int ch_quarter = 256;

    long long totalSize = 0;
    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = ch;
    int FC_OC1 = ch_half;
    int FC_OC2 = ch_quarter;
    //int FC_OC3 = 9;
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = ch;
    int fstn_FC_OC1 = ch_half;
    int fstn_FC_OC2 = ch_quarter;
    //int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    int encoderOC2 = 128;
    int encoderOC3 = ch;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    int transSize = batchSize * inChannels * inChannels;
    int transFeatSize = batchSize * fstn_inChannel * fstn_inChannel;
    //stn3d
    int stn_1 = bn*inChannels;
    int stn_2_conv = bn*OC1;
    int stn_3_conv = bn*OC2;
    int stn_4_conv = bn*OC3;
    int stn_234_bn = cal_bn(bn,OC1)+cal_bn(bn,OC2)+cal_bn(bn,OC3);
    int stn_2 = bn*OC1;
    int stn_3 = bn*OC2;
    int stn_4 = bn*OC3;
    if (USECONVMAX == 1) { stn_4 = stn_4 / 64 ;}
    int stn_5 = batchSize*OC3;
    int stn_6_fc = batchSize*FC_OC1;
    int stn_7_fc = batchSize*FC_OC2;
    int stn_6 = batchSize*FC_OC1;
    int stn_7 = batchSize*FC_OC2;
    int stn_8 = transSize;
    //part2
    int part2_1= batchSize*numPoints*encoderIC1 ;
    int part2_2= batchSize*encoderIC1*numPoints ;
    int part2_3= batchSize*fstn_inChannel*numPoints ;
    int part2_3_bn = cal_bn(bn,fstn_inChannel);
    int part2_4= batchSize*fstn_inChannel*numPoints ;
    //stnkd
    int fstn_1_conv= bn * fstn_OC1 ;
    int fstn_2_conv= bn * fstn_OC2 ;
    int fstn_3_conv= bn * fstn_OC3 ;
    int fstn_234_bn = cal_bn(bn,fstn_OC1)+cal_bn(bn,fstn_OC2)+cal_bn(bn,fstn_OC3);
    int fstn_1= bn * fstn_OC1 ;
    int fstn_2= bn * fstn_OC2 ;
    int fstn_3= bn * fstn_OC3 ;
    if (USECONVMAX == 1) { fstn_3 = fstn_3 / 64 ;}
    int fstn_4= batchSize * fstn_OC3 ;
    int fstn_5_fc= batchSize * fstn_FC_OC1 ;
    int fstn_6_fc= batchSize * fstn_FC_OC2 ;
    int fstn_5= batchSize * fstn_FC_OC1 ;
    int fstn_6= batchSize * fstn_FC_OC2 ;
    int fstn_7= transFeatSize ;
    //part4
    int part4_1= bn * fstn_inChannel ;
    int part4_2= batchSize*numPoints*fstn_inChannel ;
    int part4_3= batchSize*fstn_inChannel*numPoints ;
    int part4_4= batchSize*encoderOC2*numPoints ;int part4_4_conv= batchSize*encoderOC2*numPoints ;int part4_4_bn = cal_bn(bn,encoderOC2);
    int part4_5= bnEOC3 ;int part4_5_conv= bnEOC3 ; int part4_5_bn = cal_bn(bn,encoderOC3);
    if (USECONVMAX == 1) { part4_5 = part4_5 / 64 ;}
    int part4_6= batchSize * encoderOC3 ;
    //classify
    int cla_1= batchSize * ch_half ;int cla_1_fc= batchSize * ch_half ;
    int cla_2= batchSize * ch_quarter ;int cla_2_fc= batchSize * ch_quarter ;
    int cla_3= batchSize * 10;
    int cla_4= batchSize * 10;

    totalSize = stn_1 + stn_2 + stn_3 + stn_4 + stn_5 + stn_6 + stn_7 + stn_8 +
    part2_1 + part2_2 + part2_3 +
    fstn_1 + fstn_2 + fstn_3 + fstn_4 + fstn_5 + fstn_6 + fstn_7 + 
    part4_1 + part4_2 + part4_3 + part4_4 + part4_5 + part4_6 + 
    cla_1 + cla_2 + cla_3;

    totalSize += stn_2_conv + stn_3_conv + stn_4_conv + stn_6_fc + stn_7_fc + part2_4 +
    fstn_1_conv + fstn_2_conv + fstn_3_conv + fstn_5_fc + fstn_6_fc +
    part4_4_conv + part4_5_conv + cla_1_fc + cla_2_fc + cla_4;

    totalSize += stn_234_bn + part2_3_bn + fstn_234_bn + part4_4_bn + part4_5_bn;

    return totalSize;
}

struct fcp {
    float* weight; // Conv weight
    float* bias;   // Conv bias
};
void read_fcp(const std::string& layer, fcp& wbp,int i) {
    std::string fiStr = std::to_string(i);;
    std::string name = layer + "fc" + fiStr;  
    //std::cout << name << std::endl;
    cudaMalloc((void**)&wbp.weight, params[name + ".weight"].size() * sizeof(float));
    cudaMalloc((void**)&wbp.bias, params[name + ".bias"].size() * sizeof(float));
    cudaMemcpy(wbp.weight, params[name + ".weight"].data(), params[name + ".weight"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbp.bias, params[name + ".bias"].data(), params[name + ".bias"].size() * sizeof(float), cudaMemcpyHostToDevice);
}
void free_fcp(fcp& wbp){
    cudaFree(wbp.bias);
    cudaFree(wbp.weight);
}

struct wbBnP {
    float* weight; // Conv weight
    float* bias;   // Conv bias
    float* bn_weight; // BatchNorm weight
    float* bn_bias;   // BatchNorm bias
    float* bn_mean;   // BatchNorm running mean
    float* bn_var;    // BatchNorm running var
};
void read_wbBnP(const std::string& layer,const std::string& cf,wbBnP& wbBnP,int i,int param_offset=0) {

    std::string cfiStr = std::to_string(i);
    std::string biStr = std::to_string(i+param_offset);
    std::string name = layer + cf + cfiStr;
    std::string bnStr = layer + "bn" + biStr;   
    //std::cout << name << std::endl;
    //std::cout << bnStr << std::endl;
    cudaMalloc((void**)&wbBnP.weight, params[name + ".weight"].size() * sizeof(float));
    cudaMalloc((void**)&wbBnP.bias, params[name + ".bias"].size() * sizeof(float));
    cudaMalloc((void**)&wbBnP.bn_weight, params[bnStr + ".weight"].size() * sizeof(float));
    cudaMalloc((void**)&wbBnP.bn_bias, params[bnStr + ".bias"].size() * sizeof(float));
    cudaMalloc((void**)&wbBnP.bn_mean, params[bnStr + ".running_mean"].size() * sizeof(float));
    cudaMalloc((void**)&wbBnP.bn_var, params[bnStr + ".running_var"].size() * sizeof(float));
    cudaMemcpy(wbBnP.weight, params[name + ".weight"].data(), params[name + ".weight"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbBnP.bias, params[name + ".bias"].data(), params[name + ".bias"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbBnP.bn_weight, params[bnStr + ".weight"].data(), params[bnStr + ".weight"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbBnP.bn_bias, params[bnStr + ".bias"].data(), params[bnStr + ".bias"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbBnP.bn_mean, params[bnStr + ".running_mean"].data(), params[bnStr + ".running_mean"].size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wbBnP.bn_var, params[bnStr + ".running_var"].data(), params[bnStr + ".running_var"].size() * sizeof(float), cudaMemcpyHostToDevice);
}
void free_wbBnP(wbBnP& wbBnP){
    cudaFree(wbBnP.bias);
    cudaFree(wbBnP.weight);
    cudaFree(wbBnP.bn_bias);
    cudaFree(wbBnP.bn_mean);
    cudaFree(wbBnP.bn_var);
    cudaFree(wbBnP.bn_weight);
}

struct CB3P {
    wbBnP cb1;
    wbBnP cb2;
    wbBnP cb3;
};
void read_CB3P(const std::string& layer,CB3P& CB3P) {
    read_wbBnP(layer,"conv",CB3P.cb1,1);
    read_wbBnP(layer,"conv",CB3P.cb2,2);
    read_wbBnP(layer,"conv",CB3P.cb3,3);   
}
void free_CB3P(CB3P &CB3P){
    free_wbBnP(CB3P.cb1);
    free_wbBnP(CB3P.cb2);
    free_wbBnP(CB3P.cb3);
}


struct FB2FP {
    wbBnP fb1;
    wbBnP fb2;
    fcp   f3;
};
void read_FB2FP(const std::string& layer,FB2FP& FB2FP,int param_offset=0)    {
    read_wbBnP(layer,"fc",FB2FP.fb1,1,param_offset);
    read_wbBnP(layer,"fc",FB2FP.fb2,2,param_offset);
    read_fcp(layer,FB2FP.f3,3);
}
void free_FB2FP(FB2FP &FB2FP){
    free_wbBnP(FB2FP.fb1);
    free_wbBnP(FB2FP.fb2);
    free_fcp(FB2FP.f3);
}

struct stndP {
    CB3P cb3;
    FB2FP fb2f;
};
void read_stndP(const std::string& layer,stndP& stndP) {
    read_CB3P(layer,stndP.cb3);
    read_FB2FP(layer,stndP.fb2f,3);
}
void free_stndP(stndP& stndP){
    free_CB3P(stndP.cb3);
    free_FB2FP(stndP.fb2f);
}

struct cudaP {
    stndP stn3dp;
    stndP stnkdp;
    CB3P  featp;
    FB2FP nonep;
};
void freeDP(cudaP &dp)
{
    free_stndP(dp.stn3dp);
    free_stndP(dp.stnkdp);
    free_CB3P(dp.featp);
    free_FB2FP(dp.nonep);
}

cudaP dParams;


/****************************************************************************************
 * 读取训练集数据
 ****************************************************************************************/

using namespace H5;
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集名称
        std::vector<std::string> dataset_names;
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; i++) {
            dataset_names.push_back(file.getObjnameByIdx(i));
        }

        // 读取每个数据集
        for (const auto& name : dataset_names) {
            DataSet dataset = file.openDataSet(name + "/points");
            DataSpace dataspace = dataset.getSpace();

            // 获取数据集的维度
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, NULL);

            // 读取数据
            std::vector<float> points(dims[0] * dims[1]);
            dataset.read(points.data(), PredType::NATIVE_FLOAT);

            // 存储点云数据
            list_of_points.push_back(points);

            // 读取标签
            Attribute label_attr = file.openGroup(name).openAttribute("label");
            int label;
            label_attr.read(PredType::NATIVE_INT, &label);

            // 存储标签
            list_of_labels.push_back(label);
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    }
}





/****************************************************************************************
 * 网络搭建
 ****************************************************************************************/
__global__ void LogSoftMax_Kernel_train(int* label,float* input,float* output,float* outputDelta,int* correct_tabel,int L,int BatchSize = 32)
{
    int bx = blockIdx.x;
    int index = bx;
    float softmax[CLASSNUM]; 
    if (index < gridDim.x)
    {
        float sum = 0.0f;
        for (int l = 0; l < L; l++)
        {
            int iIdx = l + index * L;
            softmax[l] = exp(input[iIdx]);
            sum += softmax[l];
        }
        for (int l = 0; l < L; l++)
        {
            softmax[l] = softmax[l]/sum;
        }
        //计算最大值索引，并在output中写入log值
        float logvalue = log(softmax[0]);
        output[index*L] = logvalue;
        float max_value = logvalue;
        int max_index = 0;
        for (int l = 1; l < L; l++)
        {
            int iIdx = l + index * L;
            logvalue = log(softmax[l]);
            if (logvalue > max_value)
            {
                max_value = logvalue;
                max_index = l;
            }
            output[iIdx] = logvalue;
        }
        //用于计算正确率
        int correct_label = label[index];
        correct_tabel[index] = (max_index == correct_label)?1:0;
        //printf("label:%d,predict:%d\n",correct_label,max_index);
        //printf("correct_tabel:%d\n",correct_tabel[index]);
        //反向传播
        // for (int l = 0; l < L; l++)
        // {
        //     int iIdx = l + index * L;
        //     outputDelta[iIdx] = softmax[l]-(correct_label==l);
        // }
    }
}
void LogSoftMax_GPU_train(int* label,float* input,float* output,float* outputDelta,int* correct_tabel,int L,int BatchSize = 32)
{   
    dim3 blockDim(1);
    dim3 gridDim(BatchSize);
    LogSoftMax_Kernel_train<<<gridDim,blockDim>>>(label,input,output,outputDelta,correct_tabel,L,BatchSize);
}

__global__ void LogSoftMax_Kernel(float *input,int *label, int L, int BatchSize = 32)
{
    int bx = blockIdx.x;
    int index = bx;
    if (index < gridDim.x)
    {
        float sum = 0;
        for (int l = 0; l < L; l++)
        {
            int iIdx = l + index * L;
            input[iIdx] = exp(input[iIdx]);
            sum += input[iIdx];
        }
        float max_value = log(input[index * L]/sum);
        int max_index = 0;
        for (int l = 1; l < L; l++)
        {
            int iIdx = l + index * L;
            float output = log(input[iIdx]/sum);
            if (output > max_value)
            {
                max_value = output;
                max_index = l;
            }
        }
        label[index] = max_index;
    }
}
void LogSoftMax_GPU(float* input,int* label,int L,int BatchSize = 32)
{   
    dim3 blockDim(1);
    dim3 gridDim(BatchSize);
    LogSoftMax_Kernel<<<gridDim,blockDim>>>(input,label,L,BatchSize);
}

__global__ void matrix_add_I_kernel(float *input, int n,int batchSize)
{
    int curN = blockIdx.x;
    int curB = threadIdx.x;
    int index = curB* gridDim.x + curN;
    int iidx =  index*n +curN;
    if (index < n * batchSize )
        input[iidx] = input[iidx] + 1.0f;
}
void matrix_add_I(float *input, int n,int batchSize)
{
    dim3 blockDim(batchSize);
    dim3 gridDim(n);
    matrix_add_I_kernel<<<gridDim, blockDim>>>(input, n, batchSize);
    // cudaDeviceSynchronize();
}

__global__ void Maxpooling_Kernel0(float* input,float* output,int numPoints)
{
    __shared__ float sharedMax[1024];
    
    int tx = threadIdx.x;
    int channel = blockIdx.x;

    int cnum = channel * numPoints;
    float localMax = input[cnum + tx];
    sharedMax[tx]=localMax;
    __syncthreads();

    // 归约：逐步计算块内的最大值
    for (int stride = numPoints/ 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            if (sharedMax[tx + stride] > sharedMax[tx]) {
                sharedMax[tx] = sharedMax[tx + stride];
            }
        }
        __syncthreads();
    }

    // 线程0写入最终的最大值
    if (tx == 0) {
        output[channel] = sharedMax[0];
    }
}
__global__ void Maxpooling_Kernel(float* input,float* output,int numPoints,int perWarp,int perTh)
{
    __shared__ float sharedMax[32];
    
    int tx = threadIdx.x;
    int channel = blockIdx.x;
    int warpIdx = tx / 32;

    float localMax = -FLT_MAX;
    int startIdx = channel * numPoints + warpIdx * perWarp + tx;
    for (int i = 0; i < perTh; i ++) {
        int index = startIdx + i*32;
        localMax = max(localMax,input[index]);
    }

    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        localMax = max(localMax, __shfl_down_sync(0xFFFFFFFF, localMax, offset));
    }
    if (tx % 32 == 0) {
        sharedMax[warpIdx] = localMax;
    }
    __syncthreads();
    if (warpIdx == 0)
    {
        localMax = sharedMax[tx];
        for (int offset = 32 / 2; offset > 0; offset >>= 1) {
            localMax = max(localMax, __shfl_down_sync(0xFFFFFFFF, localMax, offset));
        }
        if (tx == 0)
        {
            output[channel] = localMax;
        }
    }

}
void GPU_MaxPooling(int ics, int batchSize, int numPoints,float* input, float* output)
{
    //std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    if(numPoints>1024)
    {
    dim3 blockDim(1024);
    int warpNum = 32;
    int perWarp = numPoints/warpNum;//2N/32
    int perTh = perWarp/32;
    Maxpooling_Kernel<<<gridDim, blockDim>>>(input, output,numPoints,perWarp,perTh);
    }
    else
    {
        Maxpooling_Kernel0<<<gridDim, numPoints>>>(input, output,numPoints);
    }
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    // printf("TA: %d, TB: %d, M: %d, N: %d, K: %d, ALPHA: %f, BETA: %f\n", TA, TB, M, N, K, ALPHA, BETA);
    // printf("lda: %d, ldb: %d, ldc: %d\n", lda, ldb, ldc);
    // printf("A_gpu: %p, B_gpu: %p, C_gpu: %p\n", A_gpu, B_gpu, C_gpu);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed with error: ");
        checkCublasStatus(status);
    }
    cublasDestroy(handle);
}
__global__ void BMM_Kernel(float* input_A,float* input_B,float* output,int M_A,int K_A,int K_B,int N_B,int BatchSize)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //int bz = blockIdx.z;

    int col = tx + bx * blockDim.x;
    int row = ty + by * blockDim.y;
    int batch = blockIdx.z;

    if (row < M_A && col < N_B)
    {
        float tmp = 0.0f;
        for (int k =0;k<K_A;k++)
        {
            tmp += input_A[batch * M_A * K_A + row * K_A + k] * input_B[batch * K_B * N_B + k * N_B + col];
        }
        output[batch*M_A*N_B+row*N_B+col] = tmp;
    }
}
void GPU_Bmm(float* input_A,float* input_B,float* output,int M_A,int K_A,int K_B,int N_B,int BatchSize = 1)
{
    //std::cout << "--------BMM" << std::endl;
    for(int b=0;b<BatchSize;b++)
    {
        check_error(cudaPeekAtLastError());
        
        // 强制同步，确保没有挂起的 CUDA 操作
        cudaDeviceSynchronize();
        gemm_gpu(false,false,M_A,N_B,K_A,1.0f, input_A+b*M_A*K_A,K_A,input_B+b*K_B*N_B,N_B,0.0f,output+b*M_A*N_B,N_B);
        check_error(cudaPeekAtLastError());
    }
    
    
    // const int BLK_X = 32;
    // const int BLK_Y = 32;
    // dim3 blockDim(BLK_X, BLK_Y);
    // dim3 gridDim((N_B + BLK_X - 1) / BLK_X, (M_A + BLK_Y - 1) / BLK_Y,BatchSize);//X:宽度 Y：高度
    // BMM_Kernel<<<gridDim, blockDim>>>(input_A, input_B, output, M_A, K_A, K_B, N_B, BatchSize);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());
    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void transpose_Kernel(float* input,float* output,int dim0,int dim1,int dim2)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int idx = tx + bx * blockDim.x;
    int idy = ty + by * blockDim.y;
    int index = idx + idy * dim1;

    if (idx < dim1 && idy < dim2)
    {
        for (int b=0;b<dim0;b++)
        {
            int bdd = b*dim1*dim2;
            output[bdd+index]=input[bdd+idx*dim2+idy];
        }
    }
}
void GPU_transpose(float* input,float* output,int dim0,int dim1,int dim2)
{
    const int BLK_X = 32;
    const int BLK_Y = 32;

    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((dim1 + BLK_X -1)/BLK_X,  (dim2+BLK_Y-1)/BLK_Y);
    transpose_Kernel<<<gridDim, blockDim>>>(input,output,dim0,dim1,dim2);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void linear_Kernel(int M,int batchSize,int N,float* input, 
float* fcWeights, float* fcBias,float* output)
{
    // Block index
    int bx = blockIdx.x;
    int batch = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;//0~31
    int ty = threadIdx.y;//0~3

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_row = 4 * bx + ty;

    if(current_row < M){

        int oc = current_row;
        float res = 0.0f;

        int kIteration = (N/warp_size)/4;
        if(kIteration==0) kIteration=1;
        // fcWeights = &fcWeights[current_row*N];
        // input= &input[batch*N];
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col_vec = (i*warp_size + laneId);
            float4 current_val= reinterpret_cast<float4 *>(fcWeights)[current_col_vec+current_row*N/4];
            float4 current_x = reinterpret_cast<float4 *>(input)[current_col_vec+batch*N/4];
            res += current_val.x*current_x.x;
            res += current_val.y*current_x.y;
            res += current_val.z*current_x.z;
            res += current_val.w*current_x.w;
        }

        res += __shfl_down_sync(0xffffffff, res, 16); // 0-16, 1-17, 2-18, etc.
        res += __shfl_down_sync(0xffffffff, res, 8);// 0-8, 1-9, 2-10, etc.
        res += __shfl_down_sync(0xffffffff, res, 4);// 0-4, 1-5, 2-6, etc.
        res += __shfl_down_sync(0xffffffff, res, 2);// 0-2, 1-3, 4-6, 5-7, etc.
        res += __shfl_down_sync(0xffffffff, res, 1);// 0-1, 2-3, 4-5, etc.

        
        res+=fcBias[oc];
        int index = oc + batch * M;
        if(laneId==0) output[index] = res;
    }
}
void Linear_GPU(int batchSize,int inFeatures, int outFeatures,float* cudaWeights,float* cudaBias,float* input,float* output){
    //std::cout << "------------LAYER:linear" << std::endl;
dim3 blockDim(32,4);
dim3 gridDim((outFeatures + 4 - 1) / 4,batchSize);//X:宽度 Y：高度
linear_Kernel<<<gridDim,blockDim>>>(outFeatures,batchSize,inFeatures,input,cudaWeights,cudaBias,output);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());
    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void ReLu_Kernel(float *input,float *output)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int idx = tx + bx * blockDim.x;
    int index = idx ;

    output[index] = input[index] > 0 ? input[index] : 0;

}
void ReLU_GPU(int batchSize,int numPoints,int OC,float* input,float* output){
    //std::cout << "------------LAYER:relu" << std::endl;
    // const int BLK_X = 32;
    // const int BLK_Y = 32;

    dim3 blockDim(OC);
    dim3 gridDim(batchSize*numPoints);
    ReLu_Kernel<<<gridDim, blockDim>>>(input, output);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void BatchNorm1d_Kernel(int numPoints,float* weight,float* bias,float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx + bx * blockDim.x;
    int index = idx;

    if (idx < blockDim.x * gridDim.x)
    {
        float mean = running_mean[tx];
        float var = running_var[tx];
        for (int n = 0; n < numPoints; n++)
        {
            int iIdx = index * numPoints + n;
            output[iIdx] = (input[iIdx] - mean) / sqrt(var + esp) * weight[tx] + bias[tx];
        }
    }
}
void BatchNorm1d_GPU(int numFeatures, int batchSize, int numPoints,float* weight,float* bias,float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    float* cudaWeights;
    float* cudaBias;
    float* cudaRV;
    float* cudaRM;

    cudaMalloc((void **)&cudaWeights, numFeatures * sizeof(float));
    cudaMalloc((void **)&cudaBias, numFeatures * sizeof(float));
    cudaMalloc((void **)&cudaRV, numFeatures * sizeof(float));
    cudaMalloc((void **)&cudaRM, numFeatures * sizeof(float));

    cudaMemcpy(cudaWeights, weight, numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBias, bias, numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaRV, running_var, numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaRM, running_mean, numFeatures * sizeof(float), cudaMemcpyHostToDevice);

    //std::cout << "------------LAYER:batchnorm" << std::endl;
    dim3 blockDim(numFeatures);
    dim3 gridDim(batchSize);
    BatchNorm1d_Kernel<<<gridDim, blockDim>>>(numPoints,cudaWeights,cudaBias,cudaRM,cudaRV,input,output);

    cudaFree(cudaWeights);
    cudaFree(cudaBias);
    cudaFree(cudaRV);
    cudaFree(cudaRM);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}

//ARCH CB
__global__ void CB_1024x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 128;
    int BN = 128;
    // param-set : fix
    int BK = 8;
    int Tsize = 8; //thread 8*8 = 2*2 * 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *W_shared = reinterpret_cast<float *>(smem);
    float *I_shared = reinterpret_cast<float *>(smem + 16 * 1024);
    float W_ldg_reg[4];
    float I_ldg_reg[4];

    float W_reg[2][8]={0};
    float I_reg[2][8]={0};
    float O_reg[8][8] = {0};

    int W_grow = by * BM + tx / BK * 4; // 每BK个threads:连续4行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复四次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    const char *W_ldg_ptr = (const char *)(convWeights+W_LoadG);
    const char *I_ldg_ptr = (const char *)(input + I_LoadG);

    int W_srow = tx % BK; // 转置
    int W_scol = tx / BK * 4;
    int I_srow = tx / 32;
    int I_scol = tx % 32;
    int W_StoreS = INDEX(W_srow, W_scol, BM + 4);
    int I_StoreS = INDEX(I_srow, I_scol, BN);
    uint32_t W_sts_addr = smem_u32addr(W_shared + W_StoreS);
    uint32_t I_sts_addr = smem_u32addr(I_shared + I_StoreS);

    int W_LoadS = INDEX(0, (wy * 32 + twy * 4), BM + 4);
    int I_LoadS = INDEX(0, (wx * 64 + twx * 4), BN);
    uint32_t W_lds_addr = smem_u32addr(W_shared + W_LoadS);
    uint32_t I_lds_addr = smem_u32addr(I_shared + I_LoadS);

    //1st_tile:
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
    }
    sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    W_ldg_ptr += BK * sizeof(float);
    I_ldg_ptr += BK * N * sizeof(float);
    W_sts_addr ^= 0x2000;
    I_sts_addr ^= 0x1000;

    lds128(W_reg[0][0], W_reg[0][1], W_reg[0][2], W_reg[0][3], W_lds_addr);
    lds128(W_reg[0][4], W_reg[0][5], W_reg[0][6], W_reg[0][7], W_lds_addr + 4 * 4 * sizeof(float));
    lds128(I_reg[0][0], I_reg[0][1], I_reg[0][2], I_reg[0][3], I_lds_addr);
    lds128(I_reg[0][4], I_reg[0][5], I_reg[0][6], I_reg[0][7], I_lds_addr + 4 * 8 * sizeof(float));

    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < (K / BK - 1); phase++)
    {
        // ITERATIONS : BK times
        #pragma unroll
        for (int iter = 0; iter < BK ;iter++)
        {
            // next phase: ldreg->share
            if(iter == BK - 1)
            {
                sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
                }
                __syncthreads();
                W_ldg_ptr += BK * sizeof(float);
                I_ldg_ptr += BK * N * sizeof(float);
                W_lds_addr ^= 0x2000;
                I_lds_addr ^= 0x1000;
                W_sts_addr ^= 0x2000;
                I_sts_addr ^= 0x1000;
            }
            // next iter: share->registers
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
            // next phase:global->ldreg
            if (iter == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
                }
            }
            // calculate
            {
            int cI = iter % 2;
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
            }
        }
    }
    // LAST PHASE
    #pragma unroll
    for (int iter = 0 ; iter < BK ; iter++)
    {
        // next iter: share->registers
        if (iter < (BK -1))
        {
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
        }
        // calculate
        {
            int cI = iter % 2;
#pragma unroll
            for (int i = 0; i < Tsize; ++i)
            {
#pragma unroll
                for (int j = 0; j < Tsize; ++j)
                {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
        }
    }

     int O_grow = by * BM + wy * 32 + twy * 4;
    // int O_gcol = bx * BN + wx * 64 + twx * 4;
    //int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float mean[8] = {0};
    float var[8] = {0};
    float bnW[8] = {0};
    float bnB[8] = {0};
    float cvB[8] = {0};
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int sIdx = i * 4 + j;
            int gIdx = O_grow + i*16+j;
            cvB[sIdx] = convBias[gIdx];
            mean[sIdx] = bnRM[gIdx];
            var[sIdx] = bnRV[gIdx];
            bnW[sIdx] = bnWeights[gIdx];
            bnB[sIdx] = bnBias[gIdx];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            float res2 = O_reg[i][j+4];
            float res3 = O_reg[i+4][j];
            float res4 = O_reg[i+4][j+4];
            res1 += cvB[i];
            res2 += cvB[i];
            res3 += cvB[i+4];
            res4 += cvB[i+4];
            
            res1 = __fdividef((res1 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res2 = __fdividef((res2 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res3 = __fdividef((res3 - mean[i+4]),sqrt(var[i+4] + esp))* bnW[i+4] + bnB[i+4];
            res4 = __fdividef((res4 - mean[i+4]),sqrt(var[i+4] + esp)) * bnW[i+4] + bnB[i+4];
            O_reg[i][j] =  res1;
            O_reg[i][j+4] = res2;
            O_reg[i+4][j] = res3;
            O_reg[i+4][j+4] = res4;
        }
    }
    #pragma unroll
    for (int i = 0;i <8 ;i++)
    {
        float maxT = O_reg[i][0];
        for (int j =1 ;j<8;j++)
        {
            if (O_reg[i][j]>maxT)
            {
                maxT = O_reg[i][j];
            }
        }
    O_reg[i][0]= maxT;
    }
    
    //warp内最大值规约 th 0 1 16 17 ，各8行
    #pragma unroll
    for (int offset = 8; offset > 1; offset >>= 1) {
        for (int i = 0;i < 8 ;i++)
        {
            O_reg[i][0] = max(O_reg[i][0], __shfl_down_sync(0xFFFFFFFF, O_reg[i][0], offset));
        }
    }
    if (twx == 0)
    {
        //printf("tx is %d\n",tx);
        int O_gcol = bx * 2 + wx ;
        #pragma unroll
        for (int i = 0;i<2;i++)
        {
            for (int j =0 ;j <4;j++)
            {
                int O_grow = by * BM + wy * 32 + twy * 4 + i * 16 + j;
                int O_StoreG = INDEX(O_grow, O_gcol, N/64) + bO/64;
                output[O_StoreG] = O_reg[i*4+j][0];
            }
        }
        
    }

    // if (twx == 0)
    // {
    //     int O_StoreS_0 = wx * 32 + twy * 4;  
    //     int O_StoreS_1 = O_StoreS_0 + 16;  
    //     uint32_t O_sts_addr_0 = smem_u32addr(W_shared + O_StoreS_0);
    //     uint32_t O_sts_addr_1 = smem_u32addr(W_shared + O_StoreS_1);
    // }


    // C_tile write back, reuse A&B tile shared memory buffer
    // uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warpIdx * 2048) +
    //                                    twy * 4 * 8 + twx);//每个warp 32*64 =2048；每个twy 
    // const float *C_lds_ptr = (float *)(smem + warpIdx * 2048) + twIdx;

    // uint32_t m_idx = blockIdx.y * 128 + warpIdx / 2 * 32;
    // uint32_t n_idx = blockIdx.x * 128 + warpIdx % 2 * 64 + twIdx;

    // float *C_stg_ptr = output + m_idx * N + n_idx+bO;

    
        // #pragma unroll
        // for (int i = 0; i < 2; ++i) {
        //     #pragma unroll
        //     for (int j = 0; j < 2; ++j) {
        //         StgFrag stg_frag(O_reg, j, i);//4*4 matrix

        //         C_tile_wb(stg_frag,
        //                   C_stg_ptr + i * 16 * N + j * 32,
        //                   C_lds_ptr,
        //                   C_sts_addr,
        //                   M,
        //                   N,
        //                   m_idx + i * 16,
        //                   n_idx + j * 32);
        //     }
        // }
}
__global__ void CB_64x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 64;
    int BN = 64;
    // param-set : fix
    int BK = 8;
    int Tsize = 4; //thread 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ float W_shared[512];//1024*4B = 4KB
    __shared__ float I_shared[512];//1024*4B = 4KB
    float W_reg[4]={0};
    float I_reg[4]={0};
    float O_reg[4][4] = {0};

    int W_grow = by * BM + tx / BK * 2; // 每BK个threads:连续2行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复2次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < K / BK; phase++)
    {
        //【global -> share】
        int W_srow = tx % BK ; //转置
        int W_scol = tx / BK * 2 ; 
        int I_srow = tx / 32; 
        int I_scol = tx % 32; 
        int W_StoreS = INDEX(W_srow,W_scol,BM);
        int I_StoreS = INDEX(I_srow,I_scol,BN);
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            W_shared[W_StoreS+ldg]=convWeights[W_LoadG+ldg*K];
        }
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            I_shared[I_StoreS+ldg*32]=input[I_LoadG+ldg*32];
        }
        __syncthreads();
        W_LoadG += BK;
        I_LoadG += BK * N;
        // ITERATIONS : BK times
        for (int iter = 0; iter< BK ;iter++)
        {
            //【share -> registers】
            int W_LoadS = INDEX(iter, (wy * 16 + twy * 4), BM);
            int I_LoadS = INDEX(iter, (wx * 32 + twx * 4), BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                W_reg[i] = W_shared[W_LoadS+i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                I_reg[i] = I_shared[I_LoadS+i];
            }
            // calculate
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[i] * I_reg[j];
                }
            }
        }
    }
    int O_grow = by * BM + wy * 16 + twy * 4;
    int O_gcol = bx * BN + wx * 32 + twx * 4;
    int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float mean[4] = {0};
    float var[4] = {0};
    float bnW[4] = {0};
    float bnB[4] = {0};
    float cvB[4] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int sIdx = j;
        int gIdx = O_grow + j;
        cvB[sIdx] = convBias[gIdx];
        mean[sIdx] = bnRM[gIdx];
        var[sIdx] = bnRV[gIdx];
        bnW[sIdx] = bnWeights[gIdx];
        bnB[sIdx] = bnBias[gIdx];
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            res1 += cvB[i];
            res1 = (res1 - mean[i]) / sqrt(var[i] + esp) * bnW[i] + bnB[i];
            O_reg[i][j] = res1;
        }
    }
    //store to C
    #pragma unroll
    for (int i = 0; i<4;i++)
    {
        for (int j = 0 ; j<4 ;j++)
        {
            output[O_StoreG+ i*N+j]=O_reg[i][j];
        }
    }
}
void CBWRAP_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, 
float* cudaConvWeights, float* cudaConvBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    //std::cout << "------------LAYER:CBWRAP" << std::endl;
    if (USECONVMAX)
    {
        dim3 grid128(DIV_UP(numPoints, 128), DIV_UP(outChannels, 128), batchSize);
        CB_1024x128N_kernel<<<grid128, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else
    {
        dim3 grid64(DIV_UP(numPoints, 64), DIV_UP(outChannels, 64), batchSize);
        CB_64x128N_kernel<<<grid64, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
}

//ARCH CBR
__global__ void CBR_64x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 64;
    int BN = 64;
    // param-set : fix
    int BK = 8;
    int Tsize = 4; //thread 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ float W_shared[512];//1024*4B = 4KB
    __shared__ float I_shared[512];//1024*4B = 4KB
    float W_reg[4]={0};
    float I_reg[4]={0};
    float O_reg[4][4] = {0};

    int W_grow = by * BM + tx / BK * 2; // 每BK个threads:连续2行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复2次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < K / BK; phase++)
    {
        //【global -> share】
        int W_srow = tx % BK ; //转置
        int W_scol = tx / BK * 2 ; 
        int I_srow = tx / 32; 
        int I_scol = tx % 32; 
        int W_StoreS = INDEX(W_srow,W_scol,BM);
        int I_StoreS = INDEX(I_srow,I_scol,BN);
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            W_shared[W_StoreS+ldg]=convWeights[W_LoadG+ldg*K];
        }
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            I_shared[I_StoreS+ldg*32]=input[I_LoadG+ldg*32];
        }
        __syncthreads();
        W_LoadG += BK;
        I_LoadG += BK * N;
        // ITERATIONS : BK times
        for (int iter = 0; iter< BK ;iter++)
        {
            //【share -> registers】
            int W_LoadS = INDEX(iter, (wy * 16 + twy * 4), BM);
            int I_LoadS = INDEX(iter, (wx * 32 + twx * 4), BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                W_reg[i] = W_shared[W_LoadS+i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                I_reg[i] = I_shared[I_LoadS+i];
            }
            // calculate
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[i] * I_reg[j];
                }
            }
        }
    }
    int O_grow = by * BM + wy * 16 + twy * 4;
    int O_gcol = bx * BN + wx * 32 + twx * 4;
    int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float mean[4] = {0};
    float var[4] = {0};
    float bnW[4] = {0};
    float bnB[4] = {0};
    float cvB[4] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int sIdx = j;
        int gIdx = O_grow + j;
        cvB[sIdx] = convBias[gIdx];
        mean[sIdx] = bnRM[gIdx];
        var[sIdx] = bnRV[gIdx];
        bnW[sIdx] = bnWeights[gIdx];
        bnB[sIdx] = bnBias[gIdx];
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            res1 += cvB[i];
            res1 = (res1 - mean[i]) / sqrt(var[i] + esp) * bnW[i] + bnB[i];
            res1 = max(0.0f, res1);
            O_reg[i][j] = res1;
        }
    }
    //store to C
    #pragma unroll
    for (int i = 0; i<4;i++)
    {
        for (int j = 0 ; j<4 ;j++)
        {
            output[O_StoreG+ i*N+j]=O_reg[i][j];
        }
    }
}
__global__ void CBR_128x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 128;
    int BN = 128;
    // param-set : fix
    int BK = 8;
    int Tsize = 8; //thread 8*8 = 2*2 * 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *W_shared = reinterpret_cast<float *>(smem);
    float *I_shared = reinterpret_cast<float *>(smem + 16 * 1024);
    float W_ldg_reg[4];
    float I_ldg_reg[4];

    float W_reg[2][8]={0};
    float I_reg[2][8]={0};
    float O_reg[8][8] = {0};

    int W_grow = by * BM + tx / BK * 4; // 每BK个threads:连续4行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复四次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    const char *W_ldg_ptr = (const char *)(convWeights+W_LoadG);
    const char *I_ldg_ptr = (const char *)(input + I_LoadG);

    int W_srow = tx % BK; // 转置
    int W_scol = tx / BK * 4;
    int I_srow = tx / 32;
    int I_scol = tx % 32;
    int W_StoreS = INDEX(W_srow, W_scol, BM + 4);
    int I_StoreS = INDEX(I_srow, I_scol, BN);
    uint32_t W_sts_addr = smem_u32addr(W_shared + W_StoreS);
    uint32_t I_sts_addr = smem_u32addr(I_shared + I_StoreS);

    int W_LoadS = INDEX(0, (wy * 32 + twy * 4), BM + 4);
    int I_LoadS = INDEX(0, (wx * 64 + twx * 4), BN);
    uint32_t W_lds_addr = smem_u32addr(W_shared + W_LoadS);
    uint32_t I_lds_addr = smem_u32addr(I_shared + I_LoadS);

    //1st_tile:
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
    }
    sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    W_ldg_ptr += BK * sizeof(float);
    I_ldg_ptr += BK * N * sizeof(float);
    W_sts_addr ^= 0x2000;
    I_sts_addr ^= 0x1000;

    lds128(W_reg[0][0], W_reg[0][1], W_reg[0][2], W_reg[0][3], W_lds_addr);
    lds128(W_reg[0][4], W_reg[0][5], W_reg[0][6], W_reg[0][7], W_lds_addr + 4 * 4 * sizeof(float));
    lds128(I_reg[0][0], I_reg[0][1], I_reg[0][2], I_reg[0][3], I_lds_addr);
    lds128(I_reg[0][4], I_reg[0][5], I_reg[0][6], I_reg[0][7], I_lds_addr + 4 * 8 * sizeof(float));

    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < (K / BK - 1); phase++)
    {
        // ITERATIONS : BK times
        #pragma unroll
        for (int iter = 0; iter < BK ;iter++)
        {
            // next phase: ldreg->share
            if(iter == BK - 1)
            {
                sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
                }
                __syncthreads();
                W_ldg_ptr += BK * sizeof(float);
                I_ldg_ptr += BK * N * sizeof(float);
                W_lds_addr ^= 0x2000;
                I_lds_addr ^= 0x1000;
                W_sts_addr ^= 0x2000;
                I_sts_addr ^= 0x1000;
            }
            // next iter: share->registers
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
            // next phase:global->ldreg
            if (iter == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
                }
            }
            // calculate
            {
            int cI = iter % 2;
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
            }
        }
    }
    // LAST PHASE
    #pragma unroll
    for (int iter = 0 ; iter < BK ; iter++)
    {
        // next iter: share->registers
        if (iter < (BK -1))
        {
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
        }
        // calculate
        {
            int cI = iter % 2;
#pragma unroll
            for (int i = 0; i < Tsize; ++i)
            {
#pragma unroll
                for (int j = 0; j < Tsize; ++j)
                {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
        }
    }

     int O_grow = by * BM + wy * 32 + twy * 4;
    // int O_gcol = bx * BN + wx * 64 + twx * 4;
    //int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float mean[8] = {0};
    float var[8] = {0};
    float bnW[8] = {0};
    float bnB[8] = {0};
    float cvB[8] = {0};
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int sIdx = i * 4 + j;
            int gIdx = O_grow + i*16+j;
            cvB[sIdx] = convBias[gIdx];
            mean[sIdx] = bnRM[gIdx];
            var[sIdx] = bnRV[gIdx];
            bnW[sIdx] = bnWeights[gIdx];
            bnB[sIdx] = bnBias[gIdx];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            float res2 = O_reg[i][j+4];
            float res3 = O_reg[i+4][j];
            float res4 = O_reg[i+4][j+4];
            res1 += cvB[i];
            res2 += cvB[i];
            res3 += cvB[i+4];
            res4 += cvB[i+4];
            res1 = __fdividef((res1 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res1 = max(0.0f, res1);
            res2 = __fdividef((res2 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res2 = max(0.0f, res2);
            res3 = __fdividef((res3 - mean[i+4]),sqrt(var[i+4] + esp))* bnW[i+4] + bnB[i+4];
            res3 = max(0.0f, res3);
            res4 = __fdividef((res4 - mean[i+4]),sqrt(var[i+4] + esp)) * bnW[i+4] + bnB[i+4];
            res4 = max(0.0f, res4);
            O_reg[i][j] = res1;
            O_reg[i][j+4] = res2;
            O_reg[i+4][j] = res3;
            O_reg[i+4][j+4] = res4;
        }
    }
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warpIdx * 2048) +
                                       twy * 4 * 8 + twx);
    const float *C_lds_ptr = (float *)(smem + warpIdx * 2048) + twIdx;

    uint32_t m_idx = blockIdx.y * 128 + warpIdx / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warpIdx % 2 * 64 + twIdx;

    float *C_stg_ptr = output + m_idx * N + n_idx+bO;

    
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(O_reg, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * N + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          M,
                          N,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
}
__global__ void CBR_1024x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 128;
    int BN = 128;
    // param-set : fix
    int BK = 8;
    int Tsize = 8; //thread 8*8 = 2*2 * 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *W_shared = reinterpret_cast<float *>(smem);
    float *I_shared = reinterpret_cast<float *>(smem + 16 * 1024);
    float W_ldg_reg[4];
    float I_ldg_reg[4];

    float W_reg[2][8]={0};
    float I_reg[2][8]={0};
    float O_reg[8][8] = {0};

    int W_grow = by * BM + tx / BK * 4; // 每BK个threads:连续4行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复四次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    const char *W_ldg_ptr = (const char *)(convWeights+W_LoadG);
    const char *I_ldg_ptr = (const char *)(input + I_LoadG);

    int W_srow = tx % BK; // 转置
    int W_scol = tx / BK * 4;
    int I_srow = tx / 32;
    int I_scol = tx % 32;
    int W_StoreS = INDEX(W_srow, W_scol, BM + 4);
    int I_StoreS = INDEX(I_srow, I_scol, BN);
    uint32_t W_sts_addr = smem_u32addr(W_shared + W_StoreS);
    uint32_t I_sts_addr = smem_u32addr(I_shared + I_StoreS);

    int W_LoadS = INDEX(0, (wy * 32 + twy * 4), BM + 4);
    int I_LoadS = INDEX(0, (wx * 64 + twx * 4), BN);
    uint32_t W_lds_addr = smem_u32addr(W_shared + W_LoadS);
    uint32_t I_lds_addr = smem_u32addr(I_shared + I_LoadS);

    //1st_tile:
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
    }
    sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    W_ldg_ptr += BK * sizeof(float);
    I_ldg_ptr += BK * N * sizeof(float);
    W_sts_addr ^= 0x2000;
    I_sts_addr ^= 0x1000;

    lds128(W_reg[0][0], W_reg[0][1], W_reg[0][2], W_reg[0][3], W_lds_addr);
    lds128(W_reg[0][4], W_reg[0][5], W_reg[0][6], W_reg[0][7], W_lds_addr + 4 * 4 * sizeof(float));
    lds128(I_reg[0][0], I_reg[0][1], I_reg[0][2], I_reg[0][3], I_lds_addr);
    lds128(I_reg[0][4], I_reg[0][5], I_reg[0][6], I_reg[0][7], I_lds_addr + 4 * 8 * sizeof(float));

    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < (K / BK - 1); phase++)
    {
        // ITERATIONS : BK times
        #pragma unroll
        for (int iter = 0; iter < BK ;iter++)
        {
            // next phase: ldreg->share
            if(iter == BK - 1)
            {
                sts128(W_ldg_reg[0], W_ldg_reg[1], W_ldg_reg[2], W_ldg_reg[3],W_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    sts32(I_ldg_reg[i], I_sts_addr + i * 32 * sizeof(float));
                }
                __syncthreads();
                W_ldg_ptr += BK * sizeof(float);
                I_ldg_ptr += BK * N * sizeof(float);
                W_lds_addr ^= 0x2000;
                I_lds_addr ^= 0x1000;
                W_sts_addr ^= 0x2000;
                I_sts_addr ^= 0x1000;
            }
            // next iter: share->registers
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
            // next phase:global->ldreg
            if (iter == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(W_ldg_reg[i],W_ldg_ptr + i * K * sizeof(float),true);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ldg32_nc_0(I_ldg_reg[i],I_ldg_ptr + i * 32 * sizeof(float),true);
                }
            }
            // calculate
            {
            int cI = iter % 2;
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
            }
        }
    }
    // LAST PHASE
    #pragma unroll
    for (int iter = 0 ; iter < BK ; iter++)
    {
        // next iter: share->registers
        if (iter < (BK -1))
        {
            int nI = (iter + 1) % 2;
            int nrowS = (iter + 1) % BK;
            int offW = nrowS * (BM + 4);
            int offI = nrowS * BN;
            lds128(W_reg[nI][0], W_reg[nI][1], W_reg[nI][2], W_reg[nI][3], W_lds_addr + offW * sizeof(float));
            lds128(W_reg[nI][4], W_reg[nI][5], W_reg[nI][6], W_reg[nI][7], W_lds_addr + (offW + 16) * sizeof(float));
            lds128(I_reg[nI][0], I_reg[nI][1], I_reg[nI][2], I_reg[nI][3], I_lds_addr + offI * sizeof(float));
            lds128(I_reg[nI][4], I_reg[nI][5], I_reg[nI][6], I_reg[nI][7], I_lds_addr + (offI + 32) * sizeof(float));
        }
        // calculate
        {
            int cI = iter % 2;
#pragma unroll
            for (int i = 0; i < Tsize; ++i)
            {
#pragma unroll
                for (int j = 0; j < Tsize; ++j)
                {
                    O_reg[i][j] += W_reg[cI][i] * I_reg[cI][j];
                }
            }
        }
    }

     int O_grow = by * BM + wy * 32 + twy * 4;
    // int O_gcol = bx * BN + wx * 64 + twx * 4;
    //int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float mean[8] = {0};
    float var[8] = {0};
    float bnW[8] = {0};
    float bnB[8] = {0};
    float cvB[8] = {0};
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int sIdx = i * 4 + j;
            int gIdx = O_grow + i*16+j;
            cvB[sIdx] = convBias[gIdx];
            mean[sIdx] = bnRM[gIdx];
            var[sIdx] = bnRV[gIdx];
            bnW[sIdx] = bnWeights[gIdx];
            bnB[sIdx] = bnBias[gIdx];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            float res2 = O_reg[i][j+4];
            float res3 = O_reg[i+4][j];
            float res4 = O_reg[i+4][j+4];
            res1 += cvB[i];
            res2 += cvB[i];
            res3 += cvB[i+4];
            res4 += cvB[i+4];
            
            res1 = __fdividef((res1 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res1 = max(0.0f, res1);
            res2 = __fdividef((res2 - mean[i]),sqrt(var[i] + esp)) * bnW[i] + bnB[i];
            res2 = max(0.0f, res2);
            res3 = __fdividef((res3 - mean[i+4]),sqrt(var[i+4] + esp))* bnW[i+4] + bnB[i+4];
            res3 = max(0.0f, res3);
            res4 = __fdividef((res4 - mean[i+4]),sqrt(var[i+4] + esp)) * bnW[i+4] + bnB[i+4];
            res4 = max(0.0f, res4);
            O_reg[i][j] =  res1;
            O_reg[i][j+4] = res2;
            O_reg[i+4][j] = res3;
            O_reg[i+4][j+4] = res4;
        }
    }
    #pragma unroll
    for (int i = 0;i <8 ;i++)
    {
        float maxT = O_reg[i][0];
        for (int j =1 ;j<8;j++)
        {
            if (O_reg[i][j]>maxT)
            {
                maxT = O_reg[i][j];
            }
        }
    O_reg[i][0]= maxT;
    }
    
    //warp内最大值规约 th 0 1 16 17 ，各8行
    #pragma unroll
    for (int offset = 8; offset > 1; offset >>= 1) {
        for (int i = 0;i < 8 ;i++)
        {
            O_reg[i][0] = max(O_reg[i][0], __shfl_down_sync(0xFFFFFFFF, O_reg[i][0], offset));
        }
    }
    if (twx == 0)
    {
        int O_gcol = bx * 2 + wx ;
        #pragma unroll
        for (int i = 0;i<2;i++)
        {
            for (int j =0 ;j <4;j++)
            {
                int O_grow = by * BM + wy * 32 + twy * 4 + i * 16 + j;
                int O_StoreG = INDEX(O_grow, O_gcol, N/64) + bO/64;
                output[O_StoreG] = O_reg[i*4+j][0];
            }
        }
        
    }
}
__global__ void CBRWRAP_Kernel_ic3(int TILEX,int TILEY,int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
float* convWeights, float* convBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int np = tx + bx * blockDim.x;
    int oc = ty + by * blockDim.y;
    int b = blockIdx.z;
    //printf("oc %d, batch %d, index %d\n",oc,b, index);
    if(oc >= outChannels || np >= numPoints)
        return ;
    
    float mean = bnRM[oc];
    float var = bnRV[oc];
    float bnW = bnWeights[oc];
    float bnB = bnBias[oc];
    float res = convBias[oc];

    for (int ic = 0; ic < inChannels; ic++)
    {
        int ii = b * inChannels * numPoints + ic * numPoints + np;
        int ww = oc * inChannels + ic;
        res += input[ii] * convWeights[ww];
    }
    res = __fdividef((res - mean),sqrt(var + esp)) * bnW + bnB;
    res = res > 0 ? res : 0;
    output[b * numPoints * outChannels + oc * numPoints + np] = res;
}
void CBRWRAP_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, 
float* cudaConvWeights, float* cudaConvBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    const int BLK_X = 32;
    const int BLK_Y = 32;
    dim3 blockDim(BLK_X,BLK_Y);
    dim3 grid32(DIV_UP(numPoints, 32),DIV_UP(outChannels, 32),batchSize);//X:宽度 Y：高度
    dim3 grid128(DIV_UP(numPoints, 128),DIV_UP(outChannels, 128),batchSize);
    dim3 grid64(DIV_UP(numPoints, 64),DIV_UP(outChannels, 64),batchSize);
    if (inChannels == 3)
    {
        CBRWRAP_Kernel_ic3<<<grid32, blockDim>>>(BLK_X, BLK_Y, outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else if (outChannels == 1024 && inChannels == 128  && USECONVMAX == 1)
    {
        CBR_1024x128N_kernel<<<grid128, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else if (outChannels == 128 && inChannels == 64)
    {
        CBR_128x128N_kernel<<<grid128, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else
    {
        CBR_64x128N_kernel<<<grid64, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
}
void GPU_CBR(int batchSize, int numPoints, int inics, int OC,wbBnP& wbBnP, float* input, float* reluOutput)
{
    CBRWRAP_GPU(batchSize,numPoints,inics,OC,1,input,wbBnP.weight,wbBnP.bias,
    wbBnP.bn_weight,wbBnP.bn_bias,wbBnP.bn_mean,wbBnP.bn_var,reluOutput);
}
void GPU_CBR_3 (int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,CB3P &cb3p, float* input, float* output,float* relu1_output,float* relu2_output) {
    //std::cout << "----START CBR_3" << std::endl;
    GPU_CBR(batchSize, numPoints, inics, OC1, cb3p.cb1, input, relu1_output);
    GPU_CBR(batchSize, numPoints, OC1, OC2, cb3p.cb2, relu1_output, relu2_output);
    GPU_CBR(batchSize, numPoints, OC2, OC3, cb3p.cb3, relu2_output, output);
}


// ARCH FBR
__global__ void FBRWRAP_Kernel_gemv(int M,int batchSize,int N,float* input, 
float* fcWeights, float* fcBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{
    // Block index
    int bx = blockIdx.x;
    int batch = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;//0~31
    int ty = threadIdx.y;//0~3

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_row = 4 * bx + ty;

    if(current_row < M){

        int oc = current_row;
        float mean = bnRM[oc];
        float var = bnRV[oc];
        float bnW = bnWeights[oc];
        float bnB = bnBias[oc];
        float res = 0.0f;

        int kIteration = (N/warp_size)/4;
        if(kIteration==0) kIteration=1;
        // fcWeights = &fcWeights[current_row*N];
        // input= &input[batch*N];
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col_vec = (i*warp_size + laneId);
            float4 current_val= reinterpret_cast<float4 *>(fcWeights)[current_col_vec+current_row*N/4];
            float4 current_x = reinterpret_cast<float4 *>(input)[current_col_vec+batch*N/4];
            res += current_val.x*current_x.x;
            res += current_val.y*current_x.y;
            res += current_val.z*current_x.z;
            res += current_val.w*current_x.w;
        }

        res += __shfl_down_sync(0xffffffff, res, 16); // 0-16, 1-17, 2-18, etc.
        res += __shfl_down_sync(0xffffffff, res, 8);// 0-8, 1-9, 2-10, etc.
        res += __shfl_down_sync(0xffffffff, res, 4);// 0-4, 1-5, 2-6, etc.
        res += __shfl_down_sync(0xffffffff, res, 2);// 0-2, 1-3, 4-6, 5-7, etc.
        res += __shfl_down_sync(0xffffffff, res, 1);// 0-1, 2-3, 4-5, etc.

        
        res+=fcBias[oc];
        res = __fdividef((res - mean),sqrt(var + esp)) * bnW + bnB;
        res = res > 0 ? res : 0;
        int index = oc + batch * M;
        if(laneId==0) output[index] = res;
    }
}
__global__ void FBRWRAP_Kernel(int TILEX,int TILEY,int outFeatures,int batchSize,int inFeatures,float* input, 
float* fcWeights, float* fcBias, 
float* bnWeights,float* bnBias,float* bnRM,float* bnRV,float* output,float esp = 1e-5 )
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int oc = tx + bx * blockDim.x;
    int curB = ty + by * blockDim.y;

    if (oc < outFeatures && curB < batchSize)
    {
        float mean = bnRM[oc];
        float var = bnRV[oc];
        float bnW = bnWeights[oc];
        float bnB = bnBias[oc];
        float res = fcBias[oc];
        for (int ic = 0; ic < inFeatures; ic++)
        {
            res +=
                input[curB * inFeatures + ic] *
                fcWeights[oc * inFeatures + ic];
        }
        res = __fdividef((res - mean),sqrt(var + esp)) * bnW + bnB;
        res = res > 0 ? res : 0;
        int index = oc + curB * outFeatures;
        output[index] = res;
    }
}
void FBRWRAP_GPU(int batchSize,int inFeatures,int outFeatures,float* input, 
float* cudaFcWeights, float* cudaFcBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    //std::cout << "------------LAYER:FBRWRAP" << std::endl;
    // printf("inchannel %d,numPoints %d\n",inChannels,numPoints);
    // if (inFeatures > 512)
    // {
    // const int BLK_X = 32;
    // const int BLK_Y = 32;
    // dim3 blockDim(BLK_X,BLK_Y);
    // dim3 gridDim((outFeatures + BLK_X - 1) / BLK_X,(batchSize + BLK_Y - 1) / BLK_Y);//X:宽度 Y：高度
    // FBRWRAP_Kernel<<<gridDim,blockDim>>>(BLK_X,BLK_Y,outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,
    // output);
    // }
    // else
    {
dim3 blockDim(32,4);
dim3 gridDim((outFeatures + 4 - 1) / 4,batchSize);//X:宽度 Y：高度
FBRWRAP_Kernel_gemv<<<gridDim,blockDim>>>(outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,
    output);
    }

    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());
    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}
void GPU_FBR(int batchSize, int inFeatures, 
int outFeatures,wbBnP& fbp, float* input, float* reluOutput)
{
    FBRWRAP_GPU(batchSize,inFeatures,outFeatures,input,
    fbp.weight,fbp.bias,
    fbp.bn_weight,fbp.bn_bias,
    fbp.bn_mean,fbp.bn_var,reluOutput);
}
void GPU_FBR_2_F(int OC1,int OC2,int OC3,int batchSize,int inics,FB2FP &fb2f, float* input, float* output,float* relu1_output,float* relu2_output,int param_offset=3)
{
    //std::cout << "----START FBR_2_F" << std::endl;
    GPU_FBR(batchSize,inics,OC1,fb2f.fb1,input,relu1_output);
    GPU_FBR(batchSize,OC1,OC2,fb2f.fb2,relu1_output,relu2_output);
    Linear_GPU(batchSize,OC2, OC3,fb2f.f3.weight, fb2f.f3.bias, relu2_output, output);
}


//from darknet:TODO:
__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}
__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}
__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}
__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}
__global__ void normalize_kernel(int N, float *x, float* output, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    output[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}
void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}
void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}
void normalize_gpu(float *x, float* output, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, output, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}
__global__ void madd_relu_kernel(bool relu,float* input,float *output,float* weights, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;
    if(offset < size) 
    {
        float res;
        int iIdx = (batch*n+filter)*size + offset;
        res = input[iIdx]  * weights[filter] + biases[filter];
        output[iIdx] = relu ? (res> 0 ? res : 0) : res;
    }
}
void madd_relu(bool relu,float* input,float *output,float* weights, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    madd_relu_kernel<<<dimGrid, dimBlock>>>(relu,input,output,weights, biases, n, size);
    check_error(cudaPeekAtLastError());
}

//TRAIN
__global__ void BR_Kernel(bool relu, int numPoints,float* weight,float* bias,float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx + bx * blockDim.x;
    int index = idx;

    if (idx < blockDim.x * gridDim.x)
    {
        float mean = running_mean[tx];
        float var = running_var[tx];
        for (int n = 0; n < numPoints; n++)
        {
            float res;
            int iIdx = index * numPoints + n;
            res = (input[iIdx] - mean) / sqrt(var + esp) * weight[tx] + bias[tx];
            output[iIdx] = relu ? (res> 0 ? res : 0) : res;
        }
    }
}
__global__ void CONV_64x128N_kernel(int M,int batchSize,int N,int K,float* input, 
float* convWeights, float* convBias,float* output,float esp = 1e-5 )
{   
    // param-set : variable
    int BM = 64;
    int BN = 64;
    // param-set : fix
    int BK = 8;
    int Tsize = 4; //thread 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ float W_shared[512];//1024*4B = 4KB
    __shared__ float I_shared[512];//1024*4B = 4KB
    float W_reg[4]={0};
    float I_reg[4]={0};
    float O_reg[4][4] = {0};

    int W_grow = by * BM + tx / BK * 2; // 每BK个threads:连续2行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复2次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K);
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < K / BK; phase++)
    {
        //【global -> share】
        int W_srow = tx % BK ; //转置
        int W_scol = tx / BK * 2 ; 
        int I_srow = tx / 32; 
        int I_scol = tx % 32; 
        int W_StoreS = INDEX(W_srow,W_scol,BM);
        int I_StoreS = INDEX(I_srow,I_scol,BN);
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            W_shared[W_StoreS+ldg]=convWeights[W_LoadG+ldg*K];
        }
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            I_shared[I_StoreS+ldg*32]=input[I_LoadG+ldg*32];
        }
        __syncthreads();
        W_LoadG += BK;
        I_LoadG += BK * N;
        // ITERATIONS : BK times
        for (int iter = 0; iter< BK ;iter++)
        {
            //【share -> registers】
            int W_LoadS = INDEX(iter, (wy * 16 + twy * 4), BM);
            int I_LoadS = INDEX(iter, (wx * 32 + twx * 4), BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                W_reg[i] = W_shared[W_LoadS+i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                I_reg[i] = I_shared[I_LoadS+i];
            }
            // calculate
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[i] * I_reg[j];
                }
            }
        }
    }
    int O_grow = by * BM + wy * 16 + twy * 4;
    int O_gcol = bx * BN + wx * 32 + twx * 4;
    int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //convBias and batchnorm and relu
    float cvB[4] = {0};
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int sIdx = j;
        int gIdx = O_grow + j;
        cvB[sIdx] = convBias[gIdx];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float res1 = O_reg[i][j];
            res1 += cvB[i];
            O_reg[i][j] = res1;
        }
    }
    //store to C
    #pragma unroll
    for (int i = 0; i<4;i++)
    {
        for (int j = 0 ; j<4 ;j++)
        {
            output[O_StoreG+ i*N+j]=O_reg[i][j];
        }
    }
}
__global__ void CONV_Kernel_ic3(int TILEX,int TILEY,int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
float* convWeights, float* convBias,float* output,float esp = 1e-5 )
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int np = tx + bx * blockDim.x;
    int oc = ty + by * blockDim.y;
    int b = blockIdx.z;
    if(oc >= outChannels || np >= numPoints)
        return ;
    float res = convBias[oc];
    for (int ic = 0; ic < inChannels; ic++)
    {
        int ii = b * inChannels * numPoints + ic * numPoints + np;
        int ww = oc * inChannels + ic;
        res += input[ii] * convWeights[ww];
    }
    output[b * numPoints * outChannels + oc * numPoints + np] = res;
}

void BR_train(bool relu, int batchSize,int numPoints,int outChannels,
float* weight,float* bias,float* mean,float* var,float* norm,
float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    fast_mean_gpu(input,batchSize,outChannels,numPoints,mean);
    fast_variance_gpu(input,mean,batchSize,outChannels,numPoints,var);
    scal_gpu(outChannels, .99,running_mean,1);
    axpy_gpu(outChannels, .01, mean, 1, running_mean, 1);
    scal_gpu(outChannels, .99, running_var, 1);
    axpy_gpu(outChannels, .01, var, 1, running_var, 1);
    normalize_gpu(input,norm,running_mean,running_var,batchSize,outChannels,numPoints);
    madd_relu(relu,norm,output,weight,bias,batchSize,outChannels,numPoints);
}

void CBRWRAP_GPU_train(bool relu,int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, 
float* cudaConvWeights, float* cudaConvBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float* convOutput,bn_layer& bn,float esp = 1e-5
){
    const int BLK_X = 32;
    const int BLK_Y = 32;
    dim3 blockDim(BLK_X,BLK_Y);
    dim3 grid32(DIV_UP(numPoints, 32),DIV_UP(outChannels, 32),batchSize);//X:宽度 Y：高度
    dim3 grid128(DIV_UP(numPoints, 128),DIV_UP(outChannels, 128),batchSize);
    dim3 grid64(DIV_UP(numPoints, 64),DIV_UP(outChannels, 64),batchSize);
    if (inChannels == 3)
    {
        CONV_Kernel_ic3<<<grid32, blockDim>>>(BLK_X, BLK_Y, outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,convOutput);
    }
    else
    {
        CONV_64x128N_kernel<<<grid64, 256>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias, convOutput);
    }
    //define batchnorm and relu forward
    //BR_Kernel<<<batchSize, outChannels>>>(relu,numPoints,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,convOutput,output);
    //normalize_gpu(convOutput,output,cudaBnRM,cudaBnRV,batchSize,outChannels,numPoints);
    //madd_relu(relu,output,output,cudaBnWeights,cudaBnBias,batchSize,outChannels,numPoints);
    
    BR_train(relu,batchSize,numPoints,outChannels,cudaBnWeights,cudaBnBias,bn.mean,bn.var,bn.norm,cudaBnRM,cudaBnRV,convOutput,output);
}
void GPU_CBR_train(bool relu,int batchSize, int numPoints, int inics, int OC,wbBnP& wbBnP, float* input, float* reluOutput, float* convOutput,bn_layer& bn)
{
    CBRWRAP_GPU_train(relu,batchSize,numPoints,inics,OC,1,input,wbBnP.weight,wbBnP.bias,
    wbBnP.bn_weight,wbBnP.bn_bias,wbBnP.bn_mean,wbBnP.bn_var,reluOutput,convOutput,bn);
}
void GPU_CBR_3_train (bool relu,int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,CB3P &cb3p, 
float* input, float* output,float* relu1_output,float* relu2_output,
float* conv1_output,float* conv2_output,float* conv3_output,
bn_layer& bn1,bn_layer& bn2,bn_layer& bn3) {
    std::cout << "----START CBR_3 TRAIN" << std::endl;
    GPU_CBR_train(relu,batchSize, numPoints, inics, OC1, cb3p.cb1, input, relu1_output,conv1_output,bn1);
    GPU_CBR_train(relu,batchSize, numPoints, OC1, OC2, cb3p.cb2, relu1_output, relu2_output,conv2_output,bn2);
    GPU_CBR_train(relu,batchSize, numPoints, OC2, OC3, cb3p.cb3, relu2_output, output,conv3_output,bn3);
}

__global__ void FC_Kernel_gemv(int M,int batchSize,int N,float* input, 
float* fcWeights, float* fcBias,float* output,float esp = 1e-5 )
{
    // Block index
    int bx = blockIdx.x;
    int batch = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;//0~31
    int ty = threadIdx.y;//0~3

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_row = 4 * bx + ty;

    if(current_row < M){
        int oc = current_row;
        float res = 0.0f;
        int kIteration = (N/warp_size)/4;
        if(kIteration==0) kIteration=1;
        // fcWeights = &fcWeights[current_row*N];
        // input= &input[batch*N];
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col_vec = (i*warp_size + laneId);
            float4 current_val= reinterpret_cast<float4 *>(fcWeights)[current_col_vec+current_row*N/4];
            float4 current_x = reinterpret_cast<float4 *>(input)[current_col_vec+batch*N/4];
            res += current_val.x*current_x.x;
            res += current_val.y*current_x.y;
            res += current_val.z*current_x.z;
            res += current_val.w*current_x.w;
        }
        res += __shfl_down_sync(0xffffffff, res, 16); // 0-16, 1-17, 2-18, etc.
        res += __shfl_down_sync(0xffffffff, res, 8);// 0-8, 1-9, 2-10, etc.
        res += __shfl_down_sync(0xffffffff, res, 4);// 0-4, 1-5, 2-6, etc.
        res += __shfl_down_sync(0xffffffff, res, 2);// 0-2, 1-3, 4-6, 5-7, etc.
        res += __shfl_down_sync(0xffffffff, res, 1);// 0-1, 2-3, 4-5, etc.
        res+=fcBias[oc];
        int index = oc + batch * M;
        if(laneId==0) output[index] = res;
    }
}
void FBRWRAP_GPU_train(int batchSize,int inFeatures,int outFeatures,float* input, 
float* cudaFcWeights, float* cudaFcBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float* fcOutput,float esp = 1e-5
){
        dim3 blockDim(32,4);
        dim3 gridDim((outFeatures + 4 - 1) / 4,batchSize);//X:宽度 Y：高度
        FC_Kernel_gemv<<<gridDim,blockDim>>>(outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,fcOutput);
        //BR_Kernel<<<batchSize, outFeatures>>>(true,1,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,fcOutput,output);
        //TODO:这里用上面的会更快
        normalize_gpu(fcOutput,output,cudaBnRM,cudaBnRV,batchSize,outFeatures,1);
        madd_relu(true,output,output,cudaBnWeights,cudaBnBias,batchSize,outFeatures,1);
}
void GPU_FBR_train(int batchSize, int inFeatures, int outFeatures,wbBnP& fbp, float* input, float* reluOutput, float* fcOutput)
{
    FBRWRAP_GPU_train(batchSize,inFeatures,outFeatures,input,
    fbp.weight,fbp.bias,
    fbp.bn_weight,fbp.bn_bias,
    fbp.bn_mean,fbp.bn_var,reluOutput,fcOutput);
}
void GPU_FBR_2_F_train(int OC1,int OC2,int OC3,int batchSize,int inics,FB2FP &fb2f, float* input, float* output,float* relu1_output,float* relu2_output,float* fc1_output,float* fc2_output,int param_offset=3)
{
    std::cout << "----START FBR_2_F TRAIN" << std::endl;
    GPU_FBR_train(batchSize,inics,OC1,fb2f.fb1,input,relu1_output,fc1_output);
    GPU_FBR_train(batchSize,OC1,OC2,fb2f.fb2,relu1_output,relu2_output,fc2_output);
    Linear_GPU(batchSize,OC2, OC3,fb2f.f3.weight, fb2f.f3.bias, relu2_output, output);
}


__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    //bias_updates[index] += sum;//TODO:
    bias_updates[index] = sum;
}
__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[DARKNETBLK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += DARKNETBLK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < DARKNETBLK; ++i) bias_updates[filter] =part[i];//+= part[i];//TODO:
    }
}
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    if(size == 1){
        backward_bias_conn_kernel<<<cuda_gridsize(n), DARKNETBLK>>>(bias_updates, delta, batch, n);
    }else{
        backward_bias_kernel<<<n, DARKNETBLK>>>(bias_updates, delta, batch, n, size);
    }
}
void BR_bp(bool relu,int numFeatures, int batchSize, int numPoints,float* weight,float* bias,float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    
}
void FC_bp(int batchSize, int inFeatures, int outFeatures,float* input,float* weight,float * delta_from, float * delta_gen, float* weight_up, float* bias_up) {
    //delta gen: batchsize,outf * outf,inf
    int M = batchSize;
    int N = inFeatures;
    int K = outFeatures;
    gemm_gpu(false, false, M, N, K, 1.0, delta_from,K , weight,N, 0.0, delta_gen,N);
    //WEIGHT UP: outf,batchsize * batchsize,inf
    M = outFeatures;
    N = inFeatures;
    K = batchSize;
    gemm_gpu(true, false, M, N, K, 1.0, delta_from,M,input,N,0.0,weight_up,N);
    //BIAS UP: outf,batchsize
    backward_bias_gpu(bias_up,delta_from,batchSize,outFeatures,1);
}
void FBR2F_bp(){
    std::cout << "----START FBR2F_bp" << std::endl;
}
//Initial Weight and bias, rn and rv with one and zero
void Train_GPU (int inChannels,int batchSize,int numPoints,
            int* correct_table,int* label,float* input,
            float* device_output,float* device_delta,
            const std::vector<float>& C1={},
            const std::vector<float>& C2={},
            const std::vector<float>& C3={},
            const std::vector<float>& C4={},
            bool compare=false) {
    // std::cout << "**********************START TRAINING************************" << std::endl;
    int ch = 1024;
    int ch_half = 512;
    int ch_quarter = 256;

    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = ch;
    int FC_OC1 = ch_half;
    int FC_OC2 = ch_quarter;
    int FC_OC3 = 9;
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = ch;
    int fstn_FC_OC1 = ch_half;
    int fstn_FC_OC2 = ch_quarter;
    int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    int encoderOC2 = 128;
    int encoderOC3 = ch;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    int transSize = batchSize * inChannels * inChannels;
    int transFeatSize = batchSize * fstn_inChannel * fstn_inChannel;
    //stn3d
    int stn_1 = bn*inChannels;
    int stn_2_conv = bn*OC1;
    int stn_3_conv = bn*OC2;
    int stn_4_conv = bn*OC3;
    int stn_2 = bn*OC1;
    int stn_3 = bn*OC2;
    int stn_4 = bn*OC3;
    if (USECONVMAX == 1) { stn_4 = stn_4 / 64 ;}
    int stn_5 = batchSize*OC3;
    int stn_6_fc = batchSize*FC_OC1;
    int stn_7_fc = batchSize*FC_OC2;
    int stn_6 = batchSize*FC_OC1;
    int stn_7 = batchSize*FC_OC2;
    int stn_8 = transSize;
    //part2
    int part2_1= batchSize*numPoints*encoderIC1 ;
    int part2_2= batchSize*encoderIC1*numPoints ;
    int part2_3= batchSize*fstn_inChannel*numPoints ;
    int part2_4= batchSize*fstn_inChannel*numPoints ;
    //stnkd
    int fstn_1_conv= bn * fstn_OC1 ;
    int fstn_2_conv= bn * fstn_OC2 ;
    int fstn_3_conv= bn * fstn_OC3 ;
    int fstn_1= bn * fstn_OC1 ;
    int fstn_2= bn * fstn_OC2 ;
    int fstn_3= bn * fstn_OC3 ;
    if (USECONVMAX == 1) { fstn_3 = fstn_3 / 64 ;}
    int fstn_4= batchSize * fstn_OC3 ;
    int fstn_5_fc= batchSize * fstn_FC_OC1 ;
    int fstn_6_fc= batchSize * fstn_FC_OC2 ;
    int fstn_5= batchSize * fstn_FC_OC1 ;
    int fstn_6= batchSize * fstn_FC_OC2 ;
    int fstn_7= transFeatSize ;
    //part4
    int part4_1= bn * fstn_inChannel ;
    int part4_2= batchSize*numPoints*fstn_inChannel ;
    int part4_3= batchSize*fstn_inChannel*numPoints ;
    int part4_4= batchSize*encoderOC2*numPoints ;int part4_4_conv= batchSize*encoderOC2*numPoints ;
    int part4_5= bnEOC3 ;int part4_5_conv= bnEOC3 ;
    if (USECONVMAX == 1) { part4_5 = part4_5 / 64 ;}
    int part4_6= batchSize * encoderOC3 ;
    //classify
    int cla_1= batchSize * ch_half ;int cla_1_fc= batchSize * ch_half ;
    int cla_2= batchSize * ch_quarter ;int cla_2_fc= batchSize * ch_quarter ;
    int cla_3= batchSize * 10;
    int cla_4= batchSize * 10;

    TNET net;
    long long offset = 0;
    //stn3d 
    net.input_trans = device_output+offset;offset += stn_1;

    net.conv1_output_stn_cbr = device_output+offset;offset += stn_2_conv;
    offset = alloc_bn(net.bn1_stn_cbr,device_output,offset,OC1,bn);
    net.relu1_output_stn_cbr = device_output+offset;offset += stn_2;

    net.conv2_output_stn_cbr = device_output+offset;offset += stn_3_conv;
    offset = alloc_bn(net.bn2_stn_cbr,device_output,offset,OC2,bn);
    net.relu2_output_stn_cbr = device_output+offset;offset += stn_3;

    net.conv3_output_stn_cbr = device_output+offset;offset += stn_4_conv;
    offset = alloc_bn(net.bn3_stn_cbr,device_output,offset,OC3,bn);
    net.CBR3_output = device_output+offset;offset += stn_4;

    net.maxp_output = device_output+offset;offset += stn_5;
    net.fc1_output_stn_cbr = device_output+offset;offset += stn_6_fc;
    net.fc2_output_stn_cbr = device_output+offset;offset += stn_7_fc;
    net.relu1_output_stn_fbr2f = device_output+offset;offset += stn_6;
    net.relu2_output_stn_fbr2f = device_output+offset;offset += stn_7;
    net.stn3d_out = device_output+offset;offset += stn_8;

    //part2
    net.bmm1_res = device_output+offset;offset += part2_1;
    net.bmm1_res_trans = device_output+offset;offset += part2_2;
    net.fstn_input_conv = device_output+offset;offset += part2_3;
    offset = alloc_bn(net.fstn_input_bn,device_output,offset,fstn_inChannel,bn);
    net.fstn_input = device_output+offset;offset += part2_4;

    //stnkd
    net.conv1_output_fstn_cbr = device_output+offset;offset += fstn_1_conv;
    offset = alloc_bn(net.bn1_fstn_cbr,device_output,offset,fstn_OC1,bn);
    net.relu1_output_fstn_cbr = device_output+offset;offset += fstn_1;

    net.conv2_output_fstn_cbr = device_output+offset;offset += fstn_2_conv;
    offset = alloc_bn(net.bn2_fstn_cbr,device_output,offset,fstn_OC2,bn);
    net.relu2_output_fstn_cbr = device_output+offset;offset += fstn_2;

    net.conv3_output_fstn_cbr = device_output+offset;offset += fstn_3_conv;
    offset = alloc_bn(net.bn3_fstn_cbr,device_output,offset,fstn_OC3,bn);
    net.fstn_CBR3_output = device_output+offset;offset += fstn_3;

    net.fstn_maxp_output = device_output+offset;offset += fstn_4;
    net.fc1_output_fstn_fbr2f = device_output+offset;offset += fstn_5_fc;
    net.relu1_output_fstn_fbr2f = device_output+offset;offset += fstn_5;
    net.fc2_output_fstn_fbr2f = device_output+offset;offset += fstn_6_fc;
    net.relu2_output_fstn_fbr2f = device_output+offset;offset += fstn_6;
    net.stnkd_out = device_output+offset;offset += fstn_7;

    //part4
    net.fstn_input_trans = device_output+offset;offset += part4_1;
    net.fstn_bmm1_res = device_output+offset;offset += part4_2;
    net.fstn_bmm1_res_trans = device_output+offset;offset += part4_3;

    net.cbr2_output_conv = device_output+offset;offset += part4_4_conv;
    offset = alloc_bn(net.cbr2_output_bn,device_output,offset,encoderOC2,bn);
    net.cbr2_output = device_output+offset;offset += part4_4;

    net.feat_bn3_conv = device_output+offset;offset += part4_5_conv;
    offset = alloc_bn(net.feat_bn3_bn,device_output,offset,encoderOC3,bn);
    net.feat_bn3 = device_output+offset;offset += part4_5;
    net.encoder_output = device_output+offset;offset += part4_6;
    //classify
    net.fc1_output_part5_fbr2f = device_output+offset;offset += cla_1_fc;
    net.relu1_output_part5_fbr2f = device_output+offset;offset += cla_1;
    net.fc2_output_part5_fbr2f = device_output+offset;offset += cla_2_fc;
    net.relu2_output_part5_fbr2f = device_output+offset;offset += cla_2;
    net.softmax_input = device_output+offset;offset += cla_3;
    net.softmax_output = device_output+offset;offset+= cla_4;

    // TNET delta;
    // offset = 0;
    // //stn3d 
    // //TODO:到时候直接复制就好
    // delta.input_trans = device_delta+offset;offset += stn_1;
    // delta.conv1_output_stn_cbr = device_delta+offset;offset += stn_2_conv;
    // delta.relu1_output_stn_cbr = device_delta+offset;offset += stn_2;
    // delta.conv2_output_stn_cbr = device_delta+offset;offset += stn_3_conv;
    // delta.relu2_output_stn_cbr = device_delta+offset;offset += stn_3;
    // delta.conv3_output_stn_cbr = device_delta+offset;offset += stn_4_conv;
    // delta.CBR3_output = device_delta+offset;offset += stn_4;
    // delta.maxp_output = device_delta+offset;offset += stn_5;
    // delta.fc1_output_stn_cbr = device_delta+offset;offset += stn_6_fc;
    // delta.fc2_output_stn_cbr = device_delta+offset;offset += stn_7_fc;
    // delta.relu1_output_stn_fbr2f = device_delta+offset;offset += stn_6;
    // delta.relu2_output_stn_fbr2f = device_delta+offset;offset += stn_7;
    // delta.stn3d_out = device_delta+offset;offset += stn_8;
    // //part2
    // delta.bmm1_res = device_delta+offset;offset += part2_1;
    // delta.bmm1_res_trans = device_delta+offset;offset += part2_2;
    // delta.fstn_input_conv = device_delta+offset;offset += part2_3;
    // delta.fstn_input = device_delta+offset;offset += part2_4;
    // //stnkd
    // delta.conv1_output_fstn_cbr = device_delta+offset;offset += fstn_1_conv;
    // delta.relu1_output_fstn_cbr = device_delta+offset;offset += fstn_1;
    // delta.conv2_output_fstn_cbr = device_delta+offset;offset += fstn_2_conv;
    // delta.relu2_output_fstn_cbr = device_delta+offset;offset += fstn_2;
    // delta.conv3_output_fstn_cbr = device_delta+offset;offset += fstn_3_conv;
    // delta.fstn_CBR3_output = device_delta+offset;offset += fstn_3;
    // delta.fstn_maxp_output = device_delta+offset;offset += fstn_4;
    // delta.fc1_output_fstn_fbr2f = device_delta+offset;offset += fstn_5_fc;
    // delta.relu1_output_fstn_fbr2f = device_delta+offset;offset += fstn_5;
    // delta.fc2_output_fstn_fbr2f = device_delta+offset;offset += fstn_6_fc;
    // delta.relu2_output_fstn_fbr2f = device_delta+offset;offset += fstn_6;
    // delta.stnkd_out = device_delta+offset;offset += fstn_7;
    // //part4
    // delta.fstn_input_trans = device_delta+offset;offset += part4_1;
    // delta.fstn_bmm1_res = device_delta+offset;offset += part4_2;
    // delta.fstn_bmm1_res_trans = device_delta+offset;offset += part4_3;
    // delta.cbr2_output_conv = device_delta+offset;offset += part4_4_conv;
    // delta.cbr2_output = device_delta+offset;offset += part4_4;
    // delta.feat_bn3_conv = device_delta+offset;offset += part4_5_conv;
    // delta.feat_bn3 = device_delta+offset;offset += part4_5;
    // delta.encoder_output = device_delta+offset;offset += part4_6;
    // //classify
    // delta.fc1_output_part5_fbr2f = device_delta+offset;offset += cla_1_fc;
    // delta.relu1_output_part5_fbr2f = device_delta+offset;offset += cla_1;
    // delta.fc2_output_part5_fbr2f = device_delta+offset;offset += cla_2_fc;
    // delta.relu2_output_part5_fbr2f = device_delta+offset;offset += cla_2;
    // delta.softmax_input = device_delta+offset;offset += cla_3;
    // delta.softmax_output = device_delta+offset;offset+= cla_4;

    std::cout << "PART1:STN3d, forwaring" << std::endl;
    int maxnp = USECONVMAX? numPoints / 64 : numPoints;
    GPU_transpose(input,net.input_trans,batchSize,numPoints,inChannels);
    GPU_CBR_3_train(true,OC1,OC2,OC3, batchSize, numPoints,inChannels,dParams.stn3dp.cb3, net.input_trans, 
    net.CBR3_output,net.relu1_output_stn_cbr,net.relu2_output_stn_cbr,
    net.conv1_output_stn_cbr,net.conv2_output_stn_cbr,net.conv3_output_stn_cbr,
    net.bn1_stn_cbr,net.bn2_stn_cbr,net.bn3_stn_cbr);   // conv-bn-relu * 3
    GPU_MaxPooling(OC3, batchSize, maxnp,net.CBR3_output, net.maxp_output); // Max pooling    
    GPU_FBR_2_F_train(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3,dParams.stn3dp.fb2f,net.maxp_output,
    net.stn3d_out,net.relu1_output_stn_fbr2f,net.relu1_output_stn_fbr2f,
    net.fc1_output_stn_cbr,net.fc2_output_stn_cbr);// fc-bn-relu * 2 + fc
    matrix_add_I(net.stn3d_out,3,batchSize);
    std::cout << "PART2:TRANS->BMM->TRANS->CBR, forwarding" << std::endl;
    GPU_Bmm(input,net.stn3d_out,net.bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    GPU_transpose(net.bmm1_res,net.bmm1_res_trans,batchSize,numPoints,encoderIC1);
    //GPU_CBR(batchSize,numPoints,encoderIC1,fstn_inChannel,dParams.featp.cb1,net.bmm1_res_trans,net.fstn_input);
    GPU_CBR_train(true,batchSize,numPoints,encoderIC1,fstn_inChannel,dParams.featp.cb1,net.bmm1_res_trans,net.fstn_input,net.fstn_input_conv,net.fstn_input_bn);
    
    std::cout << "PART3:STNkd, forwarding"<< std::endl;
    GPU_CBR_3_train(true,fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,dParams.stnkdp.cb3, net.fstn_input, 
    net.fstn_CBR3_output,net.relu1_output_fstn_cbr,net.relu2_output_fstn_cbr,
    net.conv1_output_fstn_cbr,net.conv2_output_fstn_cbr,net.conv3_output_fstn_cbr,
    net.bn1_fstn_cbr,net.bn2_fstn_cbr,net.bn3_fstn_cbr);   // conv-bn-relu * 3
    GPU_MaxPooling(fstn_OC3, batchSize, maxnp,net.fstn_CBR3_output, net.fstn_maxp_output); // Max pooling
    GPU_FBR_2_F_train(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3,dParams.stnkdp.fb2f,net.fstn_maxp_output,
    net.stnkd_out,net.relu1_output_fstn_fbr2f,net.relu2_output_fstn_fbr2f,
    net.fc1_output_fstn_fbr2f,net.fc2_output_fstn_fbr2f);// fc-bn-relu * 2 + fc
    matrix_add_I(net.stnkd_out,64,batchSize);

    std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM, forwarding" << std::endl;
    GPU_transpose(net.fstn_input,net.fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    GPU_Bmm(net.fstn_input_trans,net.stnkd_out,net.fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    GPU_transpose(net.fstn_bmm1_res,net.fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    GPU_CBR_train(true,batchSize,numPoints,fstn_inChannel,encoderOC2,
    dParams.featp.cb2,net.fstn_bmm1_res_trans,net.cbr2_output,net.cbr2_output_conv,net.cbr2_output_bn);
    //------CB MAX
    std::string convStr = "feat.conv3";
    std::string bnStr = "feat.bn3";
    GPU_CBR_train(false,batchSize,numPoints,encoderOC2,encoderOC3,
    dParams.featp.cb3,net.cbr2_output, net.feat_bn3,net.feat_bn3_conv,net.feat_bn3_bn);
    GPU_MaxPooling(encoderOC3, batchSize, maxnp,net.feat_bn3, net.encoder_output); // Max pooling
    

    std::cout << "PART5:CLASSIFY, forwarding" << std::endl;
    GPU_FBR_2_F_train(512,256,10,batchSize,encoderOC3,dParams.nonep,
    net.encoder_output,net.softmax_input,
    net.relu1_output_part5_fbr2f,net.relu2_output_part5_fbr2f,
    net.fc1_output_part5_fbr2f,net.fc2_output_part5_fbr2f,0);// fc-bn-relu * 2 + fc
    LogSoftMax_GPU_train(label,net.softmax_input,
    net.softmax_output,NULL,correct_table,10,batchSize);

    
    // F->RB->F->RB->F
    // MAX->B->C -> RB->C -> TRANS-> ......
    
}

int main(int argc, char *argv[]) {
    
    // 定义模型参数
    int ic = 3;
    size_t batchSize = 32;
    int use_sample = SAMPLE;
    int npoint = NPOINT;

    // 读权重：主机
    std::string dir = argv[1]; 
    read_params(dir);

    // 读输入：主机
    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    read_h5_file(file_path, list_of_points, list_of_labels);
    int all_num = list_of_points.size();
    //all_num = 64;
    //分配内存，迁移权重到device端
    read_stndP("feat.stn.", dParams.stn3dp);
    read_stndP("feat.fstn.", dParams.stnkdp);
    read_CB3P("feat.", dParams.featp);
    read_FB2FP("", dParams.nonep, 0);

    //分配内存 for 输入：device端
    size_t total_size = 0;
    float *device_all_points;
    for (const auto &points : list_of_points)
    {
        total_size += points.size();
    }
    cudaMalloc((void **)&device_all_points, total_size * sizeof(float));

    // 迁移输入到device端
    int maxNp = 0;
    int cpy_offset = 0;
    for (size_t i = 0; i < all_num; i+=batchSize) {
        size_t curB = std::min(batchSize, all_num - i);
        size_t np = list_of_points[i].size() / ic;
        for (int j = 0; j < curB; j++) {np = std::min(np, list_of_points[i + j].size() / ic);}
        np = ALIGN_DOWN(np, GEMMBLKMAX);
        int leastnp = np;
        if(use_sample==1) np = npoint;
        int bSize = np * ic;
        int bWidth = curB * bSize;
        std::vector<float> input(bWidth);

        //使用随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, np - 1); // 随机选择点的索引
        if(use_sample==1)
        {
            for (int b = 0; b < curB; ++b) {
                int step = leastnp / np;  // 计算采样的步长
                for (int j = 0; j < np; ++j) {
                    int uniform_index = j * step;  // 按固定步长采样点索引
                    std::memcpy(&input[b * bSize + j * ic], 
                                &list_of_points[i + b][uniform_index * ic], 
                                ic * sizeof(float));  // 拷贝每个点的特征
                }
            }
            // for (int b = 0; b < curB; ++b) {
            //     for (int j = 0; j < np; ++j) {
            //         int rand_index = dis(gen); // 随机生成一个索引
            //         std::memcpy(&input[b * bSize + j * ic], 
            //                     &list_of_points[i + b][rand_index * ic], 
            //                     ic * sizeof(float));  // 拷贝每个点的特征
            //     }
            // }
        }
        else
        {
            for (int b = 0; b < curB; ++b)
            {
                std::memcpy(&input[b * bSize],
                            &list_of_points[i + b][0],
                            bSize * sizeof(float));
            }
        }
        cudaMemcpy(device_all_points + cpy_offset, input.data(), bWidth * sizeof(float), cudaMemcpyHostToDevice);
        cpy_offset += bWidth;
        if (np > maxNp) {maxNp = np;}
    }

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    int correct_num =0;
    int inf_offset = 0;
    
    // 分配输出内存：device端
    int* correct_table;
    int *device_labels;
    float *device_output;
    //float *device_delta;
    long long max_output_size = cal_tnet_size(batchSize, maxNp, ic); // batchSize > curB maxNp > np
    printf("max_output_size: %lld\n", max_output_size);
    cudaMalloc((void **)&correct_table, all_num * sizeof(int));
    cudaMalloc((void **)&device_labels, all_num * sizeof(int));
    cudaMalloc((void **)&device_output, max_output_size * sizeof(float));
    //cudaMalloc((void **)&device_delta, max_output_size * sizeof(float));
    cudaMemcpy(device_labels, list_of_labels.data(), all_num * sizeof(int), cudaMemcpyHostToDevice);

    // 开始推理
    for (size_t i = 0; i < all_num; i+=batchSize) {
        size_t curB = std::min(batchSize, all_num - i);
        size_t np = list_of_points[i].size() / ic;
        for (int j = 0; j < curB; j++) {np = std::min(np, list_of_points[i + j].size() / ic);}
        np = ALIGN_DOWN(np, GEMMBLKMAX);
        if (use_sample == 1) np = npoint;
        Train_GPU(ic, curB, np, correct_table + i, device_labels + i, 
        device_all_points + inf_offset, device_output, NULL);
        inf_offset += curB * np * ic;
        //cudaMemset(device_output, 0, cal_tnet_size(curB, np, ic) * sizeof(float));
    }

    // 计算准确率
    std::vector<int> result(all_num,0);
    cudaMemcpy(result.data(), correct_table, all_num * sizeof(int), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < all_num; i++) {
        correct_num += result[i];
    }
	float correct_rate = (float)correct_num/all_num; //(float)list_of_labels.size();

    // 释放内存
    //cudaProfilerStop();
    freeDP(dParams);//权重
    cudaFree(device_labels);//label
    cudaFree(device_all_points);//输入
    cudaFree(device_output);//输出
    //cudaFree(device_delta);//delta
    cudaFree(correct_table);//正确表

	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    //cudaProfilerStop();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << std::setprecision(4) << correct_rate;
    return 0;
}



