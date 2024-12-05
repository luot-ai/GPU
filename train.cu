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
#include <random> // 包含随机数生成相关的库
#include <ctime>  // 包含 time 函数
#include <curand_kernel.h>


#define GEMMBLKMAX 128
#define ALIGN_DOWN(x, align) ((x) / (align) * (align))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define INDEX(row, col, width) ((row) * (width) + (col))
#define NPOINT 128
#define SAMPLE 1
#define USECONVMAX 0
#define CLASSNUM 10
#define DARKNETBLK 512
#define BLOCK 512
#define EPOCH 30
#define PRETRAIN 0
#define USEMATDIFF 0
#define DROPOUT 0
// #define DEBUG
// #define BACKDEBUG
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

__global__ void initialize_weights(float *weights, int width, int fan_in, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 计算标准差 std = sqrt(2.0f / fan_in)（He 初始化）
    if (idx < width) {
        unsigned long long thread_seed = seed ^ (idx * 0x5DEECE66DLL + 0xB);
        curandState state;
        curand_init(thread_seed, idx, 0, &state);  // 为每个线程初始化随机数生成器

        // He 初始化，生成均值为0，标准差为 sqrt(2.0f / fan_in) 的正态分布
        float std = sqrtf(2.0f / fan_in);
        weights[idx] = curand_normal(&state) * std;
    }
}

void para_init_he(float *weights, int width, int fan_in) {
    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;
    unsigned long long seed = static_cast<unsigned long long>(time(0));
    // 调用 kernel 初始化权重
    initialize_weights<<<numBlocks, blockSize>>>(weights, width, fan_in, seed);
    //cudaDeviceSynchronize();
}

void para_init(float* N,int width,float init = 0.2){
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_real_distribution<float> dist(-init, init);  
    float* rand = new float[width];
    for (int i = 0; i < width; ++i) {
        rand[i] = dist(gen); 
    }
    cudaMemcpy(N, rand, width * sizeof(float), cudaMemcpyHostToDevice);
    delete[] rand;
}

void para_init_val(float* N,int width,float val)
{
    float* h_bn_var = (float*)malloc(width * sizeof(float));
    for (int i = 0; i < width; ++i) {
        h_bn_var[i] = val;  // 初始化为 1.0f
    }
    cudaMemcpy(N, h_bn_var, width * sizeof(float), cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize(); 
    free(h_bn_var);
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
    float* maxp_output_idx;

    float* fc1_output_stn_cbr;
    float* fc2_output_stn_cbr;
    bn_layer bn1_stn_fbr2f;
    bn_layer bn2_stn_fbr2f;
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
    float* fstn_maxp_output_idx;

    float* fc1_output_fstn_fbr2f;
    float* fc2_output_fstn_fbr2f;
    bn_layer bn1_fstn_fbr2f;
    bn_layer bn2_fstn_fbr2f;
    float* relu1_output_fstn_fbr2f;
    float* relu2_output_fstn_fbr2f;
    float* stnkd_out;
    float* stnkd_out_trans;
    //part4
    float* fstn_input_trans;
    float* fstn_bmm1_res;
    float* fstn_bmm1_res_trans; // B C N
    float* cbr2_output;float* cbr2_output_conv;bn_layer cbr2_output_bn;
    float* feat_bn3;float* feat_bn3_conv;bn_layer feat_bn3_bn;
    float* encoder_output;
    float* encoder_output_idx;
    //classify
    float* fc1_output_part5_fbr2f;float* relu1_output_part5_fbr2f;bn_layer bn1_part5_fbr2f;
    float* fc2_output_part5_fbr2f;float* relu2_output_part5_fbr2f;bn_layer bn2_part5_fbr2f;
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
    int stn_5_idx = batchSize*OC3;
    int stn_6_fc = batchSize*FC_OC1;
    int stn_7_fc = batchSize*FC_OC2;
    int stn_67_bn = cal_bn(batchSize,FC_OC1)+cal_bn(batchSize,FC_OC2);
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
    int fstn_4_idx= batchSize * fstn_OC3 ;
    int fstn_5_fc= batchSize * fstn_FC_OC1 ;
    int fstn_6_fc= batchSize * fstn_FC_OC2 ;
    int fstn_56_bn = cal_bn(batchSize,fstn_FC_OC1)+cal_bn(batchSize,fstn_FC_OC2);
    int fstn_5= batchSize * fstn_FC_OC1 ;
    int fstn_6= batchSize * fstn_FC_OC2 ;
    int fstn_7= transFeatSize ;
    int fstn_8= transFeatSize ;
    //part4
    int part4_1= bn * fstn_inChannel ;
    int part4_2= batchSize*numPoints*fstn_inChannel ;
    int part4_3= batchSize*fstn_inChannel*numPoints ;
    int part4_4= batchSize*encoderOC2*numPoints ;int part4_4_conv= batchSize*encoderOC2*numPoints ;int part4_4_bn = cal_bn(bn,encoderOC2);
    int part4_5= bnEOC3 ;int part4_5_conv= bnEOC3 ; int part4_5_bn = cal_bn(bn,encoderOC3);
    if (USECONVMAX == 1) { part4_5 = part4_5 / 64 ;}
    int part4_6= batchSize * encoderOC3 ;
    int part4_6_idx= batchSize * encoderOC3 ;  
    //classify
    int cla_1= batchSize * ch_half ;
    int cla_1_fc= batchSize * ch_half ;
    int cla_1_bn= cal_bn(batchSize,ch_half) ;
    int cla_2= batchSize * ch_quarter ;
    int cla_2_fc= batchSize * ch_quarter ;
    int cla_2_bn= cal_bn(batchSize,ch_quarter) ;
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
    totalSize += stn_67_bn  + fstn_56_bn + cla_1_bn + cla_2_bn;
    totalSize += part4_6_idx+ fstn_4_idx + stn_5_idx;

    totalSize += fstn_8;
    return totalSize;
}

struct fcp {
    float* weight; // Conv weight
    float* bias;   // Conv bias
};
void read_fcp(const std::string& layer, fcp& wbp,int i,bool update=false,float IC=0) {
    std::string fiStr = std::to_string(i);;
    std::string name = layer + "fc" + fiStr;  
    //std::cout << name << std::endl;
    cudaMalloc((void**)&wbp.weight, params[name + ".weight"].size() * sizeof(float));
    cudaMalloc((void**)&wbp.bias, params[name + ".bias"].size() * sizeof(float));
    if(update == true)
    {
        cudaMemset(wbp.weight,0, params[name + ".weight"].size() * sizeof(float));
        cudaMemset(wbp.bias,0, params[name + ".bias"].size() * sizeof(float));
    }
    if(update == false)
    {
        int wcnt = params[name + ".weight"].size();
        int bcnt = params[name + ".bias"].size();
        if (PRETRAIN == 1)
        {
            cudaMemcpy(wbp.weight, params[name + ".weight"].data(), wcnt*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbp.bias, params[name + ".bias"].data(), bcnt* sizeof(float), cudaMemcpyHostToDevice);
        }
        else 
        {
            para_init_he(wbp.weight,wcnt,IC);
            cudaMemset(wbp.bias, 0, bcnt * sizeof(float));
        }
    }
}
void free_fcp(fcp& wbp){
    cudaFree(wbp.bias);
    cudaFree(wbp.weight);
}
void memset_fcp(const std::string& layer, fcp& wbp,int i) {
    std::string fiStr = std::to_string(i);;
    std::string name = layer + "fc" + fiStr;  
    cudaMemset(wbp.weight,0, params[name + ".weight"].size() * sizeof(float));
    cudaMemset(wbp.bias,0, params[name + ".bias"].size() * sizeof(float));
}

struct wbBnP {
    float* weight; // Conv weight
    float* bias;   // Conv bias
    float* bn_weight; // BatchNorm weight
    float* bn_bias;   // BatchNorm bias
    float* bn_mean;   // BatchNorm running mean
    float* bn_var;    // BatchNorm running var
};
void read_wbBnP(const std::string& layer,const std::string& cf,wbBnP& wbBnP,int i,int param_offset,bool update=false,
float IC=0) {

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
    if(update == true)
    {
        //printf("memset\n");
        cudaMemset(wbBnP.weight, 0, params[name + ".weight"].size() * sizeof(float)); 
        cudaMemset(wbBnP.bias, 0, params[name + ".bias"].size() * sizeof(float));
        cudaMemset(wbBnP.bn_weight, 0,params[bnStr + ".weight"].size() * sizeof(float));
        cudaMemset(wbBnP.bn_bias, 0, params[bnStr + ".bias"].size() * sizeof(float));
    }
    if(update == false)
    {
        int wcnt = params[name + ".weight"].size();
        int bcnt = params[name + ".bias"].size();
        int bn_wcnt = params[bnStr + ".weight"].size();
        int bn_bcnt = params[bnStr + ".bias"].size();
        int bn_mcnt = params[bnStr + ".running_mean"].size();
        int bn_vcnt = params[bnStr + ".running_var"].size();
        cudaMalloc((void**)&wbBnP.bn_mean, bn_mcnt * sizeof(float));
        cudaMalloc((void**)&wbBnP.bn_var, bn_vcnt * sizeof(float));
        if(PRETRAIN == 1)
        {
            cudaMemcpy(wbBnP.weight, params[name + ".weight"].data(), wcnt * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbBnP.bias, params[name + ".bias"].data(), bcnt * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbBnP.bn_weight, params[bnStr + ".weight"].data(), bn_wcnt * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbBnP.bn_bias, params[bnStr + ".bias"].data(), bn_bcnt * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbBnP.bn_mean, params[bnStr + ".running_mean"].data(), bn_mcnt * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(wbBnP.bn_var, params[bnStr + ".running_var"].data(), bn_vcnt * sizeof(float), cudaMemcpyHostToDevice);
        }
        else 
        {
            para_init_he(wbBnP.weight,wcnt,IC);
            cudaMemset(wbBnP.bias, 0, bcnt * sizeof(float));
            para_init_val(wbBnP.bn_weight,bn_wcnt,1.0f);
            cudaMemset(wbBnP.bn_bias, 0, bn_bcnt * sizeof(float));
            cudaMemset(wbBnP.bn_mean, 0, bn_mcnt * sizeof(float));
            para_init_val(wbBnP.bn_var, bn_vcnt, 1.0f);
        }
    }
}
void free_wbBnP(wbBnP& wbBnP,bool update=false){
    cudaFree(wbBnP.bias);
    cudaFree(wbBnP.weight);
    cudaFree(wbBnP.bn_bias);
    if(update == false)
    {
        cudaFree(wbBnP.bn_mean);
        cudaFree(wbBnP.bn_var);
    }
    cudaFree(wbBnP.bn_weight);
}
void memset_wbBnP(const std::string& layer,const std::string& cf,wbBnP& wbBnP,int i,int param_offset) 
{
    std::string cfiStr = std::to_string(i);
    std::string biStr = std::to_string(i+param_offset);
    std::string name = layer + cf + cfiStr;
    std::string bnStr = layer + "bn" + biStr;   
    cudaMemset(wbBnP.weight, 0, params[name + ".weight"].size() * sizeof(float)); 
    cudaMemset(wbBnP.bias, 0, params[name + ".bias"].size() * sizeof(float));
    cudaMemset(wbBnP.bn_weight, 0,params[bnStr + ".weight"].size() * sizeof(float));
    cudaMemset(wbBnP.bn_bias, 0, params[bnStr + ".bias"].size() * sizeof(float));
}

struct CB3P {
    wbBnP cb1;
    wbBnP cb2;
    wbBnP cb3;
};
void read_CB3P(const std::string& layer,CB3P& CB3P,bool update=false,
float IC1=0,float IC2=0,float IC3=0) {
    read_wbBnP(layer,"conv",CB3P.cb1,1,0,update,IC1);
    read_wbBnP(layer,"conv",CB3P.cb2,2,0,update,IC2);
    read_wbBnP(layer,"conv",CB3P.cb3,3,0,update,IC3);   
}
void free_CB3P(CB3P &CB3P,bool update=false){
    free_wbBnP(CB3P.cb1,update);
    free_wbBnP(CB3P.cb2,update);
    free_wbBnP(CB3P.cb3,update);
}
void memset_CB3P(const std::string& layer,CB3P& CB3P) {
    memset_wbBnP(layer,"conv",CB3P.cb1,1,0);
    memset_wbBnP(layer,"conv",CB3P.cb2,2,0);
    memset_wbBnP(layer,"conv",CB3P.cb3,3,0);   
}

struct FB2FP {
    wbBnP fb1;
    wbBnP fb2;
    fcp   f3;
};
void read_FB2FP(const std::string& layer,FB2FP& FB2FP,int param_offset,bool update=false,
float IC1=0,float IC2=0,float IC3=0)    {
    read_wbBnP(layer,"fc",FB2FP.fb1,1,param_offset,update,IC1);
    read_wbBnP(layer,"fc",FB2FP.fb2,2,param_offset,update,IC2);
    read_fcp(layer,FB2FP.f3,3,update,IC3);
}
void free_FB2FP(FB2FP &FB2FP,bool update=false){
    free_wbBnP(FB2FP.fb1,update);
    free_wbBnP(FB2FP.fb2,update);
    free_fcp(FB2FP.f3);
}
void memset_FB2FP(const std::string& layer,FB2FP& FB2FP,int param_offset)    
{
    memset_wbBnP(layer,"fc",FB2FP.fb1,1,param_offset);
    memset_wbBnP(layer,"fc",FB2FP.fb2,2,param_offset);
    memset_fcp(layer,FB2FP.f3,3);
}

struct stndP {
    CB3P cb3;
    FB2FP fb2f;
};
void read_stndP(const std::string& layer,stndP& stndP,bool update=false,
float IC1=0,float IC2=0,float IC3=0,
float fIC1=0,float fIC2=0,float fIC3=0) {
    read_CB3P(layer,stndP.cb3,update,IC1,IC2,IC3);
    read_FB2FP(layer,stndP.fb2f,3,update,fIC1,fIC2,fIC3);
}
void free_stndP(stndP& stndP,bool update=false){
    free_CB3P(stndP.cb3,update);
    free_FB2FP(stndP.fb2f,update);
}
void memset_stndP(const std::string& layer,stndP& stndP) {
    memset_CB3P(layer,stndP.cb3);
    memset_FB2FP(layer,stndP.fb2f,3);
}

struct cudaP {
    stndP stn3dp;
    stndP stnkdp;
    CB3P  featp;
    FB2FP nonep;
};
void freeDP(cudaP &dp,bool update=false)
{
    free_stndP(dp.stn3dp,update);
    free_stndP(dp.stnkdp,update);
    free_CB3P(dp.featp,update);
    free_FB2FP(dp.nonep,update);
}
void memsetDP(cudaP &dp)
{
    memset_stndP("feat.stn.",dp.stn3dp);
    memset_stndP("feat.fstn.",dp.stnkdp);
    memset_CB3P("feat.",dp.featp);
    memset_FB2FP("",dp.nonep,0);
}

cudaP dParams;
cudaP upParams;
cudaP moParams;


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
__global__ void gemm_64x64_128N_kernel(int M,int N,int K,
float* input_A, float* input_B, float* convBias,float* output, float beta = 0.0f)
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
    int bW = b * K * M ;
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
    int W_LoadG = INDEX(W_grow, W_gcol, K)+bW;
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
            W_shared[W_StoreS+ldg]=input_A[W_LoadG+ldg*K];
        }
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            I_shared[I_StoreS+ldg*32]=input_B[I_LoadG+ldg*32];
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
    //store to C
    if (beta!= 0.0f)
    {
        #pragma unroll
        for (int i = 0; i<4;i++)
        {
            for (int j = 0 ; j<4 ;j++)
            {
                output[O_StoreG+ i*N+j]+=O_reg[i][j];
            }
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i<4;i++)
        {
            for (int j = 0 ; j<4 ;j++)
            {
                output[O_StoreG+ i*N+j]=O_reg[i][j];
            }
        }
    }
}

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
        for (int l = 0; l < L; l++)
        {
            int iIdx = l + index * L;
            outputDelta[iIdx] = softmax[l]-(correct_label==l);
        }
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

__global__ void Maxpooling_Kernel_train(float* input,float* output,float* output_idx,int numPoints)
{
    __shared__ float sharedMax[1024];
    __shared__ int sharedMax_idx[1024];
    
    int tx = threadIdx.x;
    int channel = blockIdx.x;

    float localMax = -FLT_MAX;
    int   localMax_idx = -1;
    int cnum = channel * numPoints;
    for (int i = tx; i < numPoints; i += blockDim.x) {
        float val = input[cnum + i];
        if (val > localMax) {
            localMax = val;
            localMax_idx = i;
        }
    }
    sharedMax[tx] = localMax;
    sharedMax_idx[tx] = localMax_idx;
    __syncthreads();

    // 归约：逐步计算块内的最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            if (sharedMax[tx + stride] > sharedMax[tx]) {
                sharedMax[tx] = sharedMax[tx + stride];
                sharedMax_idx[tx] = sharedMax_idx[tx + stride];
            }
        }
        __syncthreads();
    }

    // 线程0写入最终的最大值
    if (tx == 0) {
        output[channel] = sharedMax[0];
        output_idx[channel] = sharedMax_idx[0];
    }
}
void GPU_MaxPooling_train(int ics, int batchSize, int numPoints,float* input, float* output, float* output_idx)
{
    //std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    dim3 blockDim(1024);
    Maxpooling_Kernel_train<<<gridDim, blockDim>>>(input, output,output_idx,numPoints);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
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
        float *C_gpu, int ldc , int BatchSize = 32)
{
    // if (M == 64 && K == 64 && N == 64 && TB == false && BETA == 0.0f)
    // {
    //     dim3 grid64(DIV_UP(N, 64),DIV_UP(M, 64),BatchSize);
    //     if (TA == true)
    //     {
    //         GPU_transpose(A_gpu, for_trans_mul , BatchSize, M, K);
    //         gemm_64x64_128N_kernel<<<grid64, 256>>>(K,N,M,for_trans_mul,B_gpu,NULL,C_gpu);
    //     }
    //     else
    //         gemm_64x64_128N_kernel<<<grid64, 256>>>(M,N,K,A_gpu,B_gpu,NULL,C_gpu);
    // }
    // else
    // {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cublasSgemm failed with error: ");
            checkCublasStatus(status);
        }
        cublasDestroy(handle);
    // }
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
    // if (M_A == 64 && K_A == 64 && K_B == 64 && N_B == 64)
    // {
    //     dim3 grid64(DIV_UP(N_B, 64),DIV_UP(M_A, 64),BatchSize);
    //     gemm_64x64_128N_kernel<<<grid64, 256>>>(M_A,N_B,K_A,input_A,input_B,NULL,output);
    // }
    // else
    // {
        for(int b=0;b<BatchSize;b++)
        {
        check_error(cudaPeekAtLastError());
        //cudaDeviceSynchronize();
        gemm_gpu(false,false,M_A,N_B,K_A,1.0f, input_A+b*M_A*K_A,K_A,input_B+b*K_B*N_B,N_B,0.0f,output+b*M_A*N_B,N_B);
        check_error(cudaPeekAtLastError());
        }
    // }
}

void Bmm_bp(float* input_A,float* input_B,float* delta_a,float* delta_b,float* delta_from,
int M_A,int K_A,int K_B,int N_B,int BatchSize = 1,bool genA = true,float add_b = 0.0f)
{
    //std::cout << "--------BMM" << std::endl;
    for(int b=0;b<BatchSize;b++)
    {
        check_error(cudaPeekAtLastError());
        //cudaDeviceSynchronize();
        gemm_gpu(true,false,K_A,N_B,M_A,1.0f,  input_A+b*M_A*K_A,K_A,  delta_from+b*M_A*N_B, N_B, add_b, delta_b+b*K_B*N_B,N_B);
        if(genA)
        {
            gemm_gpu(false,true,M_A,K_A,N_B,1.0f,  delta_from+b*M_A*N_B, N_B,  input_B+b*K_B*N_B,N_B, 0.0f, delta_a+b*M_A*K_A,K_A);
        }
        check_error(cudaPeekAtLastError());
    }
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
    normalize_gpu(input,norm,mean,var,batchSize,outChannels,numPoints);
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
#ifdef DEBUG
    std::cout << "----START CBR_3 TRAIN" << std::endl;
#endif
    GPU_CBR_train(relu,batchSize, numPoints, inics, OC1, cb3p.cb1, input, relu1_output,conv1_output,bn1);
    GPU_CBR_train(relu,batchSize, numPoints, OC1, OC2, cb3p.cb2, relu1_output, relu2_output,conv2_output,bn2);
    GPU_CBR_train(relu,batchSize, numPoints, OC2, OC3, cb3p.cb3, relu2_output, output,conv3_output,bn3);
}

__global__ void FC_Kernel_gemv(int M,int batchSize,int N,float* input, 
float* fcWeights, float* fcBias,float* output,float dropout = 0.0f,unsigned int seed = 0)
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
        res += fcBias[oc];
        int index = oc + batch * M;
        if(laneId==0) 
        {
            if(dropout != 0.0f)
            {
                curandState state;
                //unsigned long long thread_seed = seed ^ (idx * 0x5DEECE66DLL + 0xB);
                curand_init(seed, index, 0, &state);  
                float rand_val = curand_uniform(&state);
                if (rand_val < dropout)
                {
                    res = 0.0f;
                }
                else
                {
                    res = res / (1.0f - dropout);
                }
            }
            output[index]  = res;
        }
    }
}
void FBRWRAP_GPU_train(int batchSize,int inFeatures,int outFeatures,float* input, 
float* cudaFcWeights, float* cudaFcBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,
float* output,float* fcOutput,bn_layer& bn,float dropout = 0.0f)
{
    dim3 blockDim(32,4);
    dim3 gridDim((outFeatures + 4 - 1) / 4,batchSize);//X:宽度 Y：高度
    unsigned int seed = time(0); 
    FC_Kernel_gemv<<<gridDim,blockDim>>>(outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,fcOutput,dropout,seed);
    //BR_Kernel<<<batchSize, outFeatures>>>(true,1,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,fcOutput,output);
    //TODO:这里用上面的会更快
    // normalize_gpu(fcOutput,output,cudaBnRM,cudaBnRV,batchSize,outFeatures,1);
    // madd_relu(true,output,output,cudaBnWeights,cudaBnBias,batchSize,outFeatures,1);
    BR_train(true,batchSize,1,outFeatures,cudaBnWeights,cudaBnBias,
    bn.mean,bn.var,bn.norm,cudaBnRM,cudaBnRV,fcOutput,output);
}
void GPU_FBR_train(int batchSize, int inFeatures, int outFeatures,wbBnP& fbp, 
float* input, float* reluOutput, float* fcOutput,bn_layer& bn,float dropout = 0.0f)
{
    FBRWRAP_GPU_train(batchSize,inFeatures,outFeatures,input,
    fbp.weight,fbp.bias,
    fbp.bn_weight,fbp.bn_bias,
    fbp.bn_mean,fbp.bn_var,reluOutput,fcOutput,bn,dropout);
}
void GPU_FBR_2_F_train(int OC1,int OC2,int OC3,int batchSize,int inics,FB2FP &fb2f, 
float* input, float* output,float* relu1_output,float* relu2_output,
float* fc1_output,float* fc2_output,bn_layer& bn1,bn_layer& bn2,int param_offset=3,float dropout = 0.0f)
{
#ifdef DEBUG
    std::cout << "----START FBR_2_F TRAIN" << std::endl;
#endif
    GPU_FBR_train(batchSize,inics,OC1,fb2f.fb1,input,relu1_output,fc1_output,bn1);
    GPU_FBR_train(batchSize,OC1,OC2,fb2f.fb2,relu1_output,relu2_output,fc2_output,bn2,dropout);
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
__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter]= part[i]; //TODO:+= part[i];
    }
}
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}
__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
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
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}
__global__ void fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
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

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}
void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}
__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    delta[index] = delta[index] * 1.f/(sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}
void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}
__global__ void relu_detla_kernel(float *output,float *delta,int N)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) delta[i] = (output[i] == 0)?0:delta[i];
}
void relu_detla_gpu(float *output,float *delta,int N)
{
    relu_detla_kernel<<<cuda_gridsize(N), BLOCK>>>(output,delta,N);
    check_error(cudaPeekAtLastError());
}

__global__ void dp_detla_kernel(float *output,float *delta,int N,float dp = 0.4f)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) delta[i] = (output[i] == 0.0f) ? 0.0f : delta[i] * (1.0f / (1.0f-dp));
}
void dp_detla_gpu(float *output,float *delta,int N,float dp = 0.4f)
{
    dp_detla_kernel<<<cuda_gridsize(N), BLOCK>>>(output,delta,N);
    check_error(cudaPeekAtLastError());
}

void BR_bp(bool relu,int numFeatures, int batchSize, int numPoints,
float* weight, float* input,float* output,
float* mean, float* var, float* norm,
float* delta_from, float* delta_gen, float* delta_mean, float* delta_var,
float* weight_up, float* bias_up,float esp = 1e-5)
{
    if(relu){relu_detla_gpu(output,delta_from,batchSize*numPoints*numFeatures);}
    backward_bias_gpu(bias_up, delta_from, batchSize, numFeatures, numPoints);
    backward_scale_gpu(norm, delta_from, batchSize, numFeatures, numPoints, weight_up);

    scale_bias_gpu(delta_from, weight, batchSize, numFeatures, numPoints);

    fast_mean_delta_gpu(delta_from, var, batchSize, numFeatures, numPoints, delta_mean);
    fast_variance_delta_gpu(input, delta_from, mean, var, batchSize, numFeatures, numPoints, delta_var);
    normalize_delta_gpu(input, mean, var, delta_mean, delta_var, batchSize, numFeatures, numPoints, delta_from);
}
void F_bp(int batchSize, int inFeatures, int outFeatures,float* input,float* weight,
float * delta_from, float * delta_gen, float* weight_up, float* bias_up,float dp = 0.0f,float* output=NULL) {
    //delta gen: batchsize,outf * outf,inf
    int M = batchSize;
    int N = inFeatures;
    int K = outFeatures;
    if(dp != 0.0f){dp_detla_gpu(output,delta_from,batchSize*outFeatures,dp);}
    gemm_gpu(false, false, M, N, K, 1.0, delta_from,K , weight,N, 0.0, delta_gen,N);
    //WEIGHT UP: outf,batchsize * batchsize,inf
    M = outFeatures;
    N = inFeatures;
    K = batchSize;
    gemm_gpu(true, false, M, N, K, 1.0, delta_from,M,input,N,0.0,weight_up,N);
    //BIAS UP: outf,batchsize
    backward_bias_gpu(bias_up,delta_from,batchSize,outFeatures,1);
}
void FBR_bp(int batchSize, int inFeatures, int outFeatures,
wbBnP& fbp, wbBnP& fbp_up, float* input, float* reluOutput, float* fcOutput,
float* delta_from,float* delta_fc,float* delta_gen,
bn_layer& bn,bn_layer& bn_delta,float dp = 0.0f)
{
    BR_bp(true,outFeatures,batchSize,1,
    fbp.bn_weight,fcOutput,reluOutput,
    bn.mean,bn.var,bn.norm,
    delta_from,delta_fc,bn_delta.mean,bn_delta.var,
    fbp_up.bn_weight, fbp_up.bn_bias);
    F_bp(batchSize,inFeatures,outFeatures,input,fbp.weight,
    delta_fc, delta_gen, fbp_up.weight, fbp_up.bias, dp, fcOutput);
}
void FBR2F_bp(int OC1,int OC2,int OC3,int batchSize,int inics,
FB2FP &fb2f,FB2FP &fb2f_up, float* input, 
float* relu1_output,float* relu2_output,float* fc1_output,float* fc2_output,
float* delta_fc1,float* delta_fc2,float* delta_from,
float* delta_relu1,float* delta_relu2,float* delta_gen, 
bn_layer& bn1,bn_layer& bn1_delta,bn_layer& bn2,bn_layer& bn2_delta,float dp = 0.0f){

#ifdef BACKDEBUG
    std::cout << "----START FBR2F_bp" << std::endl;
#endif
    F_bp(batchSize,OC2,OC3,relu2_output,fb2f.f3.weight,
    delta_from,delta_relu2,fb2f_up.f3.weight,fb2f_up.f3.bias);

    FBR_bp(batchSize,OC1,OC2,
    fb2f.fb2, fb2f_up.fb2, relu1_output, relu2_output, fc2_output,
    delta_relu2, delta_fc2, delta_relu1, bn2, bn2_delta,dp);
    
    FBR_bp(batchSize,inics,OC1,
    fb2f.fb1, fb2f_up.fb1,  input, relu1_output, fc1_output,
    delta_relu1, delta_fc1, delta_gen, bn1, bn1_delta);
}

__global__ void Maxpooling_bp_Kernel(float* delta_from,float* delta_gen,float* idx,int numPoints)
{
    int tx = threadIdx.x;
    int channel = blockIdx.x;
    float idx_val       = idx[channel];
    float delta_val     = delta_from[channel];
    int cnum = channel * numPoints;
    for (int i = tx; i < numPoints; i += blockDim.x) {
        delta_gen[cnum + i] = (i == idx_val)? delta_val : 0;
    }
}
void MaxPooling_bp(int ics, int batchSize, int numPoints, float* idx, float* delta_from, float* delta_gen)
{
    //std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    dim3 blockDim(1024);
    Maxpooling_bp_Kernel<<<gridDim, blockDim>>>(delta_from, delta_gen, idx, numPoints);
}

void conv_bp(int batchSize,int numPoints,int inChannels,int outChannels,
float* input, float* weight, float * delta_from, float * delta_gen, float* weight_up, float* bias_up , float add_up=0.0f) {
    //delta gen
    int M = inChannels;
    int N = numPoints;
    int K = outChannels;
    for (int i = 0; i < batchSize; i++)
    {
        gemm_gpu(true, false, M, N, K, 1.0, weight, M, delta_from+i*K*N, N, add_up, delta_gen+i*M*N, N);
    }
    //WEIGHT UP:
    M = outChannels;
    N = inChannels;
    K = numPoints;
    for (int i = 0; i < batchSize; i++)
    {
        if (i == 0)
        {
            gemm_gpu(false, true, M, N, K, 1.0, delta_from+i*K*M,K, input+i*K*N,K, 0.0, weight_up,N);
        }
        else
        {
            gemm_gpu(false, true, M, N, K, 1.0, delta_from+i*K*M,K, input+i*K*N,K, 1.0, weight_up,N);
        }
    }
    //BIAS UP: 
    backward_bias_gpu(bias_up,delta_from,batchSize,outChannels,numPoints);
}
void CBR_bp(bool relu,int batchSize, int numPoints, int inFeatures, int outFeatures,
wbBnP& cbp, wbBnP& cbp_up, float* input, float* reluOutput, float* convOutput,
float* delta_from,float* delta_conv,float* delta_gen,
bn_layer& bn,bn_layer& bn_delta,float add_up=0.0f)
{
    BR_bp(relu,outFeatures,batchSize,numPoints,
    cbp.bn_weight,convOutput,reluOutput,
    bn.mean,bn.var,bn.norm,
    delta_from,delta_conv,bn_delta.mean,bn_delta.var,
    cbp_up.bn_weight, cbp_up.bn_bias);
    conv_bp(batchSize,numPoints,inFeatures,outFeatures,input,cbp.weight,
    delta_conv, delta_gen, cbp_up.weight, cbp_up.bias , add_up);
}
void CBR3_bp(bool relu,int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,
CB3P &cb3p,CB3P &cb3p_up, float* input, float* output,
float* relu1_output,float* relu2_output,float* conv1_output,float* conv2_output,float* conv3_output,
float* delta_conv1,float* delta_conv2,float* delta_conv3,
float* delta_from, float* delta_relu1,float* delta_relu2,float* delta_gen, 
bn_layer& bn1,bn_layer& bn2,bn_layer& bn3,
bn_layer& bn1_delta,bn_layer& bn2_delta,bn_layer& bn3_delta,float add_up=0.0f)
{
    #ifdef BACKDEBUG
    std::cout << "----START CBR3_bp" << std::endl;
    #endif
    CBR_bp(relu, batchSize, numPoints, OC2, OC3, cb3p.cb3, cb3p_up.cb3,
    relu2_output, output, conv3_output,
    delta_from, delta_conv3, delta_relu2, bn3,bn3_delta);

    CBR_bp(relu, batchSize, numPoints, OC1, OC2, cb3p.cb2, cb3p_up.cb2,
    relu1_output, relu2_output, conv2_output,
    delta_relu2, delta_conv2, delta_relu1, bn2,bn2_delta);

    CBR_bp(relu, batchSize, numPoints, inics, OC1, cb3p.cb1, cb3p_up.cb1,
    input, relu1_output, conv1_output,
    delta_relu1, delta_conv1, delta_gen, bn1,bn1_delta , add_up);
}

__global__ void BP_UPDATE_Kernal_Momentum(float *N, float *delta, float *momentum, int width, float learning_rate, float momentum_factor) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index < width) {
        //printf("index: %d\n", index);
        //printf("%f ", momentum[index]);
        //momentum_factor = 0.0f;
        momentum[index] = momentum_factor * momentum[index] + (1-momentum_factor) * delta[index];
        N[index] = N[index] + learning_rate*momentum[index];
    }
}
void BP_UPDATE_Momentum(float *N, float *delta, float *momentum, int width, float learning_rate=-0.01f, float momentum_factor=0.9f) {
    BP_UPDATE_Kernal_Momentum<<<cuda_gridsize(width), BLOCK>>>(N, delta, momentum, width, learning_rate, momentum_factor);
    check_error(cudaPeekAtLastError());
}
void FB_update(int IC,int OC,wbBnP& fbp, wbBnP& fbp_up,wbBnP& fbp_mo)
{
    BP_UPDATE_Momentum(fbp.bn_weight, fbp_up.bn_weight, fbp_mo.bn_weight, OC);
    BP_UPDATE_Momentum(fbp.bn_bias, fbp_up.bn_bias, fbp_mo.bn_bias, OC);
    BP_UPDATE_Momentum(fbp.weight, fbp_up.weight, fbp_mo.weight, OC*IC);
    BP_UPDATE_Momentum(fbp.bias, fbp_up.bias, fbp_mo.bias, OC);
}
void FBR2F_update(int OC1,int OC2,int OC3,int inics,
FB2FP &fb2f,FB2FP &fb2f_up,FB2FP &fb2f_mo)
{
    BP_UPDATE_Momentum(fb2f.f3.weight, fb2f_up.f3.weight, fb2f_mo.f3.weight, OC3*OC2);
    BP_UPDATE_Momentum(fb2f.f3.bias, fb2f_up.f3.bias, fb2f_mo.f3.bias, OC3);
    FB_update(OC1,OC2,fb2f.fb2, fb2f_up.fb2, fb2f_mo.fb2);
    FB_update(inics,OC1,fb2f.fb1, fb2f_up.fb1, fb2f_mo.fb1);
}
void CBR3_update(int OC1,int OC2,int OC3,int inics,
CB3P &cb3p,CB3P &cb3p_up,CB3P &cb3p_mo)
{
    FB_update(OC2,OC3,cb3p.cb3, cb3p_up.cb3, cb3p_mo.cb3);
    FB_update(OC1,OC2,cb3p.cb2, cb3p_up.cb2, cb3p_mo.cb2);
    FB_update(inics,OC1,cb3p.cb1, cb3p_up.cb1, cb3p_mo.cb1);
}

__global__ void compute_frobenius_norm(
    const float* trans_mult_transT, // 输入：形状为 (batch_size, num_features, num_features)
    const float* transT, 
    float* delta,         // 输出：形状为 (batch_size,)
    int batch_size, int num_features) 
{
    extern __shared__ float shared_mem[]; // 每个 block 共享内存，用于中间结果
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int feature_size = num_features * num_features;
    int bf = batch_idx * feature_size;

    // 每个线程计算一个元素的平方和
    float sum = 0.0f;
    for (int i = thread_idx; i < feature_size; i += blockDim.x) {
        float value = trans_mult_transT[bf + i];
        int row = i / num_features;
        int col = i % num_features;
        if (row == col) value -= 1.0f; 
        sum += value * value; // 计算平方和
    }

    // 共享内存归约
    shared_mem[thread_idx] = sum;
    __syncthreads();

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_mem[thread_idx] += shared_mem[thread_idx + stride];
        }
        __syncthreads();
    }

    float frobenius_norms = sqrtf(shared_mem[0]);
    for (int i = thread_idx; i < feature_size; i += blockDim.x) {
        float mulres = trans_mult_transT[bf + i];
        int row = i / num_features;
        int col = i % num_features;
        if (row == col) mulres -= 1.0f; 
        mulres = 2 * 0.001f * mulres * transT[bf + i] / frobenius_norms; 
        delta[bf+i] = mulres / batch_size;
    }
}

// 封装的函数：配置和调用 kernel
void launch_compute_frobenius_norm(
    const float* d_trans_mult_transT,  // 输入：GPU 上的 trans * trans^T 数据
    const float* d_transT,
    float* delta,          // 输出：GPU 上的 Frobenius 范数
    int batch_size, int num_features, int block_size = 1024) 
{
    int shared_mem_size = block_size * sizeof(float);

    // 启动 Kernel，每个批次对应一个 Block
    compute_frobenius_norm<<<batch_size, block_size, shared_mem_size>>>(
        d_trans_mult_transT,d_transT, delta, batch_size, num_features);

    // 检查 Kernel 运行是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void mat_diff_loss_backward(
    const float* trans_feat, // (batch_size, num_features, num_features)
    float* grad_trans_feat,  // (batch_size, num_features, num_features)
    int batch_size, int num_features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_size = num_features * num_features;

    if (idx < batch_size * feature_size) {
        int batch_idx = idx / feature_size;
        int row = (idx % feature_size) / num_features;
        int col = idx % num_features;

        // Compute trans_feat[batch_idx] * trans_feat[batch_idx]^T
        float diff = 0.0;
        for (int k = 0; k < num_features; ++k) {
            diff += trans_feat[batch_idx * feature_size + row * num_features + k] *
                    trans_feat[batch_idx * feature_size + (row+k) * num_features + col];
        }

        // Subtract identity matrix
        if (row == col) {
            diff -= 1.0f; // I[row][col] = 1
        }

        // Compute gradient for backward pass
        grad_trans_feat[idx] = 2 * diff * 0.001f * trans_feat[batch_idx * feature_size + row * num_features + col];;  // Scaled by 2
    }
}

void compute_mat_diff_grad(const float* trans_feat, float* grad, int numFeatures, int batchSize) {
    dim3 block(256);
    dim3 grid((batchSize * numFeatures * numFeatures + block.x - 1) / block.x);
    mat_diff_loss_backward<<<grid, block>>>(trans_feat, grad, batchSize, numFeatures);
}


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
    int stn_5_idx = batchSize*OC3;
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
    int fstn_4_idx= batchSize * fstn_OC3 ;
    int fstn_5_fc= batchSize * fstn_FC_OC1 ;
    int fstn_6_fc= batchSize * fstn_FC_OC2 ;
    int fstn_5= batchSize * fstn_FC_OC1 ;
    int fstn_6= batchSize * fstn_FC_OC2 ;
    int fstn_7= transFeatSize ;
    int fstn_8= transFeatSize ;
    //part4
    int part4_1= bn * fstn_inChannel ;
    int part4_2= batchSize*numPoints*fstn_inChannel ;
    int part4_3= batchSize*fstn_inChannel*numPoints ;
    int part4_4= batchSize*encoderOC2*numPoints ;int part4_4_conv= batchSize*encoderOC2*numPoints ;
    int part4_5= bnEOC3 ;int part4_5_conv= bnEOC3 ;
    if (USECONVMAX == 1) { part4_5 = part4_5 / 64 ;}
    int part4_6= batchSize * encoderOC3 ;
    int part4_6_idx= batchSize * encoderOC3 ;
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
    net.maxp_output_idx = device_output+offset;offset += stn_5_idx;

    net.fc1_output_stn_cbr = device_output+offset;offset += stn_6_fc;
    offset = alloc_bn(net.bn1_stn_fbr2f,device_output,offset,FC_OC1,batchSize);
    net.relu1_output_stn_fbr2f = device_output+offset;offset += stn_6;

    net.fc2_output_stn_cbr = device_output+offset;offset += stn_7_fc;
    offset = alloc_bn(net.bn2_stn_fbr2f,device_output,offset,FC_OC2,batchSize);
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
    net.fstn_maxp_output_idx = device_output+offset;offset += fstn_4_idx;

    net.fc1_output_fstn_fbr2f = device_output+offset;offset += fstn_5_fc;
    offset = alloc_bn(net.bn1_fstn_fbr2f,device_output,offset,fstn_FC_OC1,batchSize);
    net.relu1_output_fstn_fbr2f = device_output+offset;offset += fstn_5;
    
    net.fc2_output_fstn_fbr2f = device_output+offset;offset += fstn_6_fc;
    offset = alloc_bn(net.bn2_fstn_fbr2f,device_output,offset,fstn_FC_OC2,batchSize);
    net.relu2_output_fstn_fbr2f = device_output+offset;offset += fstn_6;
    net.stnkd_out = device_output+offset;offset += fstn_7;
    net.stnkd_out_trans = device_output+offset;offset += fstn_8;

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
    net.encoder_output_idx = device_output+offset;offset += part4_6_idx;
    //classify
    net.fc1_output_part5_fbr2f = device_output+offset;offset += cla_1_fc;
    offset = alloc_bn(net.bn1_part5_fbr2f,device_output,offset,ch_half,batchSize);
    net.relu1_output_part5_fbr2f = device_output+offset;offset += cla_1;
    net.fc2_output_part5_fbr2f = device_output+offset;offset += cla_2_fc;
    offset = alloc_bn(net.bn2_part5_fbr2f,device_output,offset,ch_quarter,batchSize);
    net.relu2_output_part5_fbr2f = device_output+offset;offset += cla_2;
    net.softmax_input = device_output+offset;offset += cla_3;
    net.softmax_output = device_output+offset;offset+= cla_4;

    TNET delta;
    offset = 0;
    //stn3d 
    delta.input_trans = device_delta+offset;offset += stn_1;

    delta.conv1_output_stn_cbr = device_delta+offset;offset += stn_2_conv;
    offset = alloc_bn(delta.bn1_stn_cbr,device_delta,offset,OC1,bn);
    delta.relu1_output_stn_cbr = device_delta+offset;offset += stn_2;

    delta.conv2_output_stn_cbr = device_delta+offset;offset += stn_3_conv;
    offset = alloc_bn(delta.bn2_stn_cbr,device_delta,offset,OC2,bn);
    delta.relu2_output_stn_cbr = device_delta+offset;offset += stn_3;

    delta.conv3_output_stn_cbr = device_delta+offset;offset += stn_4_conv;
    offset = alloc_bn(delta.bn3_stn_cbr,device_delta,offset,OC3,bn);
    delta.CBR3_output = device_delta+offset;offset += stn_4;
    
    delta.maxp_output = device_delta+offset;offset += stn_5;
    delta.maxp_output_idx = device_delta+offset;offset += stn_5_idx;

    delta.fc1_output_stn_cbr = device_delta+offset;offset += stn_6_fc;
    offset = alloc_bn(delta.bn1_stn_fbr2f,device_delta,offset,FC_OC1,batchSize);
    delta.relu1_output_stn_fbr2f = device_delta+offset;offset += stn_6;

    delta.fc2_output_stn_cbr = device_delta+offset;offset += stn_7_fc;
    offset = alloc_bn(delta.bn2_stn_fbr2f,device_delta,offset,FC_OC2,batchSize);
    delta.relu2_output_stn_fbr2f = device_delta+offset;offset += stn_7;
    delta.stn3d_out = device_delta+offset;offset += stn_8;

    //part2
    delta.bmm1_res = device_delta+offset;offset += part2_1;
    delta.bmm1_res_trans = device_delta+offset;offset += part2_2;
    delta.fstn_input_conv = device_delta+offset;offset += part2_3;
    offset = alloc_bn(delta.fstn_input_bn,device_delta,offset,fstn_inChannel,bn);
    delta.fstn_input = device_delta+offset;offset += part2_4;

    //stnkd
    delta.conv1_output_fstn_cbr = device_delta+offset;offset += fstn_1_conv;
    offset = alloc_bn(delta.bn1_fstn_cbr,device_delta,offset,fstn_OC1,bn);
    delta.relu1_output_fstn_cbr = device_delta+offset;offset += fstn_1;

    delta.conv2_output_fstn_cbr = device_delta+offset;offset += fstn_2_conv;
    offset = alloc_bn(delta.bn2_fstn_cbr,device_delta,offset,fstn_OC2,bn);
    delta.relu2_output_fstn_cbr = device_delta+offset;offset += fstn_2;

    delta.conv3_output_fstn_cbr = device_delta+offset;offset += fstn_3_conv;
    offset = alloc_bn(delta.bn3_fstn_cbr,device_delta,offset,fstn_OC3,bn);
    delta.fstn_CBR3_output = device_delta+offset;offset += fstn_3;

    delta.fstn_maxp_output = device_delta+offset;offset += fstn_4;
    delta.fstn_maxp_output_idx = device_delta+offset;offset += fstn_4_idx;

    delta.fc1_output_fstn_fbr2f = device_delta+offset;offset += fstn_5_fc;
    offset = alloc_bn(delta.bn1_fstn_fbr2f,device_delta,offset,fstn_FC_OC1,batchSize);
    delta.relu1_output_fstn_fbr2f = device_delta+offset;offset += fstn_5;
    
    delta.fc2_output_fstn_fbr2f = device_delta+offset;offset += fstn_6_fc;
    offset = alloc_bn(delta.bn2_fstn_fbr2f,device_delta,offset,fstn_FC_OC2,batchSize);
    delta.relu2_output_fstn_fbr2f = device_delta+offset;offset += fstn_6;
    delta.stnkd_out = device_delta+offset;offset += fstn_7;
    delta.stnkd_out_trans = device_output+offset;offset += fstn_8;

    //part4
    delta.fstn_input_trans = device_delta+offset;offset += part4_1;
    delta.fstn_bmm1_res = device_delta+offset;offset += part4_2;
    delta.fstn_bmm1_res_trans = device_delta+offset;offset += part4_3;

    delta.cbr2_output_conv = device_delta+offset;offset += part4_4_conv;
    offset = alloc_bn(delta.cbr2_output_bn,device_delta,offset,encoderOC2,bn);
    delta.cbr2_output = device_delta+offset;offset += part4_4;

    delta.feat_bn3_conv = device_delta+offset;offset += part4_5_conv;
    offset = alloc_bn(delta.feat_bn3_bn,device_delta,offset,encoderOC3,bn);
    delta.feat_bn3 = device_delta+offset;offset += part4_5;
    
    delta.encoder_output = device_delta+offset;offset += part4_6;
    delta.encoder_output_idx = device_delta+offset;offset += part4_6_idx;
    //classify
    delta.fc1_output_part5_fbr2f = device_delta+offset;offset += cla_1_fc;
    offset = alloc_bn(delta.bn1_part5_fbr2f,device_delta,offset,ch_half,batchSize);
    delta.relu1_output_part5_fbr2f = device_delta+offset;offset += cla_1;
    delta.fc2_output_part5_fbr2f = device_delta+offset;offset += cla_2_fc;
    offset = alloc_bn(delta.bn2_part5_fbr2f,device_delta,offset,ch_quarter,batchSize);
    delta.relu2_output_part5_fbr2f = device_delta+offset;offset += cla_2;
    delta.softmax_input = device_delta+offset;offset += cla_3;
    delta.softmax_output = device_delta+offset;offset+= cla_4;

#ifdef DEBUG
    std::cout << "PART1:STN3d, forwaring" << std::endl;
#endif
    int maxnp = USECONVMAX? numPoints / 64 : numPoints;
    GPU_transpose(input,net.input_trans,batchSize,numPoints,inChannels);
    GPU_CBR_3_train(true,OC1,OC2,OC3, batchSize, numPoints,inChannels,dParams.stn3dp.cb3, net.input_trans, 
    net.CBR3_output,net.relu1_output_stn_cbr,net.relu2_output_stn_cbr,
    net.conv1_output_stn_cbr,net.conv2_output_stn_cbr,net.conv3_output_stn_cbr,
    net.bn1_stn_cbr,net.bn2_stn_cbr,net.bn3_stn_cbr);   // conv-bn-relu * 3
    GPU_MaxPooling_train(OC3, batchSize, maxnp,net.CBR3_output, net.maxp_output,net.maxp_output_idx); // Max pooling    
    GPU_FBR_2_F_train(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3,dParams.stn3dp.fb2f,net.maxp_output,
    net.stn3d_out,net.relu1_output_stn_fbr2f,net.relu2_output_stn_fbr2f,
    net.fc1_output_stn_cbr,net.fc2_output_stn_cbr,net.bn1_stn_fbr2f,net.bn2_stn_fbr2f);// fc-bn-relu * 2 + fc
    matrix_add_I(net.stn3d_out,3,batchSize);

#ifdef DEBUG
    std::cout << "PART2:TRANS->BMM->TRANS->CBR, forwarding" << std::endl;
#endif
    GPU_Bmm(input,net.stn3d_out,net.bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    GPU_transpose(net.bmm1_res,net.bmm1_res_trans,batchSize,numPoints,encoderIC1);
    GPU_CBR_train(true,batchSize,numPoints,encoderIC1,fstn_inChannel,dParams.featp.cb1,net.bmm1_res_trans,net.fstn_input,net.fstn_input_conv,net.fstn_input_bn);

#ifdef DEBUG
    std::cout << "PART3:STNkd, forwarding"<< std::endl;
#endif
    GPU_CBR_3_train(true,fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,dParams.stnkdp.cb3, net.fstn_input, 
    net.fstn_CBR3_output,net.relu1_output_fstn_cbr,net.relu2_output_fstn_cbr,
    net.conv1_output_fstn_cbr,net.conv2_output_fstn_cbr,net.conv3_output_fstn_cbr,
    net.bn1_fstn_cbr,net.bn2_fstn_cbr,net.bn3_fstn_cbr);   // conv-bn-relu * 3
    GPU_MaxPooling_train(fstn_OC3, batchSize, maxnp,net.fstn_CBR3_output, net.fstn_maxp_output,net.fstn_maxp_output_idx); // Max pooling
    GPU_FBR_2_F_train(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3,dParams.stnkdp.fb2f,net.fstn_maxp_output,
    net.stnkd_out,net.relu1_output_fstn_fbr2f,net.relu2_output_fstn_fbr2f,
    net.fc1_output_fstn_fbr2f,net.fc2_output_fstn_fbr2f,net.bn1_fstn_fbr2f,net.bn2_fstn_fbr2f);// fc-bn-relu * 2 + fc
    matrix_add_I(net.stnkd_out,64,batchSize);

#ifdef DEBUG
    std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM, forwarding" << std::endl;
#endif
    GPU_transpose(net.fstn_input,net.fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    GPU_Bmm(net.fstn_input_trans,net.stnkd_out,net.fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    GPU_transpose(net.fstn_bmm1_res,net.fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    GPU_CBR_train(true,batchSize,numPoints,fstn_inChannel,encoderOC2,
    dParams.featp.cb2,net.fstn_bmm1_res_trans,net.cbr2_output,net.cbr2_output_conv,net.cbr2_output_bn);
    GPU_CBR_train(false,batchSize,numPoints,encoderOC2,encoderOC3,
    dParams.featp.cb3,net.cbr2_output, net.feat_bn3,net.feat_bn3_conv,net.feat_bn3_bn);
    GPU_MaxPooling_train(encoderOC3, batchSize, maxnp,net.feat_bn3, net.encoder_output, net.encoder_output_idx); // Max pooling
    
#ifdef DEBUG
    std::cout << "PART5:CLASSIFY, forwarding" << std::endl;
#endif
    float drop_rate = 0.0f;
    if (DROPOUT == 1) drop_rate = 0.4f;
    GPU_FBR_2_F_train(512,256,10,batchSize,encoderOC3,dParams.nonep,
    net.encoder_output,net.softmax_input,
    net.relu1_output_part5_fbr2f,net.relu2_output_part5_fbr2f,
    net.fc1_output_part5_fbr2f,net.fc2_output_part5_fbr2f,net.bn1_part5_fbr2f,net.bn2_part5_fbr2f,0,drop_rate);// fc-bn-relu * 2 + fc
    LogSoftMax_GPU_train(label,net.softmax_input,
    net.softmax_output,delta.softmax_input,correct_table,10,batchSize);


#ifdef BACKDEBUG
    std::cout << "BACKWARDING" << std::endl;
#endif
    FBR2F_bp(512,256,10, batchSize, encoderOC3, dParams.nonep, upParams.nonep,
    net.encoder_output,net.relu1_output_part5_fbr2f,net.relu2_output_part5_fbr2f,
    net.fc1_output_part5_fbr2f,net.fc2_output_part5_fbr2f,
    delta.fc1_output_part5_fbr2f,delta.fc2_output_part5_fbr2f,delta.softmax_input,
    delta.relu1_output_part5_fbr2f,delta.relu2_output_part5_fbr2f,
    delta.encoder_output,
    net.bn1_part5_fbr2f,delta.bn1_part5_fbr2f,
    net.bn2_part5_fbr2f,delta.bn2_part5_fbr2f, drop_rate
    );

#ifdef BACKDEBUG
    std::cout << "PART4, BACKWARDING" << std::endl;
#endif
    MaxPooling_bp(encoderOC3, batchSize, maxnp, net.encoder_output_idx, delta.encoder_output, delta.feat_bn3 );
    CBR_bp(false,batchSize,numPoints,encoderOC2,encoderOC3,
    dParams.featp.cb3, upParams.featp.cb3, net.cbr2_output, net.feat_bn3, net.feat_bn3_conv,
    delta.feat_bn3, delta.feat_bn3_conv, delta.cbr2_output, net.feat_bn3_bn, delta.feat_bn3_bn);
    CBR_bp(true,batchSize,numPoints,fstn_inChannel,encoderOC2,
    dParams.featp.cb2, upParams.featp.cb2, net.fstn_bmm1_res_trans,net.cbr2_output,net.cbr2_output_conv,
    delta.cbr2_output,delta.cbr2_output_conv, delta.fstn_bmm1_res_trans, net.cbr2_output_bn, delta.cbr2_output_bn);
    GPU_transpose(delta.fstn_bmm1_res_trans,delta.fstn_bmm1_res,batchSize,fstn_inChannel,numPoints);
    float mat_diff_addup = 0.0f;
    if(USEMATDIFF == 1)
    {
        mat_diff_addup = 1.0f;
        GPU_transpose(net.stnkd_out,net.stnkd_out_trans,batchSize,fstn_inChannel,fstn_inChannel);
        GPU_Bmm(net.stnkd_out,net.stnkd_out_trans,delta.stnkd_out_trans,64,64,64,64,batchSize);
        launch_compute_frobenius_norm(delta.stnkd_out_trans,net.stnkd_out_trans,delta.stnkd_out,batchSize,64);
        //compute_mat_diff_grad(net.stnkd_out,delta.stnkd_out,fstn_inChannel,batchSize);
    }
    Bmm_bp(net.fstn_input_trans,net.stnkd_out,delta.fstn_input_trans,delta.stnkd_out,delta.fstn_bmm1_res,
    numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize,true,mat_diff_addup);
    GPU_transpose(delta.fstn_input_trans, delta.fstn_input, batchSize,numPoints,fstn_inChannel);

#ifdef BACKDEBUG
    std::cout << "PART3:STNkd, backwarding" << std::endl;
#endif
    FBR2F_bp(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3, dParams.stnkdp.fb2f, upParams.stnkdp.fb2f,
    net.fstn_maxp_output,net.relu1_output_fstn_fbr2f,net.relu2_output_fstn_fbr2f,
    net.fc1_output_fstn_fbr2f,net.fc2_output_fstn_fbr2f,
    delta.fc1_output_fstn_fbr2f,delta.fc2_output_fstn_fbr2f,delta.stnkd_out,
    delta.relu1_output_fstn_fbr2f,delta.relu2_output_fstn_fbr2f,delta.fstn_maxp_output,
    net.bn1_fstn_fbr2f,delta.bn1_fstn_fbr2f,
    net.bn2_fstn_fbr2f,delta.bn2_fstn_fbr2f
    );
    MaxPooling_bp(fstn_OC3, batchSize, maxnp, net.fstn_maxp_output_idx, delta.fstn_maxp_output, delta.fstn_CBR3_output);
    CBR3_bp(true,fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,
    dParams.stnkdp.cb3, upParams.stnkdp.cb3, net.fstn_input, 
    net.fstn_CBR3_output,net.relu1_output_fstn_cbr,net.relu2_output_fstn_cbr,
    net.conv1_output_fstn_cbr,net.conv2_output_fstn_cbr,net.conv3_output_fstn_cbr,
    delta.conv1_output_fstn_cbr,delta.conv2_output_fstn_cbr,delta.conv3_output_fstn_cbr,
    delta.fstn_CBR3_output,delta.relu1_output_fstn_cbr,delta.relu2_output_fstn_cbr,delta.fstn_input, 
    net.bn1_fstn_cbr,net.bn2_fstn_cbr,net.bn3_fstn_cbr,
    delta.bn1_fstn_cbr,delta.bn2_fstn_cbr,delta.bn3_fstn_cbr,1.0f
    );

#ifdef DEBUG
    std::cout << "PART2:backwarding" << std::endl;
#endif
    CBR_bp(true,batchSize,numPoints,encoderIC1,fstn_inChannel,
    dParams.featp.cb1, upParams.featp.cb1, net.bmm1_res_trans, net.fstn_input, net.fstn_input_conv,
    delta.fstn_input, delta.fstn_input_conv, delta.bmm1_res_trans,
    net.fstn_input_bn,delta.fstn_input_bn);
    GPU_transpose(delta.bmm1_res_trans, delta.bmm1_res, batchSize, encoderIC1, numPoints);
    Bmm_bp(input,net.stn3d_out,NULL,delta.stn3d_out,delta.bmm1_res,
    numPoints,inChannels,inChannels,encoderIC1,batchSize,false);

#ifdef BACKDEBUG
    std::cout << "PART1:STN3d, backwarding" << std::endl;
#endif
    FBR2F_bp(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3, dParams.stn3dp.fb2f, upParams.stn3dp.fb2f,
    net.maxp_output,net.relu1_output_stn_fbr2f,net.relu2_output_stn_fbr2f,
    net.fc1_output_stn_cbr,net.fc2_output_stn_cbr,
    delta.fc1_output_stn_cbr,delta.fc2_output_stn_cbr,delta.stn3d_out,
    delta.relu1_output_stn_fbr2f,delta.relu2_output_stn_fbr2f,delta.maxp_output,
    net.bn1_stn_fbr2f,delta.bn1_stn_fbr2f,
    net.bn2_stn_fbr2f,delta.bn2_stn_fbr2f
    );
    MaxPooling_bp(OC3, batchSize, maxnp, net.maxp_output_idx, delta.maxp_output, delta.CBR3_output);
    CBR3_bp(true,OC1,OC2,OC3, batchSize, numPoints,inChannels,
    dParams.stn3dp.cb3, upParams.stn3dp.cb3, net.input_trans, 
    net.CBR3_output,net.relu1_output_stn_cbr,net.relu2_output_stn_cbr,
    net.conv1_output_stn_cbr,net.conv2_output_stn_cbr,net.conv3_output_stn_cbr,
    delta.conv1_output_stn_cbr,delta.conv2_output_stn_cbr,delta.conv3_output_stn_cbr,
    delta.CBR3_output,delta.relu1_output_stn_cbr,delta.relu2_output_stn_cbr,delta.input_trans, 
    net.bn1_stn_cbr,net.bn2_stn_cbr,net.bn3_stn_cbr,
    delta.bn1_stn_cbr,delta.bn2_stn_cbr,delta.bn3_stn_cbr
    );//这里的delta.input_trans, 没必要生成TODO:

    FBR2F_update(512,256,10,encoderOC3,dParams.nonep,upParams.nonep,moParams.nonep);
    FB_update(encoderOC2,encoderOC3,dParams.featp.cb3, upParams.featp.cb3, moParams.featp.cb3);//CB
    FB_update(fstn_inChannel,encoderOC2,dParams.featp.cb2, upParams.featp.cb2, moParams.featp.cb2);//CB
    FBR2F_update(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,fstn_OC3,dParams.stnkdp.fb2f,upParams.stnkdp.fb2f,moParams.stnkdp.fb2f);
    CBR3_update(fstn_OC1,fstn_OC2,fstn_OC3,fstn_inChannel,dParams.stnkdp.cb3, upParams.stnkdp.cb3, moParams.stnkdp.cb3);
    FB_update(encoderIC1,fstn_inChannel,dParams.featp.cb1, upParams.featp.cb1, moParams.featp.cb1);//CB
    FBR2F_update(FC_OC1,FC_OC2,FC_OC3,OC3,dParams.stn3dp.fb2f,upParams.stn3dp.fb2f,moParams.stn3dp.fb2f);
    CBR3_update(OC1,OC2,OC3,inChannels,dParams.stn3dp.cb3, upParams.stn3dp.cb3, moParams.stn3dp.cb3);
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
    std::string file_path = "./data/train_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    read_h5_file(file_path, list_of_points, list_of_labels);
    int all_num = list_of_points.size();
    all_num = 1000;
    //分配内存，迁移权重到device端
    int ch = 1024;
    int ch_half = 512;
    int ch_quarter = 256;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = ch;
    int FC_OC1 = ch_half;
    int FC_OC2 = ch_quarter;
    int fstn_inChannel = 64;//encoderOC1
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = ch;
    int fstn_FC_OC1 = ch_half;
    int fstn_FC_OC2 = ch_quarter;
    int encoderIC1 = ic;
    int encoderOC2 = 128;
    read_stndP("feat.stn.", dParams.stn3dp,false,ic,OC1,OC2,OC3,FC_OC1,FC_OC2);
    read_stndP("feat.fstn.", dParams.stnkdp,false,fstn_inChannel,fstn_OC1,fstn_OC2,fstn_OC3,fstn_FC_OC1,fstn_FC_OC2);
    read_CB3P("feat.", dParams.featp,false,encoderIC1,fstn_inChannel,encoderOC2);
    read_FB2FP("", dParams.nonep, 0,false,ch,512,256);

    read_stndP("feat.stn.", upParams.stn3dp,true);
    read_stndP("feat.fstn.", upParams.stnkdp,true);
    read_CB3P("feat.", upParams.featp,true);
    read_FB2FP("", upParams.nonep, 0 , true);

    read_stndP("feat.stn.", moParams.stn3dp,true);
    read_stndP("feat.fstn.", moParams.stnkdp,true);
    read_CB3P("feat.", moParams.featp,true);
    read_FB2FP("", moParams.nonep, 0 , true);
    //printVector_GPU(moParams.nonep.f3.weight, 10);f
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
    // auto start = std::chrono::high_resolution_clock::now();
    
    // 分配输出内存：device端
    int* correct_table;
    int *device_labels;
    float *device_output;
    float *device_delta;
    long long max_output_size = cal_tnet_size(batchSize, maxNp, ic); // batchSize > curB maxNp > np
    cudaMalloc((void **)&correct_table, all_num * sizeof(int));
    cudaMalloc((void **)&device_labels, all_num * sizeof(int));
    cudaMalloc((void **)&device_output, max_output_size * sizeof(float));
    cudaMalloc((void **)&device_delta, max_output_size * sizeof(float));
    cudaMemcpy(device_labels, list_of_labels.data(), all_num * sizeof(int), cudaMemcpyHostToDevice);

    // 开始推理
    //init_for_trans_mul(maxNp,ch,batchSize);
    for (size_t e = 0; e < EPOCH; e++)
    {
        printf("epoch: %d\n", e);
        auto start = std::chrono::high_resolution_clock::now();//STRAT
        int correct_num =0;
        int inf_offset = 0;
        for (size_t i = 0; i < all_num; i+=batchSize) {
            size_t curB = std::min(batchSize, all_num - i);
            size_t np = list_of_points[i].size() / ic;
            for (int j = 0; j < curB; j++) {np = std::min(np, list_of_points[i + j].size() / ic);}
            np = ALIGN_DOWN(np, GEMMBLKMAX);
            if (use_sample == 1) np = npoint;
            Train_GPU(ic, curB, np, correct_table + i, device_labels + i, 
            device_all_points + inf_offset, device_output, device_delta);
            inf_offset += curB * np * ic;
            //cudaMemset(device_output, 0, cal_tnet_size(curB, np, ic) * sizeof(float));
            //cudaMemset(device_delta, 0, cal_tnet_size(curB, np, ic) * sizeof(float));
            //memsetDP(moParams);
        }
        //memsetDP(moParams);
        // 计算准确率
        std::vector<int> result(all_num,0);
        cudaMemcpy(result.data(), correct_table, all_num * sizeof(int), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < all_num; i++) {
            correct_num += result[i];
        }
	    float correct_rate = (float)correct_num/all_num; 
        //END
        cudaDeviceSynchronize();// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << std::setprecision(4) << correct_rate;
    }

    // 释放内存
    freeDP(dParams);//权重
    freeDP(upParams, true);//更新权重
    freeDP(moParams, true);
    cudaFree(device_labels);//label
    cudaFree(device_all_points);//输入
    cudaFree(device_output);//输出
    cudaFree(device_delta);//delta
    cudaFree(correct_table);//正确表
    //cudaFree(for_trans_mul);

	// cudaDeviceSynchronize();// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << std::setprecision(4) << correct_rate;
    return 0;
}



