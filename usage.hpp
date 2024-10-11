#ifndef USAGE_HPP
#define USAGE_HPP


#include <random>
#include <iostream>
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
#include "Conv1d.hpp"
#include "BatchNorm1d.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Bmm.hpp"
#include "compare.hpp"
#include "usage.hpp"
#include <cuda_runtime.h>


// 字符串拼接
std::string RM(const std::string& a) 
{return a + ".running_mean";}
std::string RV(const std::string& a) 
{return a + ".running_var";}
std::map<std::string, std::vector<float>> cparams;
// 定义一个宏用于检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " (error code " << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl; // 输出换行
}
void printVector_GPU(float* vec, int size) {
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

template <typename T>
bool compareVectors(const std::vector<T>& vec1, const std::vector<T>& vec2) {


    // 首先比较大小
    if (vec1.size() != vec2.size()) {
        std::cout << "FALSE:SIZE" << std::endl;
        return false;
    }
    
    // 然后比较内容
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec1[i] != vec2[i]) {
            std::cout << "FALSE:ELEMENT" << std::endl;
            return false;
        }
    }
    std::cout << "TRUE" << std::endl;
    return true; // 两个向量相等
}
bool compareVectors_GPU_float(float* vec1, float* vec2,int size) {

    std::vector<float> vec_cpu(size);
    cudaError_t err = cudaMemcpy(vec_cpu.data(), vec2, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    // 然后比较内容
    const float epsilon = 1e-5; // 容差值
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(vec1[i] - vec_cpu[i]) > epsilon) {
            std::cout << "FALSE:ELEMENT, index " << i << "they are: " << vec1[i] << ", " << vec_cpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "TRUE" << std::endl;
    return true; // 两个向量相等
}
bool compareVectors_GPU(const std::vector<float>& vec1, float* vec2,int size) {

    std::vector<float> vec_cpu(size);
    cudaError_t err = cudaMemcpy(vec_cpu.data(), vec2, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    // 首先比较大小
    if (vec1.size() != size) {
        std::cout << "FALSE:SIZE" << std::endl;
        return false;
    }
    
    // 然后比较内容
    const float epsilon = 1e-5; // 容差值
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec_cpu[i]) > epsilon) {
            std::cout << "FALSE:ELEMENT, index " << i << "they are: " << vec1[i] << ", " << vec_cpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "TRUE" << std::endl;
    return true; // 两个向量相等
}

/****************************************************************************************
 * 网络搭建
 ****************************************************************************************/
void LogSoftMax_cpu(std::vector<float> input,std::vector<float> &output,int L,int BatchSize = 32)
{
    // 检验输入输出是否合法
    if (input.size() != L * BatchSize)
    {
        throw "LogSoftMax_cpu input size error";
    }

    for (int iter_batch = 0; iter_batch < BatchSize; iter_batch++)
    {
        // exp
        float sum = 0;
        for (int iter_L = 0; iter_L < L; iter_L++)
        {
            input[iter_L + iter_batch * L] = exp(input[iter_L + iter_batch * L]);
            sum += input[iter_L + iter_batch * L];
        }

        for (int iter_L = 0; iter_L < L; iter_L++)
        {
            output[iter_L + iter_batch * L] = log(input[iter_L + iter_batch * L] / sum);
        }
    }
}

void transpose(std::vector<float> input,std::vector<float> &output,int dim0,int dim1,int dim2)
{
    for (int iter_dim0 = 0; iter_dim0 < dim0; iter_dim0++)
    {
        for (int iter_dim1 = 0; iter_dim1 < dim1; iter_dim1++)
        {
            for (int iter_dim2 = 0; iter_dim2 < dim2; iter_dim2++)
            {
                output[iter_dim0 * dim1 * dim2 + iter_dim2 * dim1 + iter_dim1] =
                    input[iter_dim0 * dim1 * dim2 + iter_dim1 * dim2 + iter_dim2];
            }
        }
    }
}

void FBR(int i, int batchSize, int inFeatures, 
int outFeatures,const std::string &layer, std::vector<float> input, std::vector<float> &reluOutput , int param_offset)
{
    std::cout << "--------FBR" << i << std::endl;
    int bOF = batchSize * outFeatures;
    std::vector<float> fc(bOF);
    std::vector<float> bn(bOF, 0);

    std::string fiStr = std::to_string(i);
    std::string biStr = std::to_string(i+param_offset);
    std::string fcStr = layer + "fc" + fiStr;
    std::string bnStr = layer + "bn" + biStr;
    // std::cout << bnStr  << std::endl;
    Linear_CPU(batchSize,inFeatures, outFeatures,cparams[fcStr + ".weight"], cparams[fcStr + ".bias"], input, fc);
    BatchNorm1d_CPU(outFeatures, batchSize, 1,cparams[bnStr + ".weight"], cparams[bnStr + ".bias"], cparams[RM(bnStr)], cparams[RV(bnStr)], fc, bn);
    ReLU_CPU(bOF,bn, reluOutput);
}

void FBR_2_F(int OC1,int OC2,int OC3,int batchSize,int inics,const std::string& layer, std::vector<float> input, std::vector<float> &output,int param_offset=3)
{
    std::cout << "----START FBR_2_F" << std::endl;
    std::vector<float> relu1_output(batchSize*OC1);
    std::vector<float> relu2_output(batchSize*OC2);

    FBR(1,batchSize,inics,OC1,layer,input,relu1_output,param_offset);
    FBR(2,batchSize,OC1,OC2,layer,relu1_output,relu2_output,param_offset);

    std::string iStr = std::to_string(3);
    std::string fcStr = layer + "fc" + iStr;
    Linear_CPU(batchSize,OC2, OC3,cparams[fcStr + ".weight"], cparams[fcStr + ".bias"], relu2_output, output);
}

void CBR(int i, int batchSize, int numPoints, int inics, int OC,const std::string &layer, std::vector<float> input, std::vector<float> &reluOutput)
{
    std::cout << "--------CBR" << i << std::endl;
    int bnOC = batchSize * numPoints * OC;
    std::vector<float> conv(bnOC);
    std::vector<float> bn(bnOC, 0);

    std::string iStr = std::to_string(i);
    std::string convStr = layer + "conv" + iStr;
    std::string bnStr = layer + "bn" + iStr;
    //std::cout << convStr  << std::endl;

    Conv1d_CPU(batchSize,numPoints,inics, OC, 1,input, cparams[convStr + ".weight"], cparams[convStr + ".bias"], conv);
    BatchNorm1d_CPU(OC, batchSize, numPoints,cparams[bnStr + ".weight"], cparams[bnStr + ".bias"], cparams[RM(bnStr)], cparams[RV(bnStr)],conv,bn);
    ReLU_CPU(bnOC,bn,reluOutput);
    
}

void CBR_3 (int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,const std::string& layer, std::vector<float> input, std::vector<float> &output) {
    std::cout << "----START CBR_3" << std::endl;
    int bn = batchSize * numPoints;
    std::vector<float> relu1_output(bn*OC1);
    std::vector<float> relu2_output(bn*OC2);
    CBR(1,batchSize,numPoints,inics,OC1,layer,input,relu1_output);
    CBR(2,batchSize,numPoints,OC1,OC2,layer,relu1_output,relu2_output);
    CBR(3,batchSize,numPoints,OC2,OC3,layer,relu2_output,output);
    //printVector<int>(output);
}

void MaxPooling(int ics, int batchSize, int numPoints,std::vector<float> input, std::vector<float> &output)
{
    std::cout << "----START MAXPOOLING" << std::endl;
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < ics; c++) {
            float max_val = -FLT_MAX; // Initialize with the smallest possible float value
            // Find the maximum value in the N dimension
            for (int n = 0; n < numPoints; n++) {
                float value = input[b * ics * numPoints + c * numPoints + n];
                if (value > max_val) {
                    max_val = value;
                }
            }
            output[b * ics + c] = max_val; // Store the maximum value for the current batch and ic
        }
    }
}


#endif

