// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc P2.cu -o P2 -I./src/submodule -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -lhdf5_cpp -lhdf5

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
__global__ void LogSoftMax_Kernel(float* input,float* output,int L,int BatchSize = 32)
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
        for (int l = 0; l < L; l++)
        {
            int iIdx = l + index * L;
            output[iIdx] = log(input[iIdx] / sum);
        }
    }
}
void LogSoftMax_GPU(float* input,float* output,int L,int BatchSize = 32)
{   
    dim3 blockDim(1);
    dim3 gridDim(BatchSize);
    LogSoftMax_Kernel<<<gridDim,blockDim>>>(input,output,L,BatchSize);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void matrix_add_I_kernel(float *input, int n)
{
    int i = threadIdx.x;
    input[i * n + i] = input[i * n + i] + 1.0f;
}
__global__ void matrix_add_I_kernel_normal(float *input, int n,int batchSize)
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
    // if (n <= 1024)
    // {
    //     dim3 blockDim(n);
    //     dim3 gridDim(batchSize);
    //     matrix_add_I_kernel<<<gridDim, blockDim>>>(input, n,batchSize);
    // }
    // else
    // {
        dim3 blockDim(batchSize);
        dim3 gridDim(n);
        matrix_add_I_kernel_normal<<<gridDim, blockDim>>>(input, n,batchSize);
    cudaDeviceSynchronize();
}


__global__ void Maxpooling_Kernel(float* input,float* output,int numPoints)
{
    __shared__ float sharedMax[1024];
    
    int tx = threadIdx.x;
    int channel = blockIdx.x;

    float localMax = -FLT_MAX;
    int cnum = channel * numPoints;
    for (int i = tx; i < numPoints; i += blockDim.x) {
        float val = input[cnum + i];
        if (val > localMax) {
            localMax = val;
        }
    }
    sharedMax[tx] = localMax;
    __syncthreads();

    // 归约：逐步计算块内的最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
void GPU_MaxPooling(int ics, int batchSize, int numPoints,float* input, float* output)
{
    std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    dim3 blockDim(1024);
    Maxpooling_Kernel<<<gridDim, blockDim>>>(input, output,numPoints);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
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
    std::cout << "--------BMM" << std::endl;

    const int BLK_X = 32;
    const int BLK_Y = 32;

    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((N_B + BLK_X - 1) / BLK_X, (M_A + BLK_Y - 1) / BLK_Y,BatchSize);//X:宽度 Y：高度
    BMM_Kernel<<<gridDim, blockDim>>>(input_A, input_B, output, M_A, K_A, K_B, N_B, BatchSize);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void linear_Kernel(int inFeatures,float* weight,float* bias,float* input,float* output,int outFeatures,int bacthSize)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int curOC = tx + bx * blockDim.x;
    int curB = ty + by * blockDim.y;

    int index = curOC + curB * outFeatures;
    if (curOC < outFeatures && curB < bacthSize)
    {
        output[index] = bias[curOC];
        for (int ic = 0; ic < inFeatures; ic++)
        {
            output[index] +=
                input[curB * inFeatures + ic] *
                weight[curOC * inFeatures + ic];
        }
    }
}
void Linear_GPU(int batchSize,int inFeatures, int outFeatures,float* cudaWeights,float* cudaBias,float* input,float* output){
    std::cout << "------------LAYER:linear" << std::endl;
    dim3 blockDim(32,32);
    dim3 gridDim((outFeatures+31)/32,(batchSize+31)/32);
    linear_Kernel<<<gridDim,blockDim>>>(inFeatures,cudaWeights,cudaBias,input,output,outFeatures,batchSize);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());
    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
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
    std::cout << "------------LAYER:relu" << std::endl;
    // const int BLK_X = 32;
    // const int BLK_Y = 32;

    dim3 blockDim(OC);
    dim3 gridDim(batchSize*numPoints);
    ReLu_Kernel<<<gridDim, blockDim>>>(input, output);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
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

    std::cout << "------------LAYER:batchnorm" << std::endl;
    dim3 blockDim(numFeatures);
    dim3 gridDim(batchSize);
    BatchNorm1d_Kernel<<<gridDim, blockDim>>>(numPoints,cudaWeights,cudaBias,cudaRM,cudaRV,input,output);

    cudaFree(cudaWeights);
    cudaFree(cudaBias);
    cudaFree(cudaRV);
    cudaFree(cudaRM);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void Conv1d_Kernel(int outChannels,int batchSize,int numPoints,int inChannels,float* input, float* weights, float* bias, float* output)
{
    int oc = threadIdx.x;
    int b = blockIdx.x;
    int index = oc + b * blockDim.x;
    //printf("oc %d, batch %d, index %d\n",oc,b, index);
    if(index >= outChannels * batchSize)
        return ;
    for (int n=0;n<numPoints;n++)
    {
        //printf("numpoint: %d\n",n);
        float res = bias[oc];
        //printf("the res of index %d is : %f\n",index*numPoints+n,res);
        for (int ic=0;ic<inChannels;ic++ )
        {
            int ii = b*inChannels*numPoints+ic*numPoints+n;
            int ww = oc*inChannels+ic;
            //printf("input: %d,weight: %d\n",ii,ww);
            res += input[ii]*weights[ww];
            //printf("the res of index %d is : %f\n",index*numPoints+n,res);
        }
        output[index*numPoints+n]=res;
    }
    // cudaFree(w);
}
void Conv1d_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, float* weights, float* bias, float* output ){
    //int L=numPoints;
    std::cout << "------------LAYER:convolution" << std::endl;

    float* cudaWeights;
    float* cudaBias;
    cudaMalloc((void **)&cudaWeights, inChannels * outChannels * sizeof(float));
    cudaMalloc((void **)&cudaBias, outChannels * sizeof(float));
    cudaMemcpy(cudaWeights, weights, inChannels * outChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBias, bias, outChannels * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(outChannels);
    dim3 gridDim(batchSize);
    //std::cout << "WIDTH: " << numPoints << ", IC: " << inChannels << ", OC: " << outChannels << std::endl;
    //std::cout << "isize: " << input.size() << ", wsize: " << weights.size() << ", bsize: " << bias.size() << ", osize: " << output.size() << std::endl;
    Conv1d_Kernel<<<gridDim,blockDim>>>(outChannels,batchSize,numPoints,inChannels,input,cudaWeights,cudaBias,output);
    //printVector_GPU(output,batchSize*numPoints*outChannels);
    cudaFree(cudaWeights);
    cudaFree(cudaBias);
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
    //printVector_GPU(output,batchSize*numPoints*outChannels);
}

//ARCH CB
template<int TILEX,int TILEY>
__global__ void CBWRAP_Kernel(int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
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
    //printf("KERNEL: inchannel %d, outchannel %d, numPoints %d\n",inChannels,outChannels,numPoints);
    // if(oc >= outChannels || np >= numPoints)
    //     return ;
    
    __shared__ float ds_weights[TILEX][TILEY];
    __shared__ float ds_input[TILEX][TILEY];
    // __shared__ float ds_bias[TILEY];
    // __shared__ float ds_bnRM[TILEY];
    // __shared__ float ds_bnRV[TILEY];
    // __shared__ float ds_bnB[TILEY];
    // __shared__ float ds_bnW[TILEY];
    // __shared__ float ds_res[TILEY];

    //phases
    float mean = bnRM[oc];
    float var = bnRV[oc];
    float bnW = bnWeights[oc];
    float bnB = bnBias[oc];
    float res = convBias[oc];
    for (int i = 0; i < inChannels / TILEX; ++i)
    {
        // loading input and weights
        ds_weights[ty][tx] = convWeights[oc * inChannels + i * TILEX + tx];
        if (np < numPoints)
        {
            ds_input[tx][ty] = input[b * numPoints * inChannels + (i * TILEY + ty) * numPoints + np];
        }
        else
        {
            ds_input[tx][ty] = 0;
        }
        __syncthreads();
        // calculate:iterations
        for (int j = 0; j < TILEX; ++j)
        {
            res += ds_weights[ty][j] * ds_input[tx][j];
        }
        __syncthreads();
    }
    res = (res - mean) / sqrt(var + esp) * bnW + bnB;
    if(np < numPoints)
        output[b * numPoints * outChannels + oc * numPoints + np] = res;
}

template<int TILEX,int TILEY>
__global__ void CBWRAP_Kernel_np4tms(int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
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
    //printf("KERNEL: inchannel %d, outchannel %d, numPoints %d\n",inChannels,outChannels,numPoints);
    // if(oc >= outChannels || np >= numPoints)
    //     return ;
    
    __shared__ float ds_weights[TILEX][TILEY];
    __shared__ float ds_input[TILEX][TILEY];
    // __shared__ float ds_bias[TILEY];
    // __shared__ float ds_bnRM[TILEY];
    // __shared__ float ds_bnRV[TILEY];
    // __shared__ float ds_bnB[TILEY];
    // __shared__ float ds_bnW[TILEY];
    // __shared__ float ds_res[TILEY];

    //phases
    float mean = bnRM[oc];
    float var = bnRV[oc];
    float bnW = bnWeights[oc];
    float bnB = bnBias[oc];
    float res = convBias[oc];
    for (int i = 0; i < inChannels / TILEX; ++i)
    {
        // loading input and weights
        ds_weights[ty][tx] = convWeights[oc * inChannels + i * TILEX + tx];
        ds_input[tx][ty] = input[b * numPoints * inChannels + (i * TILEY + ty) * numPoints + np];
        __syncthreads();
        // calculate:iterations
        for (int j = 0; j < TILEX; ++j)
        {
            res += ds_weights[ty][j] * ds_input[tx][j];
        }
        __syncthreads();
    }
    res = (res - mean) / sqrt(var + esp) * bnW + bnB;
    output[b * numPoints * outChannels + oc * numPoints + np] = res;
}


void CBWRAP_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, 
float* cudaConvWeights, float* cudaConvBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    std::cout << "------------LAYER:CBWRAP" << std::endl;
    const int BLK_X = 8;
    const int BLK_Y = 8;
    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((numPoints + BLK_X - 1) / BLK_X, (outChannels + BLK_Y - 1) / BLK_Y, batchSize); // X:宽度 Y：高度

    if (numPoints % BLK_X == 0)
    {
        printf("hey!!\n");
        CBWRAP_Kernel_np4tms<BLK_X, BLK_Y><<<gridDim, blockDim>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else
    {
        CBWRAP_Kernel<BLK_X, BLK_Y><<<gridDim, blockDim>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                           cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }

    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());

    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}


// ARCH CBR
template<int TILEX,int TILEY>
__global__ void CBRWRAP_Kernel(int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
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
    //printf("KERNEL: inchannel %d, outchannel %d, numPoints %d\n",inChannels,outChannels,numPoints);
    // if(oc >= outChannels || np >= numPoints)
    //     return ;
    
    __shared__ float ds_weights[TILEX][TILEY];
    __shared__ float ds_input[TILEX][TILEY];
    // __shared__ float ds_bias[TILEY];
    // __shared__ float ds_bnRM[TILEY];
    // __shared__ float ds_bnRV[TILEY];
    // __shared__ float ds_bnB[TILEY];
    // __shared__ float ds_bnW[TILEY];
    // __shared__ float ds_res[TILEY];

    //phases
    float mean = bnRM[oc];
    float var = bnRV[oc];
    float bnW = bnWeights[oc];
    float bnB = bnBias[oc];
    float res = convBias[oc];
    for (int i = 0; i < inChannels / TILEX; ++i)
    {
        // loading input and weights
        ds_weights[ty][tx] = convWeights[oc * inChannels + i * TILEX + tx];
        if (np < numPoints)
        {
            ds_input[tx][ty] = input[b * numPoints * inChannels + (i * TILEY + ty) * numPoints + np];
        }
        else
        {
            ds_input[tx][ty] = 0;
        }
        __syncthreads();
        // calculate:iterations
        for (int j = 0; j < TILEX; ++j)
        {
            res += ds_weights[ty][j] * ds_input[tx][j];
        }
        __syncthreads();
    }
    res = (res - mean) / sqrt(var + esp) * bnW + bnB;
    if (res < 0)
        res = 0;
    if(np < numPoints)
        output[b * numPoints * outChannels + oc * numPoints + np] = res;
}

template<int TILEX,int TILEY>
__global__ void CBRWRAP_Kernel_np4tms(int outChannels,int batchSize,int numPoints,int inChannels,float* input, 
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
    //printf("KERNEL: inchannel %d, outchannel %d, numPoints %d\n",inChannels,outChannels,numPoints);
    // if(oc >= outChannels || np >= numPoints)
    //     return ;
    
    __shared__ float ds_weights[TILEX][TILEY];
    __shared__ float ds_input[TILEX][TILEY];
    // __shared__ float ds_bias[TILEY];
    // __shared__ float ds_bnRM[TILEY];
    // __shared__ float ds_bnRV[TILEY];
    // __shared__ float ds_bnB[TILEY];
    // __shared__ float ds_bnW[TILEY];
    // __shared__ float ds_res[TILEY];

    //phases
    float mean = bnRM[oc];
    float var = bnRV[oc];
    float bnW = bnWeights[oc];
    float bnB = bnBias[oc];
    float res = convBias[oc];
    for (int i = 0; i < inChannels / TILEX; ++i)
    {
        // loading input and weights
        ds_weights[ty][tx] = convWeights[oc * inChannels + i * TILEX + tx];
        ds_input[tx][ty] = input[b * numPoints * inChannels + (i * TILEY + ty) * numPoints + np];
        __syncthreads();
        // calculate:iterations
        for (int j = 0; j < TILEX; ++j)
        {
            res += ds_weights[ty][j] * ds_input[tx][j];
        }
        __syncthreads();
    }
    res = (res - mean) / sqrt(var + esp) * bnW + bnB;
    if (res < 0)
        res = 0;
    output[b * numPoints * outChannels + oc * numPoints + np] = res;
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
    res = (res - mean) / sqrt(var + esp) * bnW + bnB;
    res = res > 0 ? res : 0;
    output[b * numPoints * outChannels + oc * numPoints + np] = res;
}
void CBRWRAP_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, 
float* cudaConvWeights, float* cudaConvBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    std::cout << "------------LAYER:CBRWRAP" << std::endl;
    // printf("inchannel %d,numPoints %d\n",inChannels,numPoints);

    const int BLK_X = 8;
    const int BLK_Y = 8;
    dim3 blockDim(BLK_X,BLK_Y);
    //dim3 gridDim((numPoints + BLK_X - 1) / BLK_X,(outChannels + BLK_Y - 1) / BLK_Y);//X:宽度 Y：高度
    dim3 gridDim((numPoints + BLK_X - 1) / BLK_X,(outChannels + BLK_Y - 1) / BLK_Y,batchSize);//X:宽度 Y：高度

    //std::cout << "WIDTH: " << numPoints << ", IC: " << inChannels << ", OC: " << outChannels << std::endl;
    //std::cout << "isize: " << input.size() << ", wsize: " << weights.size() << ", bsize: " << bias.size() << ", osize: " << output.size() << std::endl;
    if (inChannels == 3)
    {
        CBRWRAP_Kernel_ic3<<<gridDim, blockDim>>>(BLK_X, BLK_Y, outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else
    {
        //printf("KERNEL: inchannel %d, outchannel %d, numPoints %d\n",inChannels,outChannels,numPoints);
        if (numPoints % BLK_X == 0)
        {
            printf("hey!!\n");
            CBRWRAP_Kernel_np4tms<BLK_X, BLK_Y><<<gridDim, blockDim>>>( outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
        }
        else
        {
            CBRWRAP_Kernel<BLK_X, BLK_Y><<<gridDim, blockDim>>>( outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
        }
    }
    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());
    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPU_CBR(int batchSize, int numPoints, int inics, int OC,wbBnP& wbBnP, float* input, float* reluOutput)
{
    CBRWRAP_GPU(batchSize,numPoints,inics,OC,1,input,wbBnP.weight,wbBnP.bias,
    wbBnP.bn_weight,wbBnP.bn_bias,wbBnP.bn_mean,wbBnP.bn_var,reluOutput);
}
void GPU_CBR_3 (int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,CB3P &cb3p, float* input, float* output) {
    std::cout << "----START CBR_3" << std::endl;
    int bn = batchSize * numPoints;
    float* relu1_output;
    float* relu2_output;
    
    cudaMalloc((void **)&relu1_output, bn*OC1 * sizeof(float));
    cudaMalloc((void **)&relu2_output, bn*OC2 * sizeof(float));

    GPU_CBR(batchSize, numPoints, inics, OC1, cb3p.cb1, input, relu1_output);
    GPU_CBR(batchSize, numPoints, OC1, OC2, cb3p.cb2, relu1_output, relu2_output);
    GPU_CBR(batchSize, numPoints, OC2, OC3, cb3p.cb3, relu2_output, output);
    //printVector_GPU(output, bn * OC3);
    cudaFree(relu1_output);
    cudaFree(relu2_output);
}


// ARCH FBR
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
        res = (res - mean) / sqrt(var + esp) * bnW + bnB;
        res = res > 0 ? res : 0;
        int index = oc + curB * outFeatures;
        output[index] = res;
    }
}
void FBRWRAP_GPU(int batchSize,int inFeatures,int outFeatures,float* input, 
float* cudaFcWeights, float* cudaFcBias, 
float* cudaBnWeights,float* cudaBnBias,float* cudaBnRM,float* cudaBnRV,float* output,float esp = 1e-5
){
    std::cout << "------------LAYER:FBRWRAP" << std::endl;
    // printf("inchannel %d,numPoints %d\n",inChannels,numPoints);
    const int BLK_X = 32;
    const int BLK_Y = 32;
    dim3 blockDim(BLK_X,BLK_Y);
    dim3 gridDim((outFeatures + BLK_X - 1) / BLK_X,(batchSize + BLK_Y - 1) / BLK_Y);//X:宽度 Y：高度
    FBRWRAP_Kernel<<<gridDim,blockDim>>>(BLK_X,BLK_Y,outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,
    output);

    // 检查内核启动是否成功
    CUDA_CHECK(cudaGetLastError());
    // 同步设备并检查执行错误
    CUDA_CHECK(cudaDeviceSynchronize());
}
void GPU_FBR(int batchSize, int inFeatures, 
int outFeatures,wbBnP& fbp, float* input, float* reluOutput)
{
    FBRWRAP_GPU(batchSize,inFeatures,outFeatures,input,
    fbp.weight,fbp.bias,
    fbp.bn_weight,fbp.bn_bias,
    fbp.bn_mean,fbp.bn_var,reluOutput);
}
void GPU_FBR_2_F(int OC1,int OC2,int OC3,int batchSize,int inics,FB2FP &fb2f, float* input, float* output,int param_offset=3)
{
    std::cout << "----START FBR_2_F" << std::endl;
    float* relu1_output;
    float* relu2_output;

    cudaMalloc((void **)&relu1_output, batchSize*OC1 * sizeof(float));
    cudaMalloc((void **)&relu2_output, batchSize*OC2 * sizeof(float));

    GPU_FBR(batchSize,inics,OC1,fb2f.fb1,input,relu1_output);
    GPU_FBR(batchSize,OC1,OC2,fb2f.fb2,relu1_output,relu2_output);
    Linear_GPU(batchSize,OC2, OC3,fb2f.f3.weight, fb2f.f3.bias, relu2_output, output);

    cudaFree(relu1_output);
    cudaFree(relu2_output);
}



std::vector<int> Inference_GPU (int inChannels,
            int batchSize,
            int numPoints,float* input,float* output,
            float* stn3d_out,
            float* stnkd_out,
            const std::vector<float>& C1={},
            const std::vector<float>& C2={},
            const std::vector<float>& C3={},
            const std::vector<float>& C4={},
            bool compare=false) {

    //copyParamsToDevice();
    std::cout << "**********************START INFERENCE************************" << std::endl;
    std::cout << "PART1:STN3d" << std::endl;
    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = 1024;
    int FC_OC1 = 512;
    int FC_OC2 = 256;
    int FC_OC3 = 9;
    float* CBR3_output;
    float* maxp_output;
    cudaMalloc((void **)&CBR3_output, bn * OC3 * sizeof(float));
    cudaMalloc((void **)&maxp_output, batchSize * OC3 * sizeof(float));
    GPU_CBR_3(OC1,OC2,OC3, batchSize, numPoints,inChannels,dParams.stn3dp.cb3, input, CBR3_output);   // conv-bn-relu * 3
    GPU_MaxPooling(OC3, batchSize, numPoints,CBR3_output, maxp_output); // Max pooling    
    GPU_FBR_2_F(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3,dParams.stn3dp.fb2f,maxp_output,stn3d_out);// fc-bn-relu * 2 + fc
    if (compare)
    {
        compareVectors_GPU(C1,CBR3_output,bn*OC3);
        compareVectors_GPU(C2,maxp_output,batchSize*OC3);
        compareVectors_GPU(C3,stn3d_out,batchSize*FC_OC3);
    }
    matrix_add_I(stn3d_out,3,batchSize);
    if(compare)
    {
        compareVectors_GPU(C4,stn3d_out,batchSize*FC_OC3);
        printVector(C3);
        printVector(C4);
    }

    
    cudaFree(CBR3_output);
    cudaFree(maxp_output);

    std::cout << "PART2:TRANS->BMM->TRANS->CBR" << std::endl;
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    float* input_trans;
    float* bmm1_res;
    float* bmm1_res_trans;
    float* fstn_input;
    cudaMalloc((void **)&input_trans, bn * inChannels * sizeof(float));
    cudaMalloc((void **)&bmm1_res, batchSize*numPoints*encoderIC1 * sizeof(float));
    cudaMalloc((void **)&bmm1_res_trans, batchSize*encoderIC1*numPoints * sizeof(float));
    cudaMalloc((void **)&fstn_input, batchSize*fstn_inChannel*numPoints * sizeof(float));
    GPU_transpose(input,input_trans,batchSize,inChannels,numPoints);
    GPU_Bmm(input_trans,stn3d_out,bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    GPU_transpose(bmm1_res,bmm1_res_trans,batchSize,numPoints,encoderIC1);
    GPU_CBR(batchSize,numPoints,encoderIC1,fstn_inChannel,dParams.featp.cb1,bmm1_res_trans,fstn_input);
    cudaFree(input_trans);
    cudaFree(bmm1_res);
    cudaFree(bmm1_res_trans);

    std::cout << "PART3:STNkd"<< std::endl;
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = 1024;
    int fstn_FC_OC1 = 512;
    int fstn_FC_OC2 = 256;
    int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    float* fstn_CBR3_output;
    float* fstn_maxp_output;
    cudaMalloc((void **)&fstn_CBR3_output, bn * fstn_OC3 * sizeof(float));
    cudaMalloc((void **)&fstn_maxp_output, batchSize * fstn_OC3 * sizeof(float));
    GPU_CBR_3(fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,dParams.stnkdp.cb3, fstn_input, fstn_CBR3_output);   // conv-bn-relu * 3
    GPU_MaxPooling(fstn_OC3, batchSize, numPoints,fstn_CBR3_output, fstn_maxp_output); // Max pooling
    GPU_FBR_2_F(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3,dParams.stnkdp.fb2f,fstn_maxp_output,stnkd_out);// fc-bn-relu * 2 + fc
    matrix_add_I(stnkd_out,64,batchSize);
    cudaFree(fstn_CBR3_output);
    cudaFree(fstn_maxp_output);

    std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM" << std::endl;
    int encoderOC2 = 128;
    float* fstn_input_trans;
    float* fstn_bmm1_res;
    float* fstn_bmm1_res_trans; // B C N
    float* cbr2_output;
    cudaMalloc((void **)&fstn_input_trans, bn * fstn_inChannel * sizeof(float));
    cudaMalloc((void **)&fstn_bmm1_res, batchSize*numPoints*fstn_inChannel * sizeof(float));
    cudaMalloc((void **)&fstn_bmm1_res_trans, batchSize*fstn_inChannel*numPoints * sizeof(float));
    cudaMalloc((void **)&cbr2_output, batchSize*encoderOC2*numPoints * sizeof(float));
    GPU_transpose(fstn_input,fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    GPU_Bmm(fstn_input_trans,stnkd_out,fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    GPU_transpose(fstn_bmm1_res,fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    GPU_CBR(batchSize,numPoints,fstn_inChannel,encoderOC2,dParams.featp.cb2,fstn_bmm1_res_trans,cbr2_output);
    //------CB MAX
    int encoderOC3 = 1024;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    //float* feat_conv3;
    float* feat_bn3;
    float* encoder_output;
    //cudaMalloc((void **)&feat_conv3, bnEOC3 * sizeof(float));
    cudaMalloc((void **)&feat_bn3, bnEOC3 * sizeof(float));
    cudaMalloc((void **)&encoder_output, batchSize * encoderOC3 * sizeof(float));
    std::string convStr = "feat.conv3";
    std::string bnStr = "feat.bn3";
    CBWRAP_GPU(batchSize,numPoints,encoderOC2,encoderOC3,1,cbr2_output,
    dParams.featp.cb3.weight, dParams.featp.cb3.bias,
    dParams.featp.cb3.bn_weight, dParams.featp.cb3.bn_bias, 
    dParams.featp.cb3.bn_mean, dParams.featp.cb3.bn_var,feat_bn3);
    GPU_MaxPooling(encoderOC3, batchSize, numPoints,feat_bn3, encoder_output); // Max pooling
    
    cudaFree(fstn_input_trans);
    cudaFree(fstn_bmm1_res);
    cudaFree(fstn_bmm1_res_trans); // B C N
    cudaFree(cbr2_output);
    cudaFree(feat_bn3);

    std::cout << "PART5:CLASSIFY" << std::endl;
    float* softmax_input;
    cudaMalloc((void **)&softmax_input,sizeof(float)*batchSize*10);
    GPU_FBR_2_F(512,256,10,batchSize,encoderOC3,dParams.nonep,encoder_output,softmax_input,0);// fc-bn-relu * 2 + fc
    LogSoftMax_GPU(softmax_input, output, 10 , batchSize);
    std::cout << "----FINAL RESULT" << std::endl;
    std::vector<int> result(batchSize);

    std::vector<float> softmax_output_cpu(batchSize * 10);
    cudaMemcpy(softmax_output_cpu.data(), output, batchSize * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    {
        for (int i = 0; i < batchSize; i++)
        {
            float max_value = softmax_output_cpu[i * 10];
            int max_index = 0;
            for (int j = 1; j < 10; j++)
            {
                if (softmax_output_cpu[i * 10 + j] > max_value)
                {
                    max_value = softmax_output_cpu[i * 10 + j];
                    max_index = j;
                }
            }
            result[i] = max_index;
        }
    }
    cudaFree(softmax_input);
    return result;
}



std::vector<int> Inference_CPU (int inChannels,int batchSize,int numPoints,std::vector<float> input,std::vector<float> &output,std::vector<float> &stn3d_out,std::vector<float> &stnkd_out) {
    std::cout << "**********************START INFERENCE************************" << std::endl;
    std::cout << "PART1:STN3d" << std::endl;
    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = 1024;
    int FC_OC1 = 512;
    int FC_OC2 = 256;
    int FC_OC3 = 9;
    std::vector<float> CBR3_output(bn * OC3);
    std::vector<float> maxp_output(batchSize * OC3);
    //std::vector<float> FBR2F_output(batchSize * FC_OC3);
    CBR_3(OC1,OC2,OC3, batchSize, numPoints,inChannels,"feat.stn.", input, CBR3_output);   // conv-bn-relu * 3
    MaxPooling(OC3, batchSize, numPoints,CBR3_output, maxp_output); // Max pooling    
    FBR_2_F(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3,"feat.stn.",maxp_output,stn3d_out);// fc-bn-relu * 2 + fc
    float I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < FC_OC3; ++j) {
            stn3d_out[i * FC_OC3 + j] += I[j];//batchSize * inic(3) * inic(3)
        }
    }

    std::cout << "PART2:TRANS->BMM->TRANS->CBR" << std::endl;
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    std::vector<float> input_trans(bn * inChannels);
    std::vector<float> bmm1_res(batchSize*numPoints*encoderIC1);
    std::vector<float> bmm1_res_trans(batchSize*encoderIC1*numPoints);
    std::vector<float> fstn_input(batchSize*fstn_inChannel*numPoints);
    transpose(input,input_trans,batchSize,inChannels,numPoints);
    Bmm_cpu(input_trans,stn3d_out,bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    transpose(bmm1_res,bmm1_res_trans,batchSize,numPoints,encoderIC1);
    CBR(1,batchSize,numPoints,encoderIC1,fstn_inChannel,"feat.",bmm1_res_trans,fstn_input);

    std::cout << "PART3:STNkd"<< std::endl;
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = 1024;
    int fstn_FC_OC1 = 512;
    int fstn_FC_OC2 = 256;
    int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    std::vector<float> fstn_CBR3_output(bn * fstn_OC3);
    std::vector<float> fstn_maxp_output(batchSize * fstn_OC3);
    //std::vector<float> fstn_FBR2F_output(batchSize * fstn_FC_OC3);
    CBR_3(fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,"feat.fstn.", fstn_input, fstn_CBR3_output);   // conv-bn-relu * 3
    MaxPooling(fstn_OC3, batchSize, numPoints,fstn_CBR3_output, fstn_maxp_output); // Max pooling
    FBR_2_F(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3,"feat.fstn.",fstn_maxp_output,stnkd_out);// fc-bn-relu * 2 + fc
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < fstn_FC_OC3; ++j) {
            stnkd_out[i * fstn_FC_OC3 + j] += (j % (fstn_inChannel + 1) == 0) ? 1.0f : 0.0f; //batchSize * 64 * 64
        }
    }

    std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM" << std::endl;
    int encoderOC2 = 128;
    std::vector<float> fstn_input_trans(bn * fstn_inChannel);
    std::vector<float> fstn_bmm1_res(batchSize * numPoints * fstn_inChannel);
    std::vector<float> fstn_bmm1_res_trans(batchSize * fstn_inChannel * numPoints); // B C N
    std::vector<float> cbr2_output(batchSize * encoderOC2 * numPoints);
    transpose(fstn_input,fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    Bmm_cpu(fstn_input_trans,stnkd_out,fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    transpose(fstn_bmm1_res,fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    CBR(2,batchSize,numPoints,fstn_inChannel,encoderOC2,"feat.",fstn_bmm1_res_trans,cbr2_output);
    //------CB MAX
    int encoderOC3 = 1024;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    std::vector<float> feat_conv3(bnEOC3);
    std::vector<float> feat_bn3(bnEOC3, 0);
    std::vector<float> encoder_output(batchSize * encoderOC3);
    std::string convStr = "feat.conv3";
    std::string bnStr = "feat.bn3";
    Conv1d_CPU(batchSize,numPoints,encoderOC2, encoderOC3, 1,cbr2_output, params[convStr + ".weight"], params[convStr + ".bias"], feat_conv3);
    BatchNorm1d_CPU(encoderOC3, batchSize, numPoints,params[bnStr + ".weight"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)],feat_conv3,feat_bn3);
    MaxPooling(encoderOC3, batchSize, numPoints,feat_bn3, encoder_output); // Max pooling
    

    std::cout << "PART5:CLASSIFY" << std::endl;
    std::vector<float> softmax_input(batchSize*10);
    FBR_2_F(512,256,10,batchSize,encoderOC3,"",encoder_output,softmax_input,0);// fc-bn-relu * 2 + fc
    LogSoftMax_cpu(softmax_input, output, 10 , batchSize);
    std::cout << "----FINAL RESULT" << std::endl;
    std::vector<int> result(batchSize);
    {
        for (int i = 0; i < batchSize; i++)
        {
            float max_value = output[i * 10];
            int max_index = 0;
            for (int j = 1; j < 10; j++)
            {
                if (output[i * 10 + j] > max_value)
                {
                    max_value = output[i * 10 + j];
                    max_index = j;
                }
            }
            result[i] = max_index;
        }
    }
    return result;
}

// for TEST
void STN3d(float* x, int width, int batch_size, int ic, float* output,float* C1,float* C2,float* C3) { //x:batchsize*ic*N
    float epsilon = 1e-5;//默认的固定值

    // Define dimensions for each layer
    const int conv1_out_ics = 64;
    const int conv2_out_ics = 128;
    const int conv3_out_ics = 1024;
    const int fc1_out_features = 512;
    const int fc2_out_features = 256;

    // Temporary arrays for intermediate outputs
    std::vector<float> conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> conv2_out(batch_size * conv2_out_ics * width );
    std::vector<float> conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> bn_conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> bn_conv2_out(batch_size * conv2_out_ics * width );
    std::vector<float> bn_conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> relu_conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> relu_conv2_out(batch_size * conv2_out_ics * width );
    //std::vector<float> relu_conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> fc1_out(batch_size * fc1_out_features);
    std::vector<float> fc2_out(batch_size * fc2_out_features);
    std::vector<float> bn_fc1_out(batch_size * fc1_out_features);
    std::vector<float> bn_fc2_out(batch_size * fc2_out_features);
    std::vector<float> relu_fc1_out(batch_size * fc1_out_features);
    std::vector<float> relu_fc2_out(batch_size * fc2_out_features);
    //std::vector<float> max_pool_out(batch_size * conv3_out_ics);

    // Convolution layers
    Conv1d(batch_size,ic, conv1_out_ics, 1, width, x, params["feat.stn.conv1.weight"].data(), params["feat.stn.conv1.bias"].data(), conv1_out.data());
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> conv1_out_lt(batch_size * conv1_out_ics * width );
    // Conv1d_CPU(batch_size,width,ic,conv1_out_ics,1,x_vec,params["feat.stn.conv1.weight"],params["feat.stn.conv1.bias"],conv1_out_lt);
    // compareVectors(conv1_out,conv1_out_lt);

    // int iSize = ic*batch_size*width;
    // int oSize = batch_size * conv1_out_ics * width;
    // float* x_vec;
    // float* y_vec;
    // cudaMalloc((void **)&x_vec,iSize*sizeof(float));
    // cudaMalloc((void **)&y_vec,oSize*sizeof(float));
    // cudaMemcpy(x_vec,x,iSize*sizeof(float),cudaMemcpyHostToDevice);
    // Conv1d_GPU(batch_size,width,ic,conv1_out_ics,1,x_vec,params["feat.stn.conv1.weight"].data(),params["feat.stn.conv1.bias"].data(),y_vec);
    // compareVectors_GPU(conv1_out,y_vec,oSize);
    // printVector(conv1_out);
    // printVector_GPU(y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(y_vec);

    batchNorm1d(conv1_out.data(), params["feat.stn.bn1.weight"].data(), params["feat.stn.bn1.bias"].data(), bn_conv1_out.data(), params["feat.stn.bn1.running_mean"].data(),params["feat.stn.bn1.running_var"].data(), batch_size, width,conv1_out_ics, epsilon);
    
    // int iSize = ic*batch_size*width;
    // int oSize = batch_size * conv1_out_ics * width;
    // float* x_vec;
    // float* m_vec;
    // float* y_vec;
    // cudaMalloc((void **)&x_vec,iSize*sizeof(float));
    // cudaMalloc((void **)&m_vec,oSize*sizeof(float));
    // cudaMalloc((void **)&y_vec,oSize*sizeof(float));
    // cudaMemcpy(x_vec,x,iSize*sizeof(float),cudaMemcpyHostToDevice);
    // Conv1d_GPU(batch_size,width,ic,conv1_out_ics,1,x_vec,params["feat.stn.conv1.weight"].data(),params["feat.stn.conv1.bias"].data(),m_vec);
    // BatchNorm1d_GPU(conv1_out_ics, batch_size, width,params["feat.stn.bn1.weight"].data(), params["feat.stn.bn1.bias"].data(), params["feat.stn.bn1.running_mean"].data(), params["feat.stn.bn1.running_var"].data(),m_vec,y_vec);
    // compareVectors_GPU(bn_conv1_out,y_vec,oSize);
    // printVector(conv1_out);
    // printVector_GPU(y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(y_vec);
    // cudaFree(m_vec);
    // cudaFree(m2_vec);
    
    relu(bn_conv1_out.data(), relu_conv1_out.data(), batch_size * conv1_out_ics * width );
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> relu_conv1_out_lt(batch_size * conv1_out_ics * width );
    // CBR(1,batch_size,width,ic,conv1_out_ics,"feat.stn.",x_vec,relu_conv1_out_lt);
    // compareVectors(relu_conv1_out_lt,relu_conv1_out);

    // int iSize = ic*batch_size*width;
    // int oSize = batch_size * conv1_out_ics * width;
    // float* x_vec;
    // float* m_vec;
    // float* m2_vec;
    // float* y_vec;
    // cudaMalloc((void **)&x_vec,iSize*sizeof(float));
    // cudaMalloc((void **)&m_vec,oSize*sizeof(float));
    // cudaMalloc((void **)&m2_vec,oSize*sizeof(float));
    // cudaMalloc((void **)&y_vec,oSize*sizeof(float));
    // cudaMemcpy(x_vec,x,iSize*sizeof(float),cudaMemcpyHostToDevice);
    // Conv1d_GPU(batch_size,width,ic,conv1_out_ics,1,x_vec,params["feat.stn.conv1.weight"].data(),params["feat.stn.conv1.bias"].data(),m_vec);
    // BatchNorm1d_GPU(conv1_out_ics, batch_size, width,params["feat.stn.bn1.weight"].data(), params["feat.stn.bn1.bias"].data(), params["feat.stn.bn1.running_mean"].data(), params["feat.stn.bn1.running_var"].data(),m_vec,m2_vec);
    // ReLU_GPU(batch_size,width,conv1_out_ics,m2_vec,y_vec);
    // compareVectors_GPU(relu_conv1_out,y_vec,oSize);
    // //printVector(conv1_out);
    // printVector_GPU(y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(y_vec);
    // cudaFree(m_vec);
    // cudaFree(m2_vec);

    // int iSize = ic * batch_size * width;
    // int oSize = batch_size * conv1_out_ics * width;
    // float *x_vec;
    // float *y_vec;
    // cudaMalloc((void **)&x_vec, iSize * sizeof(float));
    // cudaMalloc((void **)&y_vec, oSize * sizeof(float));
    // cudaMemcpy(x_vec, x, iSize * sizeof(float), cudaMemcpyHostToDevice);
    // GPU_CBR(1, batch_size, width, ic, conv1_out_ics, "feat.stn.", x_vec, y_vec,conv1_out,bn_conv1_out,relu_conv1_out,true);
    // compareVectors_GPU(relu_conv1_out, y_vec, oSize);
    // printVector_GPU(y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(y_vec);

    Conv1d(batch_size,conv1_out_ics, conv2_out_ics, 1, width , relu_conv1_out.data(), params["feat.stn.conv2.weight"].data(), params["feat.stn.conv2.bias"].data(), conv2_out.data());
    batchNorm1d(conv2_out.data(), params["feat.stn.bn2.weight"].data(), params["feat.stn.bn2.bias"].data(), bn_conv2_out.data(), params["feat.stn.bn2.running_mean"].data(),params["feat.stn.bn2.running_var"].data(), batch_size, width,conv2_out_ics, epsilon);
    relu(bn_conv2_out.data(), relu_conv2_out.data(), batch_size * conv2_out_ics * width );

    Conv1d(batch_size,conv2_out_ics, conv3_out_ics, 1, width , relu_conv2_out.data(), params["feat.stn.conv3.weight"].data(), params["feat.stn.conv3.bias"].data(),conv3_out.data());
    batchNorm1d(conv3_out.data(), params["feat.stn.bn3.weight"].data(), params["feat.stn.bn3.bias"].data(), bn_conv3_out.data(),params["feat.stn.bn3.running_mean"].data(),params["feat.stn.bn3.running_var"].data(), batch_size, width,conv3_out_ics, epsilon);
    relu(bn_conv3_out.data(), C1, batch_size * conv3_out_ics * width );
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> relu_conv3_out_lt(batch_size * conv3_out_ics * width );
    // CBR_3(64,128,1024,batch_size,width,ic,"feat.stn.",x_vec,relu_conv3_out_lt);
    //compareVectors(relu_conv3_out_lt,relu_conv3_out);
    // GPU-CHECK
    // int iSize = ic * batch_size * width;
    // int oSize = batch_size * conv3_out_ics * width;
    // float *x_vec;
    // float *y_vec;
    // cudaMalloc((void **)&x_vec, iSize * sizeof(float));
    // cudaMalloc((void **)&y_vec, oSize * sizeof(float));
    // cudaMemcpy(x_vec, x, iSize * sizeof(float), cudaMemcpyHostToDevice);
    // GPU_CBR_3(64,128,1024,batch_size,width,ic,"feat.stn.",x_vec,y_vec);
    // compareVectors_GPU(relu_conv3_out,y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(y_vec);

    // Max pooling
    max_along_dim(C1, C2, batch_size, conv3_out_ics, width);
    // std::vector<float> max_pool_out_LT(batch_size * conv3_out_ics);
    // MaxPooling(conv3_out_ics,batch_size,width,relu_conv3_out,max_pool_out_LT);
    // compareVectors(max_pool_out,max_pool_out_LT);

    // Fully connected layers
    FullConnect(batch_size, conv3_out_ics, fc1_out_features, C2, params["feat.stn.fc1.weight"].data(), fc1_out.data(), params["feat.stn.fc1.bias"].data());
    batchNorm1d(fc1_out.data(), params["feat.stn.bn4.weight"].data(), params["feat.stn.bn4.bias"].data(), bn_fc1_out.data(), params["feat.stn.bn4.running_mean"].data(),params["feat.stn.bn4.running_var"].data(), batch_size, 1,fc1_out_features, epsilon);
    relu(bn_fc1_out.data(), relu_fc1_out.data(), batch_size * fc1_out_features);

    FullConnect(batch_size, fc1_out_features, fc2_out_features, relu_fc1_out.data(), params["feat.stn.fc2.weight"].data(), fc2_out.data(), params["feat.stn.fc2.bias"].data());
    batchNorm1d(fc2_out.data(), params["feat.stn.bn5.weight"].data(), params["feat.stn.bn5.bias"].data(), bn_fc2_out.data(), params["feat.stn.bn5.running_mean"].data(),params["feat.stn.bn5.running_var"].data(), batch_size, 1,fc2_out_features, epsilon);
    relu(bn_fc2_out.data(), relu_fc2_out.data(), batch_size * fc2_out_features);

    FullConnect(batch_size, fc2_out_features, ic*ic, relu_fc2_out.data(), params["feat.stn.fc3.weight"].data(), C3, params["feat.stn.fc3.bias"].data());
    // std::vector<float> output_lt(batch_size * ic*ic);
    // std::vector<float> outvec(output, output+batch_size*ic*ic);
    // FBR_2_F(fc1_out_features,fc2_out_features,ic*ic,batch_size,conv3_out_ics,"feat.stn.",max_pool_out,output_lt);
    // compareVectors(outvec,output_lt);
    // GPU-CHECK:MAXP -> FBR2-F
    // int iSize = batch_size * conv3_out_ics * width;
    // int mSize = batch_size * conv3_out_ics;
    // int oSize = batch_size * ic*ic;
    // float *x_vec;
    // float *m_vec;
    // float *y_vec;
    // cudaMalloc((void **)&x_vec, iSize * sizeof(float));
    // cudaMalloc((void **)&m_vec, mSize * sizeof(float));
    // cudaMalloc((void **)&y_vec, oSize * sizeof(float));
    // cudaMemcpy(x_vec, relu_conv3_out.data(), iSize * sizeof(float), cudaMemcpyHostToDevice);
    // GPU_MaxPooling(conv3_out_ics,batch_size,width,x_vec,m_vec);
    // compareVectors_GPU(max_pool_out,m_vec,mSize);
    // GPU_FBR_2_F(fc1_out_features,fc2_out_features,ic*ic,batch_size,conv3_out_ics,"feat.stn.",m_vec,y_vec,
    // 3,relu_fc1_out,relu_fc2_out,false);
    // compareVectors_GPU_float(output,y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(m_vec);
    // cudaFree(y_vec);


    // Add identity matrix
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 9; ++j) {
            output[i * 9 + j] = C3[i*9+j]+identity[j];
        }
    }
}
void STNkd(float* x, int width, int batch_size, int ic, float* output) {
    float epsilon = 1e-5;//默认的固定值

    // Define dimensions for each layer
    const int conv1_out_ics = 64;
    const int conv2_out_ics = 128;
    const int conv3_out_ics = 1024;
    const int fc1_out_features = 512;
    const int fc2_out_features = 256;

    // Temporary arrays for intermediate outputs
    std::vector<float> conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> conv2_out(batch_size * conv2_out_ics * width );
    std::vector<float> conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> bn_conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> bn_conv2_out(batch_size * conv2_out_ics * width );
    std::vector<float> bn_conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> relu_conv1_out(batch_size * conv1_out_ics * width);
    std::vector<float> relu_conv2_out(batch_size * conv2_out_ics * width );
    std::vector<float> relu_conv3_out(batch_size * conv3_out_ics * width );
    std::vector<float> fc1_out(batch_size * fc1_out_features);
    std::vector<float> fc2_out(batch_size * fc2_out_features);
    std::vector<float> bn_fc1_out(batch_size * fc1_out_features);
    std::vector<float> bn_fc2_out(batch_size * fc2_out_features);
    std::vector<float> relu_fc1_out(batch_size * fc1_out_features);
    std::vector<float> relu_fc2_out(batch_size * fc2_out_features);
    std::vector<float> max_pool_out(batch_size * conv3_out_ics);
    
    // Convolution layers
    Conv1d(batch_size,ic, conv1_out_ics, 1, width, x, params["feat.fstn.conv1.weight"].data(),  params["feat.fstn.conv1.bias"].data(),conv1_out.data());
    batchNorm1d(conv1_out.data(), params["feat.fstn.bn1.weight"].data(), params["feat.fstn.bn1.bias"].data(), bn_conv1_out.data(), params["feat.fstn.bn1.running_mean"].data(),params["feat.fstn.bn1.running_var"].data(), batch_size, width, conv1_out_ics, epsilon);
    relu(bn_conv1_out.data(), relu_conv1_out.data(), batch_size * conv1_out_ics * width );

    Conv1d(batch_size,conv1_out_ics, conv2_out_ics, 1, width , relu_conv1_out.data(), params["feat.fstn.conv2.weight"].data(), params["feat.fstn.conv2.bias"].data(),conv2_out.data());
    batchNorm1d(conv2_out.data(), params["feat.fstn.bn2.weight"].data(), params["feat.fstn.bn2.bias"].data(), bn_conv2_out.data(), params["feat.fstn.bn2.running_mean"].data(),params["feat.fstn.bn2.running_var"].data(), batch_size, width, conv2_out_ics, epsilon);
    relu(bn_conv2_out.data(), relu_conv2_out.data(), batch_size * conv2_out_ics * width );

    Conv1d(batch_size,conv2_out_ics, conv3_out_ics, 1, width , relu_conv2_out.data(), params["feat.fstn.conv3.weight"].data(), params["feat.fstn.conv3.bias"].data(),conv3_out.data());
    batchNorm1d(conv3_out.data(), params["feat.fstn.bn3.weight"].data(), params["feat.fstn.bn3.bias"].data(), bn_conv3_out.data(),params["feat.fstn.bn3.running_mean"].data(),params["feat.fstn.bn3.running_var"].data(), batch_size, width, conv3_out_ics, epsilon);
    relu(bn_conv3_out.data(), relu_conv3_out.data(), batch_size * conv3_out_ics * width );

    // Max pooling
    max_along_dim(relu_conv3_out.data(), max_pool_out.data(), batch_size, conv3_out_ics,width);

    // Fully connected layers
    FullConnect(batch_size, conv3_out_ics, fc1_out_features, max_pool_out.data(), params["feat.fstn.fc1.weight"].data(), fc1_out.data(), params["feat.fstn.fc1.bias"].data());
    batchNorm1d(fc1_out.data(), params["feat.fstn.bn4.weight"].data(), params["feat.fstn.bn4.bias"].data(), bn_fc1_out.data(), params["feat.fstn.bn4.running_mean"].data(),params["feat.fstn.bn4.running_var"].data(), batch_size, 1,fc1_out_features, epsilon);
    relu(bn_fc1_out.data(), relu_fc1_out.data(), batch_size * fc1_out_features);

    FullConnect(batch_size, fc1_out_features, fc2_out_features, relu_fc1_out.data(), params["feat.fstn.fc2.weight"].data(), fc2_out.data(), params["feat.fstn.fc2.bias"].data());
    batchNorm1d(fc2_out.data(), params["feat.fstn.bn5.weight"].data(), params["feat.fstn.bn5.bias"].data(), bn_fc2_out.data(), params["feat.fstn.bn5.running_mean"].data(),params["feat.fstn.bn5.running_var"].data(), batch_size, 1,fc2_out_features, epsilon);
    relu(bn_fc2_out.data(), relu_fc2_out.data(), batch_size * fc2_out_features);

    FullConnect(batch_size, fc2_out_features, ic*ic, relu_fc2_out.data(), params["feat.fstn.fc3.weight"].data(), output, params["feat.fstn.fc3.bias"].data());

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < ic * ic; ++j) {
            output[i * (ic * ic) + j] += (j % (ic + 1) == 0) ? 1.0f : 0.0f; // 适应 ic 数量
        }
    }
}
void PointNetEncoder(float* x, int batch_size, int ic, int N, float* trans, float* trans_feat, float* final_x,float* C1,float* C2,float* C3) {
    float epsilon = 1e-5;//默认的固定值

    const int conv1_out_ics = 64;
    const int conv2_out_ics = 128;
    const int conv3_out_ics = 1024;
    std::vector<float> conv1_out(batch_size * conv1_out_ics * N);
    std::vector<float> conv2_out(batch_size * conv2_out_ics * N);
    std::vector<float> relu_conv1_out(batch_size * conv1_out_ics * N);
    std::vector<float> relu_conv2_out(batch_size * conv2_out_ics * N);
    std::vector<float> conv3_out(batch_size * conv3_out_ics * N);
    std::vector<float> bn_conv1_out(batch_size * conv1_out_ics * N);
    std::vector<float> bn_conv2_out(batch_size * conv2_out_ics * N);
    std::vector<float> bn_conv3_out(batch_size * conv3_out_ics * N);
    //std::vector<float> max_pool_out(batch_size * conv3_out_ics);
    std::vector<float> x_trans(batch_size * ic * N);
    std::vector<float> x_trans_trans(batch_size * ic * N);
    std::vector<float> x_trans_trans_trans(batch_size * conv1_out_ics * N);
    std::vector<float> x_trans_trans_trans_mul_trans_feat(batch_size * conv1_out_ics * N);
    // Apply STN3d
    STN3d(x, N, batch_size, ic, trans,C1,C2,C3); //trans:batchsize*ic*ic
    
    // Transpose input data: [B, C, N] -> [B, N, C]
    transpose_xtf(x, x_trans.data(), batch_size, ic, N); //x_trans:batchsize*N*ic
    std::vector<float> x_trans_mul_trans(batch_size * ic * N );
    bmm(x_trans.data(), trans, x_trans_mul_trans.data(), batch_size, N, ic, ic); //x_trans_mul_trans:batchsize*N*ic

    // int iSize = ic*batch_size*N;
    // int i1Size = ic*ic*batch_size;
    // int oSize = ic*batch_size*N;
    // float* x_vec;
    // float* x1_vec;
    // float* y_vec;
    // cudaMalloc((void **)&x_vec,iSize*sizeof(float));
    // cudaMalloc((void **)&x1_vec,i1Size*sizeof(float));
    // cudaMalloc((void **)&y_vec,oSize*sizeof(float));
    // cudaMemcpy(x_vec,x_trans.data(),iSize*sizeof(float),cudaMemcpyHostToDevice);
    // cudaMemcpy(x1_vec,trans,i1Size*sizeof(float),cudaMemcpyHostToDevice);
    // GPU_Bmm(x_vec,x1_vec,y_vec,N,ic,ic,ic,batch_size);
    // compareVectors_GPU(x_trans_mul_trans,y_vec,oSize);
    // cudaFree(x_vec);
    // cudaFree(x1_vec);
    // cudaFree(y_vec);

    transpose_xtf(x_trans_mul_trans.data(), x_trans_trans.data(), batch_size, N, ic);//x_trans_trans:batchsize*ic*N

    Conv1d(batch_size,ic, conv1_out_ics, 1, N, x_trans_trans.data(), params["feat.conv1.weight"].data(),params["feat.conv1.bias"].data(), conv1_out.data());
    batchNorm1d(conv1_out.data(), params["feat.bn1.weight"].data(), params["feat.bn1.bias"].data(), bn_conv1_out.data(), params["feat.bn1.running_mean"].data(),params["feat.bn1.running_var"].data(), batch_size, N, conv1_out_ics, epsilon);
    relu(bn_conv1_out.data(), relu_conv1_out.data(), batch_size * conv1_out_ics * N );//conv1_out:batch_size * conv1_out_ics * N

    STNkd(relu_conv1_out.data(), N, batch_size, conv1_out_ics, trans_feat);//trans_feat:batchsize*conv1_out_ics*conv1_out_ics
    transpose_xtf(relu_conv1_out.data(), x_trans_trans_trans.data(), batch_size, conv1_out_ics, N);// x_trans_trans_trans:batchsize*N*conv1_out_ics
    bmm(x_trans_trans_trans.data(), trans_feat, x_trans_trans_trans_mul_trans_feat.data(), batch_size, N, conv1_out_ics, conv1_out_ics); //
    transpose_xtf(x_trans_trans_trans_mul_trans_feat.data(), x_trans_trans_trans.data(), batch_size, N, conv1_out_ics);

    Conv1d(batch_size,conv1_out_ics, conv2_out_ics, 1, N, x_trans_trans_trans.data(), params["feat.conv2.weight"].data(), params["feat.conv2.bias"].data(),conv2_out.data());
    batchNorm1d(conv2_out.data(), params["feat.bn2.weight"].data(), params["feat.bn2.bias"].data(), bn_conv2_out.data(), params["feat.bn2.running_mean"].data(),params["feat.bn2.running_var"].data(), batch_size, N, conv2_out_ics, epsilon);
    relu(bn_conv2_out.data(), relu_conv2_out.data(), batch_size * conv2_out_ics * N );//conv2_out.data():batch_size * conv2_out_ics * N

    Conv1d(batch_size,conv2_out_ics, conv3_out_ics, 1, N, relu_conv2_out.data(), params["feat.conv3.weight"].data(),params["feat.conv3.bias"].data(), conv3_out.data());
    batchNorm1d(conv3_out.data(), params["feat.bn3.weight"].data(), params["feat.bn3.bias"].data(), bn_conv3_out.data(), params["feat.bn3.running_mean"].data(),params["feat.bn3.running_var"].data(), batch_size, N,  conv3_out_ics, epsilon);
   
    max_along_dim(bn_conv3_out.data(), final_x, batch_size, conv3_out_ics,N);//final_x:batchsize*1024(conv3_out_ics)
}
std::vector<int> get_model(float* x, int batch_size, int ic, int N, float* trans, float* trans_feat, float* final_x,float* C1,float* C2,float* C3){
    float epsilon = 1e-5;//默认的固定值
    const int fc1_in_features = 1024;
    const int fc1_out_features = 512;
    const int fc2_out_features = 256;
    const int fc3_out_features = 10;
    std::vector<float> fc1_out(batch_size * fc1_out_features);
    std::vector<float> fc2_out(batch_size * fc2_out_features);
    std::vector<float> bn_fc1_out(batch_size * fc1_out_features);
    std::vector<float> bn_fc2_out(batch_size * fc2_out_features);
    std::vector<float> fc3_out(batch_size * fc3_out_features);
    std::vector<float> fc1_in(batch_size * fc1_in_features);
    std::vector<float> relu_fc1_out(batch_size * fc1_out_features);
    std::vector<float> relu_fc2_out(batch_size * fc2_out_features);
    
    PointNetEncoder(x,batch_size,ic,N, trans, trans_feat, fc1_in.data(),C1,C2,C3);
    
    FullConnect(batch_size, fc1_in_features, fc1_out_features, fc1_in.data(), params["fc1.weight"].data(), fc1_out.data(), params["fc1.bias"].data());
    batchNorm1d(fc1_out.data(), params["bn1.weight"].data(), params["bn1.bias"].data(), bn_fc1_out.data(), params["bn1.running_mean"].data(),params["bn1.running_var"].data(), batch_size, 1, fc1_out_features, epsilon);
    relu(bn_fc1_out.data(), relu_fc1_out.data(), batch_size * fc1_out_features);

    FullConnect(batch_size, fc1_out_features, fc2_out_features, relu_fc1_out.data(), params["fc2.weight"].data(), fc2_out.data(), params["fc2.bias"].data());
    batchNorm1d(fc2_out.data(), params["bn2.weight"].data(), params["bn2.bias"].data(), bn_fc2_out.data(), params["bn2.running_mean"].data(),params["bn2.running_var"].data(), batch_size, 1, fc2_out_features, epsilon);
    relu(bn_fc2_out.data(), relu_fc2_out.data(), batch_size * fc2_out_features);

    FullConnect(batch_size, fc2_out_features, fc3_out_features, relu_fc2_out.data(), params["fc3.weight"].data(), fc3_out.data(), params["fc3.bias"].data());//batchsize*fc3_out_features

    log_softmax(fc3_out.data(), final_x, batch_size, fc3_out_features);

    std::vector<int> result(batch_size);
        for(int b = 0; b <  batch_size; ++b)
        {
            int max_index=0;
            float max=-FLT_MAX;
            for(int index=0;index<10;index++)
            {
                if(final_x[b*10+index]>max)
                {
                    max_index = index;
                    max = final_x[b*10+index];
                }
            }
            result[b]=max_index;
        }
        return result;
}

int main(int argc, char *argv[]) {
    
    // 读取模型参数
    std::string dir = argv[1]; 
    read_params(dir);

    // 读取训练集数据
    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    read_h5_file(file_path, list_of_points, list_of_labels);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    int batchSize = 4;
    int ic = 3;
    int correct_num =0;

    //迁移参数
    read_stndP("feat.stn.", dParams.stn3dp);
    read_stndP("feat.fstn.", dParams.stnkdp);
    read_CB3P("feat.", dParams.featp);
    read_FB2FP("", dParams.nonep, 0);

    // 开始计时，使用chrono计时，不支持其它计时方式
    std::cout << "total :" << list_of_labels.size() << std::endl;
    for (size_t i = 0; i < list_of_points.size(); i+=batchSize) {

        std::cout << "ITERATION: " << i << ": ";

        //当前循环BATCHSIZE
        size_t curB = (batchSize < list_of_points.size() - i) ? batchSize : list_of_points.size() - i;
        
        //当前循环中NUMPOINTS最少的点
        int np = list_of_points[i].size() / ic;
        for (int j = 0; j < curB; j++) 
        {
            np = (list_of_points[i + j].size() / ic < np) ? list_of_points[i + j].size() / ic : np;
        }

        std::cout << "CUTOFF INPUT: " << i << "np is : " << np <<std::endl;;
        std::vector<float> input(curB * np * ic);
        std::vector<float> trans(curB * ic * ic);
        std::vector<float> trans_feat(curB * 64 * 64);
        std::vector<float> final_x(curB * 10);
        for (int b = 0; b < curB; ++b)
        {
            for (int w = 0; w < np; ++w)
            {
                for (int c = 0; c < ic; ++c)
                {
                    input[b * np * ic + w * ic + c] = list_of_points[i + b][w * ic + c];
                }
            }
        }

        //输入输出
        std::vector<float> input_trans(curB * np * ic);
        transpose(input, input_trans, curB, np, ic);
        std::vector<int> result(curB,0);

        //推理与结果
        int transSize = curB * ic * ic;
        int transFeatSize = curB * 64 * 64;
        float *device_input_trans;
        cudaMalloc((void **)&device_input_trans, curB * np * ic * sizeof(float));
        cudaMemcpy(device_input_trans, input_trans.data(), curB * np * ic * sizeof(float), cudaMemcpyHostToDevice);
        float *trans_lt_gpu;
        float *trans_feat_lt_gpu;
        float *netOut_gpu;
        cudaMalloc((void **)&trans_lt_gpu, transSize * sizeof(float));
        cudaMalloc((void **)&trans_feat_lt_gpu, transFeatSize * sizeof(float));
        cudaMalloc((void **)&netOut_gpu, curB * 10 * sizeof(float));
        result = Inference_GPU(ic,curB,np,device_input_trans,netOut_gpu,trans_lt_gpu,trans_feat_lt_gpu);
        //result=Inference_CPU(ic,curB,np, input_trans,final_x,trans,trans_feat);
        for (int b = 0; b < curB; ++b)
        {
            correct_num += (result[b] == list_of_labels[i + b]);
            if (result[b] == list_of_labels[i + b])
                std::cout << (i + b) << std::endl;
        }
        std::cout << "END INFERENCE:: iter :" << i << " correct_num :" << correct_num << " iter_batchsize :" << curB << std::endl;
        std::cout << "total :" << list_of_labels.size() << std::endl;
        printVector<int> (result);
    }
    std::cout << "total :" << list_of_labels.size() << std::endl;
    std::cout << "correct_num :" << correct_num << std::endl;
	float correct_rate = (float)correct_num/(float)list_of_labels.size();
    freeDP(dParams);
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << correct_rate;

    return 0;
}

// int main(int argc, char *argv[]) {
    
//     // 读取模型参数
//     std::string dir = "./params/150epoch";  
//     read_params(dir);

//     // 读取训练集数据：此处无用
//     std::string file_path = "./data/test_point_clouds.h5";
//     std::vector<std::vector<float>> list_of_points;
//     std::vector<int> list_of_labels;
//     read_h5_file(file_path, list_of_points, list_of_labels);

//     //设定参数：ic*b*np
//     int ic = 3;
//     int b = 32;
//     int np = 13;
//     int transSize = b * ic *ic;
//     int transFeatSize = b * 64 * 64; 
//     std::vector<float> input(ic * b * np);
//     //基准
//     std::vector<float> trans(b * ic * ic);
//     std::vector<float> trans_feat(b * 64 * 64);
//     std::vector<float> final_y(b * 10);
//     std::vector<int> result;
//     std::vector<int> result_lt;
//     //待验证:REVISION
//     // std::vector<float> trans_lt(b * ic * ic);
//     // std::vector<float> trans_feat_lt(b * 64 * 64);
//     // std::vector<float> netOut(b * 10);
//     float *trans_lt_gpu;
//     float *trans_feat_lt_gpu;
//     float *netOut_gpu;
//     cudaMalloc((void **)&trans_lt_gpu, transSize * sizeof(float));
//     cudaMalloc((void **)&trans_feat_lt_gpu, transFeatSize * sizeof(float));
//     cudaMalloc((void **)&netOut_gpu, b * 10 * sizeof(float));

//     // 生成输入
//     std::random_device rd;  
//     std::mt19937 eng(rd()); 
//     std::uniform_real_distribution<float> distr(0.0f, 1.0f); 
//     for (auto& value : input) {
//         value = distr(eng); 
//     }
//     std::vector<float> input_trans(b * np  * ic);
//     transpose(input,input_trans,b,np,ic );

//     //推理:REVISION
//     std::vector<float> C1(b * np * 1024);
//     std::vector<float> C2(b * 1024);
//     std::vector<float> C3(b * 9);
//     result=get_model(input_trans.data(), b, ic, np, trans.data(), trans_feat.data(), final_y.data(),C1.data(),C2.data(),C3.data());

//     float* device_input_trans;
//     cudaMalloc((void **)&device_input_trans, b * np  * ic * sizeof(float));
//     cudaMemcpy(device_input_trans, input_trans.data(), b * np  * ic * sizeof(float), cudaMemcpyHostToDevice);
//     result_lt=Inference_GPU(ic, b, np, device_input_trans, netOut_gpu, trans_lt_gpu, trans_feat_lt_gpu,C1,C2,C3,trans,false);
//     //result_lt=Inference_CPU(ic, b, np, input_trans, netOut_gpu, trans_lt, trans_feat_lt);
    

//     //对比结果
//     compareVectors_GPU(trans,trans_lt_gpu,transSize);
//     compareVectors_GPU(trans_feat,trans_feat_lt_gpu,transFeatSize);
//     cudaFree(trans_lt_gpu);
//     cudaFree(trans_feat_lt_gpu);
//     cudaFree(netOut_gpu);
//     //对比结果
//     compareVectors<int>(result_lt,result);
//     for (const auto& value : result) {
//         std::cout << value << " ";
//     }
//     std::cout << std::endl;
//     for (const auto& value : result_lt) {
//         std::cout << value << " ";
//     }
//     std::cout << std::endl;
//     return 0;
// }

