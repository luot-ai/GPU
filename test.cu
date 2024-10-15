// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

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
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
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
    //std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    dim3 blockDim(1024);
    Maxpooling_Kernel<<<gridDim, blockDim>>>(input, output,numPoints);
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
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

    const int BLK_X = 32;
    const int BLK_Y = 32;

    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((N_B + BLK_X - 1) / BLK_X, (M_A + BLK_Y - 1) / BLK_Y,BatchSize);//X:宽度 Y：高度
    BMM_Kernel<<<gridDim, blockDim>>>(input_A, input_B, output, M_A, K_A, K_B, N_B, BatchSize);
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
    //std::cout << "------------LAYER:linear" << std::endl;
    dim3 blockDim(32,32);
    dim3 gridDim((outFeatures+31)/32,(batchSize+31)/32);
    linear_Kernel<<<gridDim,blockDim>>>(inFeatures,cudaWeights,cudaBias,input,output,outFeatures,batchSize);
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
    //std::cout << "------------LAYER:convolution" << std::endl;

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
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
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
    //std::cout << "------------LAYER:CBWRAP" << std::endl;
    const int BLK_X = 8;
    const int BLK_Y = 8;
    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((numPoints + BLK_X - 1) / BLK_X, (outChannels + BLK_Y - 1) / BLK_Y, batchSize); // X:宽度 Y：高度

    if (numPoints % BLK_X == 0)
    {
        //printf("hey!!\n");
        CBWRAP_Kernel_np4tms<BLK_X, BLK_Y><<<gridDim, blockDim>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                                  cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }
    else
    {
        CBWRAP_Kernel<BLK_X, BLK_Y><<<gridDim, blockDim>>>(outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                                           cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
    }

    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());

    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
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
    //std::cout << "------------LAYER:CBRWRAP" << std::endl;
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
            //printf("hey!!\n");
            CBRWRAP_Kernel_np4tms<BLK_X, BLK_Y><<<gridDim, blockDim>>>( outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
        }
        else
        {
            CBRWRAP_Kernel<BLK_X, BLK_Y><<<gridDim, blockDim>>>( outChannels, batchSize, numPoints, inChannels, input, cudaConvWeights, cudaConvBias,
                                              cudaBnWeights, cudaBnBias, cudaBnRM, cudaBnRV, output);
        }
    }
    // // 检查内核启动是否成功
    // CUDA_CHECK(cudaGetLastError());
    // // 同步设备并检查执行错误
    // CUDA_CHECK(cudaDeviceSynchronize());
}
void GPU_CBR(int batchSize, int numPoints, int inics, int OC,wbBnP& wbBnP, float* input, float* reluOutput)
{
    CBRWRAP_GPU(batchSize,numPoints,inics,OC,1,input,wbBnP.weight,wbBnP.bias,
    wbBnP.bn_weight,wbBnP.bn_bias,wbBnP.bn_mean,wbBnP.bn_var,reluOutput);
}
void GPU_CBR_3 (int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,CB3P &cb3p, float* input, float* output) {
    //std::cout << "----START CBR_3" << std::endl;
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
    //std::cout << "------------LAYER:FBRWRAP" << std::endl;
    // printf("inchannel %d,numPoints %d\n",inChannels,numPoints);
    const int BLK_X = 32;
    const int BLK_Y = 32;
    dim3 blockDim(BLK_X,BLK_Y);
    dim3 gridDim((outFeatures + BLK_X - 1) / BLK_X,(batchSize + BLK_Y - 1) / BLK_Y);//X:宽度 Y：高度
    FBRWRAP_Kernel<<<gridDim,blockDim>>>(BLK_X,BLK_Y,outFeatures,batchSize,inFeatures,input,cudaFcWeights,cudaFcBias,cudaBnWeights,cudaBnBias,cudaBnRM,cudaBnRV,
    output);

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
void GPU_FBR_2_F(int OC1,int OC2,int OC3,int batchSize,int inics,FB2FP &fb2f, float* input, float* output,int param_offset=3)
{
    //std::cout << "----START FBR_2_F" << std::endl;
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
    // std::cout << "**********************START INFERENCE************************" << std::endl;
    // std::cout << "PART1:STN3d" << std::endl;
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
        // compareVectors_GPU(C1,CBR3_output,bn*OC3);
        // compareVectors_GPU(C2,maxp_output,batchSize*OC3);
        // compareVectors_GPU(C3,stn3d_out,batchSize*FC_OC3);
    }
    matrix_add_I(stn3d_out,3,batchSize);
    if(compare)
    {
        // compareVectors_GPU(C4,stn3d_out,batchSize*FC_OC3);
        // printVector(C3);
        // printVector(C4);
    }

    
    cudaFree(CBR3_output);
    cudaFree(maxp_output);

    // std::cout << "PART2:TRANS->BMM->TRANS->CBR" << std::endl;
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

    // std::cout << "PART3:STNkd"<< std::endl;
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

    // std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM" << std::endl;
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

    // std::cout << "PART5:CLASSIFY" << std::endl;
    float* softmax_input;
    cudaMalloc((void **)&softmax_input,sizeof(float)*batchSize*10);
    GPU_FBR_2_F(512,256,10,batchSize,encoderOC3,dParams.nonep,encoder_output,softmax_input,0);// fc-bn-relu * 2 + fc
    LogSoftMax_GPU(softmax_input, output, 10 , batchSize);
    // std::cout << "----FINAL RESULT" << std::endl;
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
    //std::cout << "total :" << list_of_labels.size() << std::endl;
    for (size_t i = 0; i < list_of_points.size(); i+=batchSize) {

        //std::cout << "ITERATION: " << i << ": ";

        //当前循环BATCHSIZE
        size_t curB = (batchSize < list_of_points.size() - i) ? batchSize : list_of_points.size() - i;
        
        //当前循环中NUMPOINTS最少的点
        int np = list_of_points[i].size() / ic;
        for (int j = 0; j < curB; j++) 
        {
            np = (list_of_points[i + j].size() / ic < np) ? list_of_points[i + j].size() / ic : np;
        }

        //std::cout << "CUTOFF INPUT: " << i << "np is : " << np <<std::endl;;
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
        float *device_input;
        float *device_input_trans;
        cudaMalloc((void **)&device_input_trans, curB * np * ic * sizeof(float));
        cudaMalloc((void **)&device_input, curB * np * ic * sizeof(float));
        cudaMemcpy(device_input, input.data(), curB * np * ic * sizeof(float), cudaMemcpyHostToDevice);
        GPU_transpose(device_input,device_input_trans,curB, np, ic);
        std::vector<int> result(curB,0);

        //推理与结果
        int transSize = curB * ic * ic;
        int transFeatSize = curB * 64 * 64;
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
        }
        //std::cout << "END INFERENCE:: iter :" << i << " correct_num :" << correct_num << " iter_batchsize :" << curB << std::endl;
        //std::cout << "total :" << list_of_labels.size() << std::endl;
        //printVector<int> (result);
    }
    //std::cout << "total :" << list_of_labels.size() << std::endl;
    //std::cout << "correct_num :" << correct_num << std::endl;
	float correct_rate = (float)correct_num/(float)list_of_labels.size();
    freeDP(dParams);
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << std::setprecision(4) << correct_rate;
    return 0;
}


