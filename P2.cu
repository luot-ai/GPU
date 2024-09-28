// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc P2.cu -o P2 -I./src/submodule -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

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


// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 字符串拼接
std::string RM(const std::string& a) 
{return a + ".running_mean";}
std::string RV(const std::string& a) 
{return a + ".running_var";}

template <typename T>
void printVector(const std::vector<T>& vec) {
    for (const auto& value : vec) {
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
    Linear_CPU(batchSize,inFeatures, outFeatures,params[fcStr + ".weight"], params[fcStr + ".bias"], input, fc);
    BatchNorm1d_CPU(outFeatures, batchSize, 1,params[bnStr + ".weight"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)], fc, bn);
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
    Linear_CPU(batchSize,OC2, OC3,params[fcStr + ".weight"], params[fcStr + ".bias"], relu2_output, output);
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

    Conv1d_CPU(batchSize,numPoints,inics, OC, 1,input, params[convStr + ".weight"], params[convStr + ".bias"], conv);
    BatchNorm1d_CPU(OC, batchSize, numPoints,params[bnStr + ".weight"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)],conv,bn);
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


std::vector<int> Inference_CPU (int inChannels,
            int batchSize,
            int numPoints,std::vector<float> input,std::vector<float> &output,
            std::vector<float> &stn3d_out,
            std::vector<float> &stnkd_out) {
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


void STN3d(float* x, int width, int batch_size, int ic, float* output) { //x:batchsize*ic*N
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
    Conv1d(batch_size,ic, conv1_out_ics, 1, width, x, params["feat.stn.conv1.weight"].data(), params["feat.stn.conv1.bias"].data(), conv1_out.data());
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> conv1_out_lt(batch_size * conv1_out_ics * width );
    // Conv1d_CPU(batch_size,width,ic,conv1_out_ics,1,x_vec,params["feat.stn.conv1.weight"],params["feat.stn.conv1.bias"],conv1_out_lt);
    // compareVectors(conv1_out,conv1_out_lt);
    batchNorm1d(conv1_out.data(), params["feat.stn.bn1.weight"].data(), params["feat.stn.bn1.bias"].data(), bn_conv1_out.data(), params["feat.stn.bn1.running_mean"].data(),params["feat.stn.bn1.running_var"].data(), batch_size, width,conv1_out_ics, epsilon);
    relu(bn_conv1_out.data(), relu_conv1_out.data(), batch_size * conv1_out_ics * width );
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> relu_conv1_out_lt(batch_size * conv1_out_ics * width );
    // CBR(1,batch_size,width,ic,conv1_out_ics,"feat.stn.",x_vec,relu_conv1_out_lt);
    // compareVectors(relu_conv1_out_lt,relu_conv1_out);

    Conv1d(batch_size,conv1_out_ics, conv2_out_ics, 1, width , relu_conv1_out.data(), params["feat.stn.conv2.weight"].data(), params["feat.stn.conv2.bias"].data(), conv2_out.data());
    batchNorm1d(conv2_out.data(), params["feat.stn.bn2.weight"].data(), params["feat.stn.bn2.bias"].data(), bn_conv2_out.data(), params["feat.stn.bn2.running_mean"].data(),params["feat.stn.bn2.running_var"].data(), batch_size, width,conv2_out_ics, epsilon);
    relu(bn_conv2_out.data(), relu_conv2_out.data(), batch_size * conv2_out_ics * width );

    Conv1d(batch_size,conv2_out_ics, conv3_out_ics, 1, width , relu_conv2_out.data(), params["feat.stn.conv3.weight"].data(), params["feat.stn.conv3.bias"].data(),conv3_out.data());
    batchNorm1d(conv3_out.data(), params["feat.stn.bn3.weight"].data(), params["feat.stn.bn3.bias"].data(), bn_conv3_out.data(),params["feat.stn.bn3.running_mean"].data(),params["feat.stn.bn3.running_var"].data(), batch_size, width,conv3_out_ics, epsilon);
    relu(bn_conv3_out.data(), relu_conv3_out.data(), batch_size * conv3_out_ics * width );
    // std::vector<float> x_vec(x, x + ic*batch_size*width);
    // std::vector<float> relu_conv3_out_lt(batch_size * conv3_out_ics * width );
    // CBR_3(64,128,1024,batch_size,width,ic,"feat.stn.",x_vec,relu_conv3_out_lt);
    // compareVectors(relu_conv3_out_lt,relu_conv3_out);

    // Max pooling
    max_along_dim(relu_conv3_out.data(), max_pool_out.data(), batch_size, conv3_out_ics, width);
    // std::vector<float> max_pool_out_LT(batch_size * conv3_out_ics);
    // MaxPooling(conv3_out_ics,batch_size,width,relu_conv3_out,max_pool_out_LT);
    // compareVectors(max_pool_out,max_pool_out_LT);

    // Fully connected layers
    FullConnect(batch_size, conv3_out_ics, fc1_out_features, max_pool_out.data(), params["feat.stn.fc1.weight"].data(), fc1_out.data(), params["feat.stn.fc1.bias"].data());
    batchNorm1d(fc1_out.data(), params["feat.stn.bn4.weight"].data(), params["feat.stn.bn4.bias"].data(), bn_fc1_out.data(), params["feat.stn.bn4.running_mean"].data(),params["feat.stn.bn4.running_var"].data(), batch_size, 1,fc1_out_features, epsilon);
    relu(bn_fc1_out.data(), relu_fc1_out.data(), batch_size * fc1_out_features);

    FullConnect(batch_size, fc1_out_features, fc2_out_features, relu_fc1_out.data(), params["feat.stn.fc2.weight"].data(), fc2_out.data(), params["feat.stn.fc2.bias"].data());
    batchNorm1d(fc2_out.data(), params["feat.stn.bn5.weight"].data(), params["feat.stn.bn5.bias"].data(), bn_fc2_out.data(), params["feat.stn.bn5.running_mean"].data(),params["feat.stn.bn5.running_var"].data(), batch_size, 1,fc2_out_features, epsilon);
    relu(bn_fc2_out.data(), relu_fc2_out.data(), batch_size * fc2_out_features);

    FullConnect(batch_size, fc2_out_features, ic*ic, relu_fc2_out.data(), params["feat.stn.fc3.weight"].data(), output, params["feat.stn.fc3.bias"].data());
    // std::vector<float> output_lt(batch_size * ic*ic);
    // std::vector<float> outvec(output, output+batch_size*ic*ic);
    // FBR_2_F(fc1_out_features,fc2_out_features,ic*ic,batch_size,conv3_out_ics,"feat.stn.",max_pool_out,output_lt);
    // compareVectors(outvec,output_lt);

    // Add identity matrix
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 9; ++j) {
            output[i * 9 + j] += identity[j];
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

void PointNetEncoder(float* x, int batch_size, int ic, int N, float* trans, float* trans_feat, float* final_x) {
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
    STN3d(x, N, batch_size, ic, trans); //trans:batchsize*ic*ic
    
    // Transpose input data: [B, C, N] -> [B, N, C]
    transpose_xtf(x, x_trans.data(), batch_size, ic, N); //x_trans:batchsize*N*ic
    
    std::vector<float> x_trans_mul_trans(batch_size * ic * N );
    bmm(x_trans.data(), trans, x_trans_mul_trans.data(), batch_size, N, ic, ic); //x_trans_mul_trans:batchsize*N*ic

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

std::vector<int> get_model(float* x, int batch_size, int ic, int N, float* trans, float* trans_feat, float* final_x){
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
    
    PointNetEncoder(x,batch_size,ic,N, trans, trans_feat, fc1_in.data());
    
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
//     int np = 10;
//     std::vector<float> input(ic * b * np);
//     //基准
//     std::vector<float> trans(b * ic * ic);
//     std::vector<float> trans_feat(b * 64 * 64);
//     std::vector<float> final_y(b * 10);
//     std::vector<int> result;
//     //待验证
//     std::vector<float> trans_lt(b * ic * ic);
//     std::vector<float> trans_feat_lt(b * 64 * 64);
//     std::vector<float> final_x(b * 10);
//     std::vector<int> result_lt;

//     // 生成输入
//     std::random_device rd;  
//     std::mt19937 eng(rd()); 
//     std::uniform_real_distribution<float> distr(0.0f, 1.0f); 
//     for (auto& value : input) {
//         value = distr(eng); 
//     }
//     std::vector<float> input_trans(b * np  * ic);
//     transpose(input,input_trans,b,np,ic );

//     //推理
//     result_lt=Inference_CPU(ic, b, np, input_trans, final_x, trans_lt, trans_feat_lt);
//     result=get_model(input_trans.data(), b, ic, np, trans.data(), trans_feat.data(), final_y.data());

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

int main(int argc, char *argv[]) {
    
    // 读取模型参数
    std::string dir = "./params/150epoch"; 
    read_params(dir);

    // 读取训练集数据
    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    read_h5_file(file_path, list_of_points, list_of_labels);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    int batchSize = 32;
    int ic = 3;
    int correct_num =0;

    // 开始计时，使用chrono计时，不支持其它计时方式
    for (size_t i = 0; i < list_of_points[i].size(); i+=batchSize) {

        std::cout << "ITERATION: " << i << ": ";

        //当前循环BATCHSIZE
        size_t curB = (batchSize < list_of_points.size() - i) ? batchSize : list_of_points.size() - i;
        
        //当前循环中NUMPOINTS最少的点
        int np = list_of_points[i].size() / ic;
        for (int j = 0; j < curB; j++) 
        {
            np = (list_of_points[i + j].size() / ic < np) ? list_of_points[i + j].size() / ic : np;
        }

        // 进行截断
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
        result=Inference_CPU(ic,curB,np, input_trans,final_x,trans,trans_feat);
        for (int b = 0; b < curB; ++b)
        {
            correct_num += (result[b] == list_of_labels[i + b]);
            if (result[b] == list_of_labels[i + b])
                std::cout << (i + b) << std::endl;
        }
        std::cout << "iter :" << i << " correct_num :" << correct_num << " iter_batchsize :" << curB << std::endl;
        //printVector<int> (result);
    }
    std::cout << "total :" << list_of_labels.size() << std::endl;
    std::cout << "correct_num :" << correct_num << std::endl;
	float correct_rate = (float)correct_num/(float)list_of_labels.size();

	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << correct_rate;

    return 0;
}