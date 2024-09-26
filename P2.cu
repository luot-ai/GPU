// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -lhdf5 -lhdf5_cpp

#include <iostream>
#include <vector>
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
    std::map<std::string, std::vector<float>> params;

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

/****************************************************************************************
 * 网络搭建
 ****************************************************************************************/
void LogSoftMax_cpu(std::vector<float> input,
                    std::vector<float> &output,
                     int L,
                    int BatchSize = 1)
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

template <int i, int batchSize, int inFeatures, 
int outFeatures>
void FBR(const std::string &layer, std::vector<float> input, std::vector<float> &reluOutput , int param_offset)
{
    int bOF = batchSize * outFeatures;
    std::vector<float> fc(bOF);
    std::vector<float> bn(bOF, 0);

    std::string fiStr = std::to_string(i);
    std::string biStr = std::to_string(i+param_offset);
    std::string fcStr = layer + "fc" + fiStr;
    std::string bnStr = layer + "bn" + biStr;
    Linear_CPU<inFeatures, outFeatures>(params[fcStr + ".weight"], params[fcStr + ".bias"], input, fc);
    BatchNorm1d_CPU<outFeatures, batchSize, 1>(params[bnStr + ".weights"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)], fc, bn);
    ReLU_CPU<bOF>(bn, reluOutput);
}

template <int OC1,int OC2,int OC3,int batchSize,int inChannels>
void FBR_2_F(const std::string& layer, std::vector<float> input, std::vector<float> &output,int param_offset=3)
{
    std::vector<float> relu1_output(batchSize*OC1);
    std::vector<float> relu2_output(batchSize*OC2);

    FBR<1,batchSize,inChannels,OC1>(layer,input,relu1_output,param_offset);
    FBR<2,batchSize,OC1,OC2>(layer,relu1_output,relu2_output,param_offset);

    std::string iStr = std::to_string(3);
    std::string fcStr = layer + "fc" + iStr;
    Linear_CPU<OC2, OC3>(params[fcStr + ".weight"], params[fcStr + ".bias"], relu2_output, output);
}

template <int i, int batchSize, int numPoints, int inChannels, int OC>
void CBR(const std::string &layer, std::vector<float> input, std::vector<float> &reluOutput)
{
    int bnOC = batchSize * numPoints * OC;
    std::vector<float> conv(bnOC);
    std::vector<float> bn(bnOC, 0);

    std::string iStr = std::to_string(i);
    std::string convStr = layer + "conv" + iStr;
    std::string bnStr = layer + "bn" + iStr;

    Conv1d_CPU<inChannels, OC, 1>(input, params[convStr + ".weights"], params[convStr + ".bias"], conv);
    BatchNorm1d_CPU<OC, batchSize, numPoints>(params[bnStr + ".weights"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)],conv,bn);
    ReLU_CPU<bnOC>(bn,reluOutput);
}

template <int OC1,int OC2,int OC3,int batchSize,int numPoints,int inChannels>
void CBR_3 (const std::string& layer, std::vector<float> input, std::vector<float> &output) {

    int bn = batchSize * numPoints;
    std::vector<float> relu1_output(bn*OC1);
    std::vector<float> relu2_output(bn*OC2);
    CBR<1,batchSize,numPoints,inChannels,OC1>(layer,input,relu1_output);
    CBR<2,batchSize,numPoints,OC1,OC2>(layer,relu1_output,relu2_output);
    CBR<3,batchSize,numPoints,OC2,OC3>(layer,relu2_output,output);
    return 0;
}

template <int inChannels, int batchSize, int numPoints>
void STN3d (float* x) {

    conv1d_gpu<inChannels, 64, 1>(d_conv1_weight, d_conv1_bias, d_input_transpose, feat_stn_conv1_output, num_points);
   //(conv1): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
    int stnConv1OutChannels = 64 ;
    float *stnConv1Weights;
    float *stnConv1Bias;
    float *stnConv1Output;

    std::vector<float> conv1_weight = params["feat.stn.conv1.weight"];
    std::vector<float> conv1_bias = params["feat.stn.conv1.bias"];

    cudaMalloc((void **)&stnConv1Weights, stnConv1OutChannels * inChannels * sizeof(float));
    cudaMalloc((void **)&stnConv1Bias, stnConv1OutChannels * sizeof(float));
    cudaMalloc((void **)&stnConv1Output, batchSize * numPoints * stnConv1OutChannels * sizeof(float));


    cudaMemcpy(d_conv1_weight, conv1_weight.data(), 64 * InputChannel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_bias.data(), 64 * sizeof(float), cudaMemcpyHostToDevice);
    conv1d_gpu<InputChannel, 64, 1>(d_conv1_weight, d_conv1_bias, d_input_transpose, feat_stn_conv1_output, num_points);
    cudaDeviceSynchronize();
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_input_transpose);
    print_vector_gpu(feat_stn_conv1_output, "feat_stn_conv1_output");
}

template <int channels, int batchSize, int numPoints>
void MaxPooling(std::vector<float> input, std::vector<float> &output)
{

}

template <int inChannels,
            int batchSize,
            int numPoints>
std::vector<int> Inference_CPU (float* input,std::vector<float> &output) {
    //--Encoder
    //------STN3d
    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = 1024;
    int FC_OC1 = 512;
    int FC_OC2 = 256;
    int FC_OC3 = 9;
    std::vector<float> CBR3_output(bn * OC3);
    std::vector<float> maxp_output(batchSize * OC3);
    std::vector<float> FBR2F_output(batchSize * FC_OC3);//batchSize * inchannel(3) * 3
    std::vector<float> STN_trans(batchSize * FC_OC3);
    CBR_3<OC1,OC2,OC3, batchSize, numPoints,inChannels>("feat.stn.", input, CBR3_output);   // conv-bn-relu * 3
    MaxPooling<OC3, batchSize, numPoints>(CBR3_output, maxp_output); // Max pooling
    FBR_2_F<FC_OC1,FC_OC2,FC_OC3,batchSize,OC3>("feat.stn.",maxp_output,FBR2F_output);// fc-bn-relu * 2 + fc
    // FBR2F_output + I
    std::vector<float> I(batchSize * FC_OC3, 0);
    for (int i = 0; i < batchSize; i++)
    {
        I[i * FC_OC3] = 1;
        I[i * FC_OC3 + 4] = 1;
        I[i * FC_OC3 + 8] = 1;
        for (int j = 0; j < FC_OC3; j++)
        {
            int idx = i * FC_OC3 + j;
            STN_trans[idx] = FBR2F_output[idx] + I[idx];
        }
    }
    //------TRANS->BMM->TRANS->CBR
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    std::vector<float> input_trans(bn * inChannels);
    std::vector<float> bmm1_res(batchSize*numPoints*encoderIC1);
    std::vector<float> bmm1_res_trans(batchSize*encoderIC1*numPoints);
    std::vector<float> fstn_input(batchSize*fstn_inChannel*numPoints);
    transpose(input,input_trans,batchSize,inChannels,numPoints);
    Bmm_cpu(input_trans,STN_trans,bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    transpose(bmm1_res,bmm1_res_trans,batchSize,numPoints,encoderIC1);
    CBR<1,batchSize,numPoints,encoderIC1,fstn_inChannel>("feat.",bmm1_res_trans,fstn_input);
    //------STNkd
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = 1024;
    int fstn_FC_OC1 = 512;
    int fstn_FC_OC2 = 256;
    int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    std::vector<float> fstn_CBR3_output(bn * fstn_OC3);
    std::vector<float> fstn_maxp_output(batchSize * fstn_OC3);
    std::vector<float> fstn_FBR2F_output(batchSize * fstn_FC_OC3);//batchSize * inchannel(3) * 3
    std::vector<float> FSTN_trans(batchSize * fstn_FC_OC3);
    CBR_3<fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel>("feat.fstn.", fstn_input, fstn_CBR3_output);   // conv-bn-relu * 3
    MaxPooling<fstn_OC3, batchSize, numPoints>(fstn_CBR3_output, fstn_maxp_output); // Max pooling
    FBR_2_F<fstn_OC1,fstn_OC2,fstn_OC3,batchSize,fstn_OC3>("feat.fstn.",fstn_maxp_output,fstn_FBR2F_output);// fc-bn-relu * 2 + fc
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < fstn_FC_OC3; ++j) {
            FSTN_trans[i * fstn_FC_OC3 + j] += (j % (fstn_inChannel + 1) == 0) ? 1.0f : 0.0f; // 适应 channel 数量
        }
    }
    //------TRANS->BMM->TRANS->CBR
    int encoderOC2 = 128;
    std::vector<float> fstn_input_trans(bn * fstn_inChannel);
    std::vector<float> fstn_bmm1_res(batchSize * numPoints * fstn_inChannel);
    std::vector<float> fstn_bmm1_res_trans(batchSize * fstn_inChannel * numPoints); // B C N
    std::vector<float> cbr2_output(batchSize * encoderOC2 * numPoints);
    transpose(fstn_input,fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    Bmm_cpu(fstn_input_trans,FSTN_trans,fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    transpose(fstn_bmm1_res,fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    CBR<2,batchSize,numPoints,fstn_inChannel,encoderOC2>("feat.",fstn_bmm1_res_trans,cbr2_output);
    //------CB MAX
    int encoderOC3 = 1024;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    std::vector<float> feat_conv3(bnEOC3);
    std::vector<float> feat_bn3(bnEOC3, 0);
    std::vector<float> encoder_output(batchSize * encoderOC3);
    std::string convStr = "feat.conv3";
    std::string bnStr = "feat.bn3";
    Conv1d_CPU<encoderOC2, encoderOC3, 1>(cbr2_output, params[convStr + ".weights"], params[convStr + ".bias"], feat_conv3);
    BatchNorm1d_CPU<encoderOC3, batchSize, numPoints>(params[bnStr + ".weights"], params[bnStr + ".bias"], params[RM(bnStr)], params[RV(bnStr)],feat_conv3,feat_bn3);
    MaxPooling<encoderOC3, batchSize, numPoints>(feat_bn3, encoder_output); // Max pooling
    
    //--CLASSIFY
    std::vector<float> softmax_input(batchSize*10);
    FBR_2_F<512,256,10,batchSize,encoderOC3>(".",encoder_output,softmax_input,0);// fc-bn-relu * 2 + fc
    std::vector<float> softmax_output(batchSize * 10);
    LogSoftMax_CPU(softmax_input, softmax_output, 10 , batchSize);
    
    std::vector<int> result(batchSize);
    {
        for (int i = 0; i < batchSize; i++)
        {
            float max_value = softmax_output[i * 10];
            int max_index = 0;
            for (int j = 1; j < 10; j++)
            {
                if (softmax_output[i * 10 + j] > max_value)
                {
                    max_value = softmax_output[i * 10 + j];
                    max_index = j;
                }
            }
            result[i] = max_index;
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    
    std::string dir = "./params/150epoch";  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集点云数据和标签
	// cout << dir;
	
    // 读取模型参数
    read_params(dir);

    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;
    // 读取训练集数据
    read_h5_file(file_path, list_of_points, list_of_labels);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    int batchSize = 32;
    int channel = 3;
    int correct_num =0;

    // 开始计时，使用chrono计时，不支持其它计时方式
    for (size_t i = 0; i < list_of_points.size(); i+=batchSize) {
        // TODO ...在这里实现利用CUDA对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
		// 打印每一帧的数据，仅用于调试！
    
        // std::cout << "Points " << i << ": ";
        // // for (const auto& point : list_of_points[i]) {
        // //     std::cout << point << " ";
        // // }
        // std::cout << "\nLabel: " << list_of_labels[i] << std::endl;

        size_t curL = (batchSize < list_of_points.size() - i) ? batchSize : list_of_points.size() - i;
        // 该batch中最小的点的大小（N最小）
        int min_width = list_of_points[i].size() / channel;
        for (int j = 0; j < curL; j++) // 遍历一个batch里的点
        {
            min_width = (list_of_points[i + j].size() / channel < min_width) ? list_of_points[i + j].size() / channel : min_width;
        }
        // 进行截断

        std::vector<float> input(curL * min_width * channel);
        std::vector<float> trans(curL * channel * channel);
        std::vector<float> trans_feat(curL * 64 * 64);
        std::vector<float> final_x(curL * 10);

        for (int b = 0; b < curL; ++b)
        {
            for (int w = 0; w < min_width; ++w)
            {
                for (int c = 0; c < channel; ++c)
                {
                    input[b * min_width * channel + w * channel + c] = list_of_points[i + b][w * channel + c];
                }
            }
        }
        std::vector<float> input_trans(curL * min_width * channel);
        transpose(input.data(), input_trans.data(), curL, min_width, channel);

        get_model(input_trans.data(), curL, channel, min_width, trans.data(), trans_feat.data(), final_x.data());

        for (int b = 0; b < curL; ++b)
        {
            int max_index = 0;
            float max = -FLT_MAX;
            for (int index = 0; index < 10; index++)
            {
                if (final_x[b * 10 + index] > max)
                {
                    max_index = index;
                    max = final_x[b * 10 + index];
                }
            }
            correct_num += (max_index == list_of_labels[i + b]);
            if (max_index == list_of_labels[i + b])
                printf("%d\n", i + b);
        }
    }
	
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":0.0001";

    return 0;
}