#include <string>
#include <vector>
#include <map>

// 字符串拼接
std::string RM(const std::string& a) 
{return a + ".running_mean";}
std::string RV(const std::string& a) 
{return a + ".running_var";}



// 通用的Layer处理函数
void CBR(const std::string& layer, int OC, int batchSize, int numPoints, 
                  std::vector<float> input, std::vector<float>& reluOutput) {
    
    int bnOC=batchSize* numPoints * OC;
    std::vector<float> conv(bnOC);
    std::vector<float> bn(bnOC,0);
    std::vector<float> relu(bnOC);

    std::string iStr = std::to_string(i);
    std::string convStr = layer + "conv" + iStr;
    std::string bnStr = layer + "bn" + iStr;

    // conv1(num_features,64,1)->bn1->relu
    Conv1d_CPU<inChannels, OC, 1>(params[convStr+"weight"], params[convStr+"bias"], x, conv, numPoints);
    BatchNorm1d_CPU<OC>(params[bnStr+"weight"], params[bnStr+"bias"], params[RM(bnStr)], params[RV(bnStr)], conv, bn, 1, numPoints);
    ReLu_CPU<OC>(bn, relu, numPoints);
}

template <int n,int batchSize,int numPoints>
void CBR_3 (const std::string& layer, std::vector<float> input) {

    int bn = batchSize * numPoints;
    int OC[3] = {64,128,256};
    std::vector<float> relu1_output(bn*OC[0]);
    std::vector<float> relu2_output(bn*OC[1]);
    std::vector<float> relu3_output(bn*OC[2]);

    CBR()
    

    // 定义层名和对应输出通道数的映射
    // std::vector<std::pair<std::string, int>> layers = {
    //     {"feat.stn.conv1", 64},
    //     {"feat.stn.conv2", 128},
    //     {"feat.stn.conv3", 256},
    //     // 根据n层继续添加...
    // };
    // for (const auto& layer : layers) {
    //     std::vector<float> convOutput, bnOutput, reluOutput;
    //     sCBR(layer.first, layer.second, batchSize, numPoints, params, convOutput, bnOutput, reluOutput);
    // }
    return 0;
}
