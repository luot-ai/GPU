#include <string>
#include <vector>
#include <map>

// 通用获取参数的函数
std::vector<float> getParam(const std::map<std::string, std::vector<float>>& params, const std::string& layer, const std::string& param) {
    return params.at(layer + "." + param);
}

// 通用初始化输出的函数
std::vector<float> initializeOutput(int size) {
    return std::vector<float>(size, 0);
}

// 通用的Layer处理函数
void processLayer(const std::string& layerName, int outputChannels, int batchSize, int numPoints, 
                  const std::map<std::string, std::vector<float>>& params,
                  std::vector<float>& convOutput, std::vector<float>& bnOutput, std::vector<float>& reluOutput) {
    
    convOutput = initializeOutput(batchSize * numPoints * outputChannels);
    bnOutput = initializeOutput(batchSize * numPoints * outputChannels);
    reluOutput = initializeOutput(batchSize * numPoints * outputChannels);

    // 获取 conv 参数
    std::vector<float> convWeight = getParam(params, layerName, "weight");
    std::vector<float> convBias = getParam(params, layerName, "bias");

    // 获取 bn 参数
    std::vector<float> bnWeight = getParam(params, layerName, "bn.weight");
    std::vector<float> bnBias = getParam(params, layerName, "bn.bias");
    std::vector<float> bnRunningMean = getParam(params, layerName, "bn.running_mean");
    std::vector<float> bnRunningVar = getParam(params, layerName, "bn.running_var");

    // 这里可以添加具体的卷积、批归一化和ReLU操作逻辑
    // 假设已经有相应的计算函数，可以在此调用
}

int main() {
    // 假设 params 已经填充了所有的参数
    std::map<std::string, std::vector<float>> params;

    int batchSize = 32;
    int numPoints = 1024;

    // Layer 1
    std::vector<float> conv1Output, bn1Output, relu1Output;
    processLayer("feat.stn.conv1", 64, batchSize, numPoints, params, conv1Output, bn1Output, relu1Output);

    // Layer 2
    std::vector<float> conv2Output, bn2Output, relu2Output;
    processLayer("feat.stn.conv2", 128, batchSize, numPoints, params, conv2Output, bn2Output, relu2Output);

    return 0;
}
