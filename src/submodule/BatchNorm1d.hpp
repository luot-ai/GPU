#ifndef BATCHNORM1D_HPP
#define BATCHNORM1D_HPP

void BatchNorm1d_CPU(int numFeatures, int batchSize, int numPoints,std::vector<float> weight,
                 std::vector<float> bias,
                 std::vector<float> running_mean,
                 std::vector<float> running_var,
                 std::vector<float> input,
                 std::vector<float> &output,
                 float esp = 1e-5)
{
    // Input: (BatchSize, numFeatures, numPoints)
    // Output: (BatchSize, numFeatures, numPoints)
    // 检验输入输出是否合法
    std::cout << "------------LAYER:batchnorm" << std::endl;
    if (input.size() != batchSize * numPoints * numFeatures)
    {
        throw "BatchNorm1d_cpu input size error";
    }

    for(int b=0;b<batchSize;b++)
    {
        for (int ic = 0; ic < numFeatures; ic++)
        {
            float mean = running_mean[ic];
            float var = running_var[ic];
            for (int n = 0; n < numPoints; n++)
            {
                output[b * numFeatures * numPoints + n + ic * numPoints] =
                    (input[b * numFeatures * numPoints + n + ic * numPoints] - mean) / sqrt(var + esp) * weight[ic] + bias[ic];
            }
        }
    }
    //std::cout << "bn done" << std::endl;
}

#endif 