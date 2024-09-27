#ifndef LINEAR_HPP
#define LINEAR_HPP

void Linear_CPU(int inFeatures, int outFeatures,std::vector<float> weight,
                std::vector<float> bias,
                std::vector<float> input,
                std::vector<float> &output){
    if (input.size() != inFeatures)
    {
        throw "Linear_cpu input size error";
    }

    if (weight.size() != inFeatures * outFeatures)
    {
        throw "Linear_cpu weight size error";
    }

    if (bias.size() != outFeatures)
    {
        throw "Linear_cpu bias size error";
    }

    for (int oc = 0; oc < outFeatures; oc++)
    {
        output[oc] = bias[oc];
        for (int ic = 0; ic < inFeatures; ic++)
        {
            output[oc] +=
                input[ic] *
                weight[ic + oc * inFeatures];
        }
    }
}

#endif