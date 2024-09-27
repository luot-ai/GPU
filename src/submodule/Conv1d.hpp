#ifndef CONV1D_HPP
#define CONV1D_HPP


void Conv1d_CPU(int numPoints,int inChannels,
          int outChannels,
          int kSize,std::vector<float> input, std::vector<float> weights, std::vector<float> bias, std::vector<float> &output ){
    int L=numPoints;
    std::cout << "conv begin" << std::endl;
    std::cout << "WIDTH: " << numPoints << ", IC: " << inChannels << ", OC: " << outChannels << std::endl;
    std::cout << "isize: " << input.size() << ", wsize: " << weights.size() << ", bsize: " << bias.size() << ", osize: " << output.size() << std::endl;
    for (int oc = 0; oc < outChannels; oc++)
    {
        for (int n = 0; n < L; n++)
        {
            output[n + oc * L] = bias[oc];
            //std::cout << "bias" << n  << std::endl;
            for (int ic = 0; ic < inChannels; ic++)
            {
                output[oc * L + n] +=
                    input[ic * L + n] *
                    weights[ic + oc * inChannels];
            }
            //std::cout << "madd" << std::endl;
        }
    }
    std::cout << "conv done" << std::endl;
}

template <int inChannels,
          int outChannels,
          int kSize>
void Conv1d(float* input, float* kernel, float* bias, float* output ){

}

#endif