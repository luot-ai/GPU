#ifndef CONV1D_HPP
#define CONV1D_HPP

template <int inChannels,
          int outChannels,
          int kSize>
void Conv1d_CPU(std::vector<float> input, std::vector<float> weights, std::vector<float> bias, std::vector<float> &output ){

}

template <int inChannels,
          int outChannels,
          int kSize>
void Conv1d(float* input, float* kernel, float* bias, float* output ){

}

#endif