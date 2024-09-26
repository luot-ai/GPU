#ifndef BATCHNORM1D_HPP
#define BATCHNORM1D_HPP

template <int numFeatures, int batchSize, int numPoints>
void BatchNorm1d_CPU(std::vector<float> weight,
                 std::vector<float> bias,
                 std::vector<float> running_mean,
                 std::vector<float> running_var,
                 std::vector<float> input,
                 std::vector<float> &output,
                 float esp = 1e-5)
{

}

#endif 