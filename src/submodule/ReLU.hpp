#ifndef RELU_HPP
#define RELU_HPP

void ReLU_CPU(int size,std::vector<float> input,
              std::vector<float> &output){
                
    if (input.size() != size)
    {
        throw "ReLu_cpu input size error";
    }

    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

#endif 