#ifndef LINEAR_HPP
#define LINEAR_HPP

void Linear_CPU(int batchSize,int inFeatures, int outFeatures,std::vector<float> weight,
                std::vector<float> bias,
                std::vector<float> input,
                std::vector<float> &output){
    std::cout << "------------LAYER:linear" << std::endl;
    //std::cout << "batch: " << batchSize << ", inFeatures: " << inFeatures << ", outFeatures: " << outFeatures << std::endl;
    // std::cout << bias.size()  << std::endl;// out
    // std::cout << weight.size()  << std::endl;// in out
    // std::cout << input.size()  << std::endl;//b in 
    // std::cout << output.size()  << std::endl;//b out 
    for (int b = 0; b < batchSize; b++)
    {
        for (int oc = 0; oc < outFeatures; oc++)
        {   
            int b_of=b*outFeatures;
            int b_if=b*inFeatures;
            output[b_of+oc] = bias[oc];
            for (int ic = 0; ic < inFeatures; ic++)
            {
                output[b_of+oc] +=
                    input[b_if+ic] *
                    weight[oc * inFeatures+ic];
            }
        }
    }
}

#endif