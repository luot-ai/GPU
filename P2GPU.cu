
__global__ void Maxpooling_Kernel(float* input,float* output,int numPoints)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx + bx * blockDim.x;

    if (idx < gridDim.x)
    {
        float max_val = -FLT_MAX;
        for (int n = 0; n < numPoints; n++)
        {
            float value = input[idx*numPoints + n];
            if (value > max_val)
            {
                max_val = value;
            }
        }
        output[idx] = max_val;
    }
}
void GPU_MaxPooling(int ics, int batchSize, int numPoints,float* input, float* output)
{
    std::cout << "----START MAXPOOLING" << std::endl;
    dim3 gridDim(ics*batchSize);
    dim3 blockDim(ics);
    Maxpooling_Kernel<<<gridDim, blockDim>>>(input, output,numPoints);
}

__global__ void BMM_Kernel(float* input,float* output,int numPoints)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx + bx * blockDim.x;

    for (int iter_batch = 0; iter_batch < BatchSize; iter_batch++)
    {
        for (int iter_M_A = 0; iter_M_A < M_A; iter_M_A++)
        {
            for (int iter_N_B = 0; iter_N_B < N_B; iter_N_B++)
            {
                output[iter_batch * M_A * N_B + iter_M_A * N_B + iter_N_B] = 0;
                for (int iter_K_A = 0; iter_K_A < K_A; iter_K_A++)
                {
                    output[iter_batch * M_A * N_B + iter_M_A * N_B + iter_N_B] +=
                        input_A[iter_batch * M_A * K_A + iter_M_A * K_A + iter_K_A] *
                        input_B[iter_batch * K_B * N_B + iter_K_A * N_B + iter_N_B];
                }
            }
        }
    }
}
void GPU_Bmm(float* input_A,float* input_B,float* output,int M_A,int K_A,int K_B,int N_B,int BatchSize = 1)
{
    std::cout << "--------BMM" << std::endl;

    const int BLK_X = 32;
    const int BLK_Y = 32;

    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim(BatchSize * M_A * N_B);
    BMM_Kernel<<<gridDim, blockDim>>>(input, output, X,Y);
}
void GPU_transpose(float* input,float* output,int dim0,int dim1,int dim2)
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
void Linear_GPU(int batchSize,int inFeatures, int outFeatures,float* weight,float* bias,float* input,float* output){
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

__global__ void ReLu_Kernel(float *input,float *output,int X,int Y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int idx = tx + bx * blockDim.x;
    int idy = ty + by * blockDim.y;
    int index = idx + idy * X;

    if (idx < X && idy < Y)
    {
        output[index] = input[index] > 0 ? input[index] : 0;
    }
}
void ReLU_GPU(int X,int Y,float* input,float* output){
    std::cout << "------------LAYER:relu" << std::endl;
    const int BLK_X = 32;
    const int BLK_Y = 32;

    dim3 blockDim(BLK_X, BLK_Y);
    dim3 gridDim((X + BLK_X - 1) / BLK_X, (Y + BLK_Y - 1) / BLK_Y);
    ReLu_Kernel<<<gridDim, blockDim>>>(input, output, X,Y);
}


void BatchNorm1d_GPU(int numFeatures, int batchSize, int numPoints,float* weight,float* bias,float* running_mean,float* running_var,float* input,float* output,float esp = 1e-5)
{
    std::cout << "------------LAYER:batchnorm" << std::endl;
}
void Conv1d_GPU(int batchSize,int numPoints,int inChannels,int outChannels,int kSize,float* input, float* weights, float* bias, float* output ){
    //int L=numPoints;
    std::cout << "------------LAYER:convolution" << std::endl;
    // std::cout << "WIDTH: " << numPoints << ", IC: " << inChannels << ", OC: " << outChannels << std::endl;
    // std::cout << "isize: " << input.size() << ", wsize: " << weights.size() << ", bsize: " << bias.size() << ", osize: " << output.size() << std::endl;
    
}
void LogSoftMax_GPU(float* input,float* output,int L,int BatchSize = 32)
{
    // 检验输入输出是否合法
    // if (input.size() != L * BatchSize)
    // {
    //     throw "LogSoftMax_GPU input size error";
    // }

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
void transpose_GPU(float* input,float* output,int dim0,int dim1,int dim2)
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

void GPU_FBR(int i, int batchSize, int inFeatures, 
int outFeatures,const std::string &layer, float* input, float* reluOutput , int param_offset)
{
    std::cout << "--------FBR" << i << std::endl;
    int bOF = batchSize * outFeatures;
    float* fc;
    float* bn;
    cudaMalloc((void **)&fc, bOF);
    cudaMalloc((void **)&bn, bOF);

    std::string fiStr = std::to_string(i);
    std::string biStr = std::to_string(i+param_offset);
    std::string fcStr = layer + "fc" + fiStr;
    std::string bnStr = layer + "bn" + biStr;
    // std::cout << bnStr  << std::endl;
    Linear_GPU(batchSize,inFeatures, outFeatures,params[fcStr + ".weight"].data(), params[fcStr + ".bias"].data(), input, fc);
    BatchNorm1d_GPU(outFeatures, batchSize, 1,params[bnStr + ".weight"].data(), params[bnStr + ".bias"].data(), params[RM(bnStr)].data(), params[RV(bnStr)].data(), fc, bn);
    ReLU_GPU(batchSize,outFeatures,bn, reluOutput);

    cudaFree(fc);
    cudaFree(bn);
}

void GPU_FBR_2_F(int OC1,int OC2,int OC3,int batchSize,int inics,const std::string& layer, float* input, float* output,int param_offset=3)
{
    std::cout << "----START FBR_2_F" << std::endl;
    float* relu1_output;
    float* relu2_output;

    cudaMalloc((void **)&relu1_output, batchSize*OC1);
    cudaMalloc((void **)&relu2_output, batchSize*OC2);

    GPU_FBR(1,batchSize,inics,OC1,layer,input,relu1_output,param_offset);
    GPU_FBR(2,batchSize,OC1,OC2,layer,relu1_output,relu2_output,param_offset);

    std::string iStr = std::to_string(3);
    std::string fcStr = layer + "fc" + iStr;
    Linear_GPU(batchSize,OC2, OC3,params[fcStr + ".weight"].data(), params[fcStr + ".bias"].data(), relu2_output, output);

    cudaFree(relu1_output);
    cudaFree(relu2_output);
}

void GPU_CBR(int i, int batchSize, int numPoints, int inics, int OC,const std::string &layer, float* input, float* reluOutput)
{
    std::cout << "--------CBR" << i << std::endl;
    int bnOC = batchSize * numPoints * OC;
    float* conv;
    float* bn;

    cudaMalloc((void **)&conv, bnOC);
    cudaMalloc((void **)&bn, bnOC);

    std::string iStr = std::to_string(i);
    std::string convStr = layer + "conv" + iStr;
    std::string bnStr = layer + "bn" + iStr;
    //std::cout << convStr  << std::endl;

    Conv1d_GPU(batchSize,numPoints,inics, OC, 1,input, params[convStr + ".weight"].data(), params[convStr + ".bias"].data(), conv);
    BatchNorm1d_GPU(OC, batchSize, numPoints,params[bnStr + ".weight"].data(), params[bnStr + ".bias"].data(), params[RM(bnStr)].data(), params[RV(bnStr)].data(),conv,bn);
    ReLU_GPU(batchSize*numPoints,OC,bn,reluOutput);
    cudaFree(conv);
    cudaFree(bn);
    
}

void GPU_CBR_3 (int OC1,int OC2,int OC3,int batchSize,int numPoints,int inics,const std::string& layer, float* input, float* output) {
    std::cout << "----START CBR_3" << std::endl;
    int bn = batchSize * numPoints;
    float* relu1_output;
    float* relu2_output;
    
    cudaMalloc((void **)&relu1_output, bn*OC1 * sizeof(float));
    cudaMalloc((void **)&relu2_output, bn*OC2 * sizeof(float));

    GPU_CBR(1,batchSize,numPoints,inics,OC1,layer,input,relu1_output);
    GPU_CBR(2,batchSize,numPoints,OC1,OC2,layer,relu1_output,relu2_output);
    GPU_CBR(3,batchSize,numPoints,OC2,OC3,layer,relu2_output,output);
    //printVector<int>(output);
    cudaFree(relu1_output);
    cudaFree(relu2_output);
}

std::vector<int> Inference_GPU (int inChannels,
            int batchSize,
            int numPoints,float* input,float* output,
            float* stn3d_out,
            float* stnkd_out) {
    std::cout << "**********************START INFERENCE************************" << std::endl;
    std::cout << "PART1:STN3d" << std::endl;
    int bn = batchSize * numPoints;
    int OC1 = 64;
    int OC2 = 128;
    int OC3 = 1024;
    int FC_OC1 = 512;
    int FC_OC2 = 256;
    int FC_OC3 = 9;
    float* CBR3_output;
    float* maxp_output;
    cudaMalloc((void **)&CBR3_output, bn * OC3 * sizeof(float));
    cudaMalloc((void **)&maxp_output, batchSize * OC3 * sizeof(float));
    GPU_CBR_3(OC1,OC2,OC3, batchSize, numPoints,inChannels,"feat.stn.", input, CBR3_output);   // conv-bn-relu * 3
    GPU_MaxPooling(OC3, batchSize, numPoints,CBR3_output, maxp_output); // Max pooling    
    GPU_FBR_2_F(FC_OC1,FC_OC2,FC_OC3,batchSize,OC3,"feat.stn.",maxp_output,stn3d_out);// fc-bn-relu * 2 + fc
    float I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < FC_OC3; ++j) {
            stn3d_out[i * FC_OC3 + j] += I[j];//batchSize * inic(3) * inic(3)
        }
    }
    cudaFree(CBR3_output);
    cudaFree(maxp_output);

    std::cout << "PART2:TRANS->BMM->TRANS->CBR" << std::endl;
    int encoderIC1 = inChannels;
    int fstn_inChannel = 64;//encoderOC1
    float* input_trans;
    float* bmm1_res;
    float* bmm1_res_trans;
    float* fstn_input;
    cudaMalloc((void **)&input_trans, bn * inChannels * sizeof(float));
    cudaMalloc((void **)&bmm1_res, batchSize*numPoints*encoderIC1 * sizeof(float));
    cudaMalloc((void **)&bmm1_res_trans, batchSize*encoderIC1*numPoints * sizeof(float));
    cudaMalloc((void **)&fstn_input, batchSize*fstn_inChannel*numPoints * sizeof(float));
    GPU_transpose(input,input_trans,batchSize,inChannels,numPoints);
    GPU_Bmm(input_trans,stn3d_out,bmm1_res,numPoints,inChannels,inChannels,encoderIC1,batchSize);
    GPU_transpose(bmm1_res,bmm1_res_trans,batchSize,numPoints,encoderIC1);
    GPU_CBR(1,batchSize,numPoints,encoderIC1,fstn_inChannel,"feat.",bmm1_res_trans,fstn_input);
    cudaFree(input_trans);
    cudaFree(bmm1_res);
    cudaFree(bmm1_res_trans);

    std::cout << "PART3:STNkd"<< std::endl;
    int fstn_OC1 = 64;
    int fstn_OC2 = 128;
    int fstn_OC3 = 1024;
    int fstn_FC_OC1 = 512;
    int fstn_FC_OC2 = 256;
    int fstn_FC_OC3 = fstn_inChannel * fstn_inChannel ;
    float* fstn_CBR3_output;
    float* fstn_maxp_output;
    cudaMalloc((void **)&fstn_CBR3_output, bn * fstn_OC3 * sizeof(float));
    cudaMalloc((void **)&fstn_maxp_output, batchSize * fstn_OC3 * sizeof(float));
    GPU_CBR_3(fstn_OC1,fstn_OC2,fstn_OC3, batchSize, numPoints,fstn_inChannel,"feat.fstn.", fstn_input, fstn_CBR3_output);   // conv-bn-relu * 3
    GPU_MaxPooling(fstn_OC3, batchSize, numPoints,fstn_CBR3_output, fstn_maxp_output); // Max pooling
    GPU_FBR_2_F(fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3,batchSize,fstn_OC3,"feat.fstn.",fstn_maxp_output,stnkd_out);// fc-bn-relu * 2 + fc
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < fstn_FC_OC3; ++j) {
            stnkd_out[i * fstn_FC_OC3 + j] += (j % (fstn_inChannel + 1) == 0) ? 1.0f : 0.0f; //batchSize * 64 * 64
        }
    }
    cudaFree(fstn_CBR3_output);
    cudaFree(fstn_maxp_output);

    std::cout << "PART4:TRANS->BMM->TRANS->CBR->CBM" << std::endl;
    int encoderOC2 = 128;
    float* fstn_input_trans;
    float* fstn_bmm1_res;
    float* fstn_bmm1_res_trans; // B C N
    float* cbr2_output;
    cudaMalloc((void **)&fstn_input_trans, bn * fstn_inChannel * sizeof(float));
    cudaMalloc((void **)&fstn_bmm1_res, batchSize*numPoints*fstn_inChannel * sizeof(float));
    cudaMalloc((void **)&fstn_bmm1_res_trans, batchSize*fstn_inChannel*numPoints * sizeof(float));
    cudaMalloc((void **)&cbr2_output, batchSize*encoderOC2*numPoints * sizeof(float));
    GPU_transpose(fstn_input,fstn_input_trans,batchSize,fstn_inChannel,numPoints);
    GPU_Bmm(fstn_input_trans,stnkd_out,fstn_bmm1_res,numPoints,fstn_inChannel,fstn_inChannel,fstn_inChannel,batchSize);
    GPU_transpose(fstn_bmm1_res,fstn_bmm1_res_trans,batchSize,numPoints,fstn_inChannel);
    GPU_CBR(2,batchSize,numPoints,fstn_inChannel,encoderOC2,"feat.",fstn_bmm1_res_trans,cbr2_output);
    //------CB MAX
    int encoderOC3 = 1024;
    int bnEOC3 = batchSize * numPoints * encoderOC3;
    float* feat_conv3;
    float* feat_bn3;
    float* encoder_output;
    cudaMalloc((void **)&feat_conv3, bnEOC3 * sizeof(float));
    cudaMalloc((void **)&feat_bn3, bnEOC3 * sizeof(float));
    cudaMalloc((void **)&encoder_output, batchSize * encoderOC3 * sizeof(float));
    std::string convStr = "feat.conv3";
    std::string bnStr = "feat.bn3";
    Conv1d_GPU(batchSize,numPoints,encoderOC2, encoderOC3, 1,cbr2_output, params[convStr + ".weight"].data(), params[convStr + ".bias"].data(), feat_conv3);
    BatchNorm1d_GPU(encoderOC3, batchSize, numPoints,params[bnStr + ".weight"].data(), params[bnStr + ".bias"].data(), params[RM(bnStr)].data(), params[RV(bnStr)].data(),feat_conv3,feat_bn3);
    GPU_MaxPooling(encoderOC3, batchSize, numPoints,feat_bn3, encoder_output); // Max pooling
    cudaFree(fstn_input_trans);
    cudaFree(fstn_bmm1_res);
    cudaFree(fstn_bmm1_res_trans); // B C N
    cudaFree(cbr2_output);
    cudaFree(feat_conv3);
    cudaFree(feat_bn3);

    std::cout << "PART5:CLASSIFY" << std::endl;
    float* softmax_input;
    cudaMalloc((void **)&softmax_input,batchSize*10);
    GPU_FBR_2_F(512,256,10,batchSize,encoderOC3,"",encoder_output,softmax_input,0);// fc-bn-relu * 2 + fc
    LogSoftMax_GPU(softmax_input, output, 10 , batchSize);
    std::cout << "----FINAL RESULT" << std::endl;
    std::vector<int> result(batchSize);
    {
        for (int i = 0; i < batchSize; i++)
        {
            float max_value = output[i * 10];
            int max_index = 0;
            for (int j = 1; j < 10; j++)
            {
                if (output[i * 10 + j] > max_value)
                {
                    max_value = output[i * 10 + j];
                    max_index = j;
                }
            }
            result[i] = max_index;
        }
    }
    cudaFree(softmax_input);
    return result;
}

