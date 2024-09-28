#ifndef COMPARE_HPP
#define COMPARE_HPP
void transpose_xtf(float* input, float* output,int dim0,int dim1,int dim2)
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

void Conv1d(int batch_size, int in_ics, int out_ics, int kernel_size, int width, 
            float* input_array, float* kernel, float* bias, float* output) {
    // 这里 kernel_size 始终为 1，所以只需处理每个输入的对应通道
    for (int b = 0; b < batch_size; b++) {
        for (int iter_output_ic = 0; iter_output_ic < out_ics; iter_output_ic++)
        {
            for (int iter_L = 0; iter_L < width; iter_L++)
            {
                output[b * width *  out_ics+iter_L + iter_output_ic * width] = bias[iter_output_ic];
                for (int iter_input_ic = 0; iter_input_ic < in_ics; iter_input_ic++)
                {
                    output[b * width *  out_ics + iter_output_ic * width + iter_L] +=
                        input_array[b * width *  in_ics + iter_input_ic * width + iter_L] *
                        kernel[iter_input_ic + iter_output_ic * in_ics];
                }
            }
        }
    }
}

void FullConnect(int input_size, int in_features, int out_features, 
                      const float* input, const float* weight, 
                      float* output, const float* bias) {
    for(int b =0 ; b < input_size ; b++)
    {
        for (int iter_output_ic = 0; iter_output_ic < out_features; iter_output_ic++)
        {
            output[b * out_features + iter_output_ic] = bias[iter_output_ic];
            for (int iter_input_ic = 0; iter_input_ic < in_features; iter_input_ic++)
            {
                output[b * out_features + iter_output_ic] +=
                    input[b * in_features +iter_input_ic] *
                    weight[iter_input_ic + iter_output_ic * in_features];
            }
        }
    }
}

void relu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

void batchNorm1d(float *x, float *weight, float *bias, float *output, 
                  float *mean, float *variance, int batch_size, int width, 
                  int num_features, float epsilon) {
    for(int b=0;b<batch_size;b++)
    {
        for (int iter_input_ic = 0; iter_input_ic < num_features; iter_input_ic++)
        {
            float running_mean = mean[iter_input_ic];
            float running_var = variance[iter_input_ic];
            for (int iter_L = 0; iter_L < width; iter_L++)
            {
                output[b * num_features * width + iter_L + iter_input_ic * width] =
                    (x[b * num_features * width + iter_L + iter_input_ic * width] - running_mean) / sqrt(running_var + epsilon) * weight[iter_input_ic] + bias[iter_input_ic];
            }
        }
    }
}   

void max_along_dim(const float* x, float* result, int batchsize, int ic, int N) {
    // Iterate over each batch and each ic
    for (int b = 0; b < batchsize; b++) {
        for (int c = 0; c < ic; c++) {
            float max_val = -FLT_MAX; // Initialize with the smallest possible float value
            // Find the maximum value in the N dimension
            for (int n = 0; n < N; n++) {
                float value = x[b * ic * N + c * N + n];
                if (value > max_val) {
                    max_val = value;
                }
            }
            result[b * ic + c] = max_val; // Store the maximum value for the current batch and ic
        }
    }
}
void bmm(const float* A, const float* B, float* C, int batch_size, int n, int m, int p) {
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                C[b * n * p + i * p + j] = 0; // 初始化 C[b][i][j]
                for (int k = 0; k < m; ++k) {
                    C[b * n * p + i * p + j] += A[b * n * m + i * m + k] * B[b * m * p + k * p + j];
                }
            }
        }
    }
}
void log_softmax(float* input, float* output, int batchsize, int features) {
    for (int iter_batch = 0; iter_batch < batchsize; iter_batch++)
    {
        // exp
        float sum = 0;
        for (int iter_L = 0; iter_L <features; iter_L++)
        {
            input[iter_L + iter_batch * features] = exp(input[iter_L + iter_batch * features]);
            sum += input[iter_L + iter_batch * features];
        }

        for (int iter_L = 0; iter_L < features; iter_L++)
        {
            output[iter_L + iter_batch * features] = log(input[iter_L + iter_batch * features] / sum);
        }
    }
}

#endif 
