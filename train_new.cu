#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mma.h>
#include <random>
#include <string>
#include <vector>
void printVector_GPU(float* vec, int size) {
    printf("size:%d\n",size);
    // 在主机端创建一个标准向量以存储从设备复制的数据
    std::vector<float> vec_cpu(size);

    // 从设备复制数据到主机
    cudaError_t err = cudaMemcpy(vec_cpu.data(), vec, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 打印向量内容
    for (const auto& value : vec_cpu) {
        std::cout << value << " ";
    }
    std::cout << std::endl; // 输出换行

}
// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

using namespace H5;

void read_h5_file(const std::string &file_path,
                  std::vector<std::vector<float>> &list_of_points,
                  std::vector<int> &list_of_labels) {
  try {
    H5File file(file_path, H5F_ACC_RDONLY);
    hsize_t num_objs = file.getNumObjs();
    for (hsize_t i = 0; i < num_objs; i++) {
      std::string name = file.getObjnameByIdx(i);
      Group group = file.openGroup(name);
      DataSet dataset = group.openDataSet("points");
      DataSpace dataspace = dataset.getSpace();
      hsize_t dims[2];
      dataspace.getSimpleExtentDims(dims, NULL);
      std::vector<float> points(dims[0] * dims[1]);
      dataset.read(points.data(), PredType::NATIVE_FLOAT);
      list_of_points.push_back(points);
      Attribute label_attr = group.openAttribute("label");
      int label;
      label_attr.read(PredType::NATIVE_INT, &label);
      list_of_labels.push_back(label);
    }
  } catch (FileIException &error) {
    error.printErrorStack();
    exit(EXIT_FAILURE);
  } catch (DataSetIException &error) {
    error.printErrorStack();
    exit(EXIT_FAILURE);
  } catch (DataSpaceIException &error) {
    error.printErrorStack();
    exit(EXIT_FAILURE);
  } catch (DataTypeIException &error) {
    error.printErrorStack();
    exit(EXIT_FAILURE);
  }
}

void write_param(const std::vector<float> &data, const std::string &filepath) {
  std::ofstream file(filepath);
  if (file.is_open()) {
    for (const auto &value : data) {
      file << value << std::endl;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filepath << std::endl;
  }
}

void save_model_params_and_buffers_to_txt(
    const std::map<std::string, std::vector<float>> &params,
    const std::string &dir) {
  for (const auto &pair : params) {
    const std::string &name = pair.first;
    const std::vector<float> &values = pair.second;
    std::string fileName = dir + "/" + name + ".txt";
    write_param(values, fileName);
  }
}

///////////////////////// Kernel Functions ///////////////////////////

__global__ void convert_to_channel_first_kernel(float *n_first_points,
                                                float *channel_first_points,
                                                int num_points,
                                                int batch_size) {
  // [batch, n, channel] -> [batch, channel, n]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < num_points * batch_size; i += stride) {
    int batch_idx = i / num_points;
    int point_idx = i % num_points;

    int base_idx = batch_idx * num_points * 3 + point_idx * 3;
    int output_base_idx = batch_idx * 3 * num_points;

    channel_first_points[output_base_idx + 0 * num_points + point_idx] =
        n_first_points[base_idx]; // x
    channel_first_points[output_base_idx + 1 * num_points + point_idx] =
        n_first_points[base_idx + 1]; // y
    channel_first_points[output_base_idx + 2 * num_points + point_idx] =
        n_first_points[base_idx + 2]; // z
  }
}

__global__ void add_identity_matrix_kernel(float *input, float *output, int k,
                                           int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * k * k) {
    int b = idx / (k * k);
    int idx_in_matrix = idx % (k * k);
    int row = idx_in_matrix / k;
    int col = idx_in_matrix % k;
    float value = input[b * k * k + row * k + col];

    if (row == col) {
      value += 1.0f;
    }
    output[b * k * k + row * k + col] = value;
  }
}

__global__ void conv1d_kernel(const float *input, float *output,
                              const float *weights, const float *bias,
                              int in_channels, int out_channels, int batch_size,
                              int num_points) {
  int out_c = blockIdx.x;
  int sample_idx = blockIdx.y;
  int point_idx = threadIdx.x;

  if (out_c >= out_channels || sample_idx >= batch_size ||
      point_idx >= num_points) {
    return;
  }

  float sum = 0.0f;
  for (int in_c = 0; in_c < in_channels; ++in_c) {
    sum += input[sample_idx * in_channels * num_points + in_c * num_points +
                 point_idx] *
           weights[out_c * in_channels + in_c];
  }
  output[sample_idx * out_channels * num_points + out_c * num_points +
         point_idx] = sum + bias[out_c];
}

__global__ void conv1d_backward_kernel(const float *input, const float *weights,
                                       const float *output_grad,
                                       float *input_grad, float *weights_grad,
                                       float *bias_grad, int in_channels,
                                       int out_channels, int batch_size,
                                       int num_points) {
  int sample_idx = blockIdx.x;
  int point_idx = threadIdx.x;
  int in_c = blockIdx.y;

  if (sample_idx < batch_size && in_c < in_channels && point_idx < num_points) {
    float input_val = input[sample_idx * in_channels * num_points +
                            in_c * num_points + point_idx];
    float grad_input = 0.0f;

    for (int out_c = 0; out_c < out_channels; ++out_c) {
      float grad_output = output_grad[sample_idx * out_channels * num_points +
                                      out_c * num_points + point_idx];
      float weight = weights[out_c * in_channels + in_c];

      grad_input += weight * grad_output;

      atomicAdd(&weights_grad[out_c * in_channels + in_c],
                input_val * grad_output);
      atomicAdd(&bias_grad[out_c], grad_output);
    }
    input_grad[sample_idx * in_channels * num_points + in_c * num_points +
               point_idx] = grad_input;
  }
}

__global__ void batchnorm_kernel(const float *input, float *output,
                                 const float *mean_cache,
                                 const float *variance_cache,
                                 const float *gamma, const float *beta,
                                 float epsilon, int channels, int batch_size,
                                 int num_points) {
  int channel = blockIdx.x;
  int batch = blockIdx.y;
  int point_idx = threadIdx.x + blockDim.x * blockIdx.z;

  if (point_idx < num_points) {
    int index =
        batch * channels * num_points + channel * num_points + point_idx;

    float mean = mean_cache[channel];
    float variance = variance_cache[channel];
    float normalized = (input[index] - mean) / sqrtf(variance + epsilon);
    output[index] = gamma[channel] * normalized + beta[channel];
  }
}

__global__ void
batchnorm_backward_kernel(const float *input, const float *output_grad,
                          const float *mean, const float *variance,
                          const float *gamma, float *input_grad,
                          float *gamma_grad, float *beta_grad, float epsilon,
                          int channels, int batch_size, int num_points) {
  int c = blockIdx.x;
  if (c >= channels)
    return;

  int N = batch_size * num_points;
  float mu = mean[c];
  float var = variance[c];
  float std_inv = rsqrtf(var + epsilon);

  extern __shared__ float shared_mem[];
  float *shared_sum_dy = shared_mem;
  float *shared_sum_dy_xhat = &shared_mem[blockDim.x];

  float sum_dy = 0.0f;
  float sum_dy_xhat = 0.0f;

  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    int sample_idx = idx / num_points;
    int point_idx = idx % num_points;
    int data_idx =
        sample_idx * channels * num_points + c * num_points + point_idx;

    float x = input[data_idx];
    float dy = output_grad[data_idx];
    float x_hat = (x - mu) * std_inv;

    atomicAdd(&gamma_grad[c], dy * x_hat);
    atomicAdd(&beta_grad[c], dy);

    sum_dy += dy;
    sum_dy_xhat += dy * x_hat;
  }

  // Store per-thread sums in shared memory
  shared_sum_dy[threadIdx.x] = sum_dy;
  shared_sum_dy_xhat[threadIdx.x] = sum_dy_xhat;
  __syncthreads();

  // Reduce sums over all threads
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum_dy[threadIdx.x] += shared_sum_dy[threadIdx.x + stride];
      shared_sum_dy_xhat[threadIdx.x] +=
          shared_sum_dy_xhat[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Broadcast the reduced sums to all threads
  if (threadIdx.x == 0) {
    shared_sum_dy[0] = shared_sum_dy[0];
    shared_sum_dy_xhat[0] = shared_sum_dy_xhat[0];
  }
  __syncthreads();

  sum_dy = shared_sum_dy[0];
  sum_dy_xhat = shared_sum_dy_xhat[0];

  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
    int sample_idx = idx / num_points;
    int point_idx = idx % num_points;
    int data_idx =
        sample_idx * channels * num_points + c * num_points + point_idx;

    float x = input[data_idx];
    float dy = output_grad[data_idx];
    float x_hat = (x - mu) * std_inv;

    float dx = gamma[c] * std_inv * (dy - sum_dy / N - x_hat * sum_dy_xhat / N);
    input_grad[data_idx] = dx;
  }
}

__global__ void compute_mean_kernel(const float *input, float *mean,
                                    int batch_size, int channels,
                                    int num_points) {
  int c = blockIdx.x;
  int sample_idx = blockIdx.y;
  int idx = threadIdx.x;

  extern __shared__ float shared_sum[];

  float sum = 0.0f;
  for (int p = idx; p < num_points; p += blockDim.x) {
    sum += input[sample_idx * channels * num_points + c * num_points + p];
  }

  shared_sum[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&mean[c], shared_sum[0] / (batch_size * num_points));
  }
}

__global__ void compute_variance_kernel(const float *input, const float *mean,
                                        float *variance, int batch_size,
                                        int channels, int num_points) {
  int c = blockIdx.x;
  int sample_idx = blockIdx.y;
  int idx = threadIdx.x;

  extern __shared__ float shared_sum_sq[];

  float mean_c = mean[c];
  float diff_sq = 0.0f;
  for (int p = idx; p < num_points; p += blockDim.x) {
    float val = input[sample_idx * channels * num_points + c * num_points + p];
    diff_sq += (val - mean_c) * (val - mean_c);
  }

  shared_sum_sq[threadIdx.x] = diff_sq;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&variance[c], shared_sum_sq[0] / (batch_size * num_points));
  }
}

__global__ void update_running_mean_variance_kernel(
    float *running_mean, float *running_variance, const float *mean,
    const float *variance, float momentum, int channels) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels) {
    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean[c];
    running_variance[c] =
        momentum * running_variance[c] + (1.0f - momentum) * variance[c];
  }
}

__global__ void fclayer_kernel(const float *input, float *output,
                               const float *weights, const float *bias,
                               int in_features, int out_features,
                               int batch_size) {
  int out_f = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = blockIdx.y;

  if (out_f < out_features && sample_idx < batch_size) {
    float sum = 0.0f;
    for (int in_f = 0; in_f < in_features; ++in_f) {
      sum += input[sample_idx * in_features + in_f] *
             weights[out_f * in_features + in_f];
    }
    output[sample_idx * out_features + out_f] = sum + bias[out_f];
  }
}

__global__ void fclayer_backward_kernel(const float *input,
                                        const float *weights,
                                        const float *output_grad,
                                        float *input_grad, float *weights_grad,
                                        float *bias_grad, int in_features,
                                        int out_features, int batch_size) {
  int out_f = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = blockIdx.y;

  if (out_f < out_features && sample_idx < batch_size) {
    float grad_output = output_grad[sample_idx * out_features + out_f];
    atomicAdd(&bias_grad[out_f], grad_output);

    for (int in_f = 0; in_f < in_features; ++in_f) {
      float input_val = input[sample_idx * in_features + in_f];
      atomicAdd(&weights_grad[out_f * in_features + in_f],
                input_val * grad_output);
      atomicAdd(&input_grad[sample_idx * in_features + in_f],
                weights[out_f * in_features + in_f] * grad_output);
    }
  }
}

__global__ void relu_kernel(const float *input, float *output, int batch_size,
                            int channel, int num_point) {
  int total_size = batch_size * channel * num_point;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void relu_backward_kernel(const float *input,
                                     const float *output_grad,
                                     float *input_grad, int batch_size,
                                     int channel, int num_point) {
  int total_size = batch_size * channel * num_point;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    input_grad[idx] = output_grad[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
  }
}

__global__ void max_pool_kernel(const float *input, float *output,
                                int num_points, int channels) {
  int sample_idx = blockIdx.x;
  int channel_idx = blockIdx.y;

  extern __shared__ float shared_max[];
  int thread_idx = threadIdx.x;

  shared_max[thread_idx] = -FLT_MAX;
  for (int p = thread_idx; p < num_points; p += blockDim.x) {
    float val = input[sample_idx * channels * num_points +
                      channel_idx * num_points + p];
    if (val > shared_max[thread_idx]) {
      shared_max[thread_idx] = val;
    }
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      shared_max[thread_idx] =
          fmaxf(shared_max[thread_idx], shared_max[thread_idx + stride]);
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    output[sample_idx * channels + channel_idx] = shared_max[0];
  }
}

__global__ void max_pool_backward_kernel(const float *input,
                                         const float *output_grad,
                                         float *input_grad, int num_points,
                                         int channels) {
  int sample_idx = blockIdx.x;
  int channel_idx = blockIdx.y;

  extern __shared__ float shared_mem[];
  float *shared_max_val = shared_mem;
  int *shared_max_p = (int *)&shared_mem[blockDim.x];

  int tid = threadIdx.x;
  shared_max_val[tid] = -FLT_MAX;
  shared_max_p[tid] = -1;

  for (int p = tid; p < num_points; p += blockDim.x) {
    int idx = sample_idx * channels * num_points + channel_idx * num_points + p;
    input_grad[idx] = 0.0f;
  }
  __syncthreads();

  for (int p = tid; p < num_points; p += blockDim.x) {
    float val = input[sample_idx * channels * num_points +
                      channel_idx * num_points + p];
    if (val > shared_max_val[tid]) {
      shared_max_val[tid] = val;
      shared_max_p[tid] = p;
    }
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (shared_max_val[tid] < shared_max_val[tid + stride]) {
        shared_max_val[tid] = shared_max_val[tid + stride];
        shared_max_p[tid] = shared_max_p[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int max_p = shared_max_p[0];
    if (max_p != -1) {
      input_grad[sample_idx * channels * num_points + channel_idx * num_points +
                 max_p] = output_grad[sample_idx * channels + channel_idx];
    }
  }
}

__global__ void
batch_matrix_multiply_kernel(const float *inputA, // [batch, channel, num_point]
                             const float *inputB, // [batch, channel, channel]
                             float *output,       // [batch, channel, num_point]
                             int batch_size, int channel, int num_points) {
  int batch_id = blockIdx.x;
  int channel_id = blockIdx.y;
  int point_id = blockIdx.z * blockDim.x + threadIdx.x;

  if (batch_id >= batch_size || channel_id >= channel || point_id >= num_points)
    return;

  const float *A = inputA + batch_id * channel * num_points;
  const float *B = inputB + batch_id * channel * channel;
  float *out = output + batch_id * channel * num_points;

  float result = 0.0f;
  for (int k = 0; k < channel; ++k) {
    float a_val = A[k * num_points + point_id];
    float b_val = B[k * channel + channel_id];
    result += a_val * b_val;
  }

  out[channel_id * num_points + point_id] = result;
}

__global__ void batch_matrix_multiply_backward_inputA_kernel(
    const float *output_grad, // [batch, channel, num_point]
    const float *inputB,      // [batch, channel, channel]
    float *inputA_grad,       // [batch, channel, num_point]
    int batch_size, int channel, int num_points) {
  int batch_id = blockIdx.x;
  int channelA_id = blockIdx.y;
  int point_id = threadIdx.x;

  if (batch_id >= batch_size || channelA_id >= channel ||
      point_id >= num_points) {
    return;
  }

  float grad = 0.0f;

  for (int c = 0; c < channel; ++c) {
    int out_idx = batch_id * channel * num_points + c * num_points + point_id;
    float dOut = output_grad[out_idx];
    int B_idx = batch_id * channel * channel + c * channel + channelA_id;
    float B = inputB[B_idx];
    grad += dOut * B;
  }

  int inputA_grad_idx =
      batch_id * channel * num_points + channelA_id * num_points + point_id;

  inputA_grad[inputA_grad_idx] = grad;
}

__global__ void batch_matrix_multiply_backward_inputB_kernel(
    const float *output_grad, // [batch, channel, num_point]
    const float *inputA,      // [batch, channel, num_point]
    float *inputB_grad,       // [batch, channel, channel]
    int batch_size, int channel, int num_points) {
  int batch_id = blockIdx.x;
  int channelB_id = blockIdx.y;
  int channelA_id = threadIdx.x;

  if (batch_id >= batch_size || channelA_id >= channel ||
      channelB_id >= channel) {
    return;
  }

  int inputB_grad_idx =
      batch_id * channel * channel + channelB_id * channel + channelA_id;

  float grad = 0.0f;

  for (int p = 0; p < num_points; ++p) {
    int inputA_idx =
        batch_id * channel * num_points + channelA_id * num_points + p;
    int output_grad_idx =
        batch_id * channel * num_points + channelB_id * num_points + p;

    float A = inputA[inputA_idx];
    float dOut = output_grad[output_grad_idx];
    grad += A * dOut;
  }

  atomicAdd(&inputB_grad[inputB_grad_idx], grad);
}

__global__ void accumulate_gradients_kernel(const float *grad1,
                                            const float *grad2,
                                            float *total_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    total_grad[idx] = grad1[idx] + grad2[idx];
  }
}

__global__ void softmax_kernel(const float *input, float *output,
                               int num_classes, int batch_size) {
  int sample_idx = blockIdx.x;
  extern __shared__ float shared_mem[];

  float max_val = -FLT_MAX;
  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    float val = input[sample_idx * num_classes + i];
    if (val > max_val) {
      max_val = val;
    }
  }
  shared_mem[threadIdx.x] = max_val;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x] =
          fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  max_val = shared_mem[0];
  __syncthreads();

  float sum = 0.0f;
  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    float exp_input = expf(input[sample_idx * num_classes + i] - max_val);
    output[sample_idx * num_classes + i] = exp_input;
    sum += exp_input;
  }
  shared_mem[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  sum = shared_mem[0];
  __syncthreads();

  if (sum < 1e-10f) {
    sum = 1e-10f;
  }

  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    output[sample_idx * num_classes + i] /= sum;
  }
}

__global__ void softmax_backward_kernel(const float *output_grad,
                                        const float *pred, float *logits_grad,
                                        int batch_size, int num_classes) {
  int sample_idx = blockIdx.x;

  extern __shared__ float shared_mem[];

  // Compute dot product of pred and output_grad
  float dot = 0.0f;
  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    float pred_val = pred[sample_idx * num_classes + i];
    float grad_output_val = output_grad[sample_idx * num_classes + i];
    dot += pred_val * grad_output_val;
  }
  shared_mem[threadIdx.x] = dot;
  __syncthreads();

  // Reduce the dot product across threads
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  dot = shared_mem[0];
  __syncthreads();

  // Compute the gradient
  for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
    float pred_current = pred[sample_idx * num_classes + i];
    float grad_output = output_grad[sample_idx * num_classes + i];
    float grad = pred_current * (grad_output - dot);
    logits_grad[sample_idx * num_classes + i] = grad;
  }
}

__global__ void cross_entropy_loss_kernel(const float *pred, const int *labels,
                                          float *loss, int batch_size,
                                          int num_classes) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int label = labels[idx];
    float prob = pred[idx * num_classes + label];
    prob = fmaxf(prob, 1e-20f); // Preventing log(0)
    loss[idx] = -logf(prob);
  }
}

__global__ void cross_entropy_loss_backward_kernel(const float *pred,
                                                   const int *labels,
                                                   float *input_grad,
                                                   int batch_size,
                                                   int num_classes) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int label = labels[idx];
    for (int c = 0; c < num_classes; ++c) {
      if (c == label) {
        input_grad[idx * num_classes + c] = pred[idx * num_classes + c] - 1.0f;
      } else {
        input_grad[idx * num_classes + c] = pred[idx * num_classes + c];
      }
    }
  }
}

__global__ void sgd_update_kernel(float *param, const float *grad, float lr,
                                  float clip, int size) {
  extern __shared__ float shared_grad[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Each thread loads a part of the gradient array into shared memory
  if (idx < size) {
    shared_grad[tid] = grad[idx];
  } else {
    shared_grad[tid] = 0.0f;
  }
  __syncthreads();

  // Compute the sum of squares (L2 norm) of the gradients
  float norm = 0.0f;
  for (int i = 0; i < blockDim.x; i++) {
    norm += shared_grad[i] * shared_grad[i];
  }
  __syncthreads();

  // Compute the block-wide norm
  if (tid == 0) {
    float block_norm = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
      block_norm += shared_grad[i] * shared_grad[i];
    }
    atomicAdd(&shared_grad[0],
              block_norm); // Shared memory holds the sum of squares
  }
  __syncthreads();

  norm = sqrtf(shared_grad[0]); // Norm of the entire gradient vector

  // Apply norm gradient clipping
  if (norm > clip) {
    float scale = clip / norm;
    if (idx < size) {
      param[idx] -= lr * grad[idx] * scale;
    }
  } else {
    if (idx < size) {
      param[idx] -= lr * grad[idx];
    }
  }
}

///////////////////////// Layer Definitions ///////////////////////////

struct ConvolutionalLayer {
  std::vector<float> h_weight;
  std::vector<float> h_bias;

  float *d_weight;
  float *d_bias;

  float *d_weight_grad;
  float *d_bias_grad;

  float *d_input_cache;

  int in_channels;
  int out_channels;

  ConvolutionalLayer()
      : d_weight(nullptr), d_bias(nullptr), d_weight_grad(nullptr),
        d_bias_grad(nullptr), d_input_cache(nullptr) {}

  ConvolutionalLayer(int in_c, int out_c)
      : d_weight(nullptr), d_bias(nullptr), d_weight_grad(nullptr),
        d_bias_grad(nullptr), d_input_cache(nullptr), in_channels(in_c),
        out_channels(out_c) {
    h_weight.resize(out_channels * in_channels);
    h_bias.resize(out_channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_channels));
    for (int i = 0; i < out_channels * in_channels; ++i) {
      h_weight[i] = dist(gen);
    }
    std::fill(h_bias.begin(), h_bias.end(), 0.0f);
  }

  void copy_to_device() {
    CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                          h_weight.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_weight_grad, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weight_grad, 0, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias_grad, 0, h_bias.size() * sizeof(float)));
  }

  void copy_to_host() {
    CUDA_CHECK(cudaMemcpy(h_weight.data(), d_weight,
                          h_weight.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias.data(), d_bias, h_bias.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  void forward(float *&d_input, float *&d_output, int batch_size,
               int num_points) {
    // allocate memory for output
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * out_channels * num_points *
                                         sizeof(float)));

    // cache input
    if (d_input_cache != nullptr) {
      CUDA_CHECK(cudaFree(d_input_cache));
    }
    CUDA_CHECK(cudaMalloc(&d_input_cache, batch_size * in_channels *
                                              num_points * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input_cache, d_input,
                          batch_size * in_channels * num_points * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // calculate
    dim3 block(num_points);
    dim3 grid(out_channels, batch_size);
    conv1d_kernel<<<grid, block>>>(d_input, d_output, d_weight, d_bias,
                                   in_channels, out_channels, batch_size,
                                   num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(float *&d_output_grad, float *&d_input_grad, int batch_size,
                int num_points) {
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch_size * in_channels * num_points *
                                             sizeof(float)));
    CUDA_CHECK(
        cudaMemset(d_input_grad, 0,
                   batch_size * in_channels * num_points * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_weight_grad, 0, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias_grad, 0, h_bias.size() * sizeof(float)));

    dim3 grid(batch_size, out_channels);
    dim3 block(1024);
    conv1d_backward_kernel<<<grid, block>>>(
        d_input_cache, d_weight, d_output_grad, d_input_grad, d_weight_grad,
        d_bias_grad, in_channels, out_channels, batch_size, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void update_parameters(float learning_rate) {
    int size = out_channels * in_channels;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_weight, d_weight_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size = out_channels;
    num_blocks = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_bias, d_bias_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

struct FullyConnectedLayer {
  std::vector<float> h_weight;
  std::vector<float> h_bias;

  float *d_weight;
  float *d_bias;

  float *d_weight_grad;
  float *d_bias_grad;

  float *d_input_cache;

  int in_features;
  int out_features;

  FullyConnectedLayer()
      : d_weight(nullptr), d_bias(nullptr), d_weight_grad(nullptr),
        d_bias_grad(nullptr), d_input_cache(nullptr) {}

  FullyConnectedLayer(int in_f, int out_f)
      : d_weight(nullptr), d_bias(nullptr), d_weight_grad(nullptr),
        d_bias_grad(nullptr), d_input_cache(nullptr), in_features(in_f),
        out_features(out_f) {
    h_weight.resize(out_features * in_features);
    h_bias.resize(out_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));
    for (int i = 0; i < out_features * in_features; ++i) {
      h_weight[i] = dist(gen);
    }
    std::fill(h_bias.begin(), h_bias.end(), 0.0f);
  }

  void copy_to_device() {
    CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                          h_weight.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_weight_grad, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, h_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weight_grad, 0, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias_grad, 0, h_bias.size() * sizeof(float)));
  }

  void copy_to_host() {
    CUDA_CHECK(cudaMemcpy(h_weight.data(), d_weight,
                          h_weight.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias.data(), d_bias, h_bias.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  void forward(float *&d_input, float *&d_output, int batch_size) {
    CUDA_CHECK(
        cudaMalloc(&d_output, batch_size * out_features * sizeof(float)));

    if (d_input_cache != nullptr) {
      CUDA_CHECK(cudaFree(d_input_cache));
    }
    CUDA_CHECK(
        cudaMalloc(&d_input_cache, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input_cache, d_input,
                          batch_size * in_features * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    dim3 block(256);
    dim3 grid((out_features + block.x - 1) / block.x, batch_size);
    fclayer_kernel<<<grid, block>>>(d_input, d_output, d_weight, d_bias,
                                    in_features, out_features, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(float *&d_output_grad, float *&d_input_grad, int batch_size) {
    CUDA_CHECK(
        cudaMalloc(&d_input_grad, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(d_input_grad, 0, batch_size * in_features * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_weight_grad, 0, h_weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias_grad, 0, h_bias.size() * sizeof(float)));

    dim3 block(1024);
    dim3 grid((out_features + block.x - 1) / block.x, batch_size);
    fclayer_backward_kernel<<<grid, block>>>(
        d_input_cache, d_weight, d_output_grad, d_input_grad, d_weight_grad,
        d_bias_grad, in_features, out_features, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void update_parameters(float learning_rate) {
    int size = out_features * in_features;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_weight, d_weight_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size = out_features;
    num_blocks = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_bias, d_bias_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

struct BatchNormLayer {
  std::vector<float> h_gamma;
  std::vector<float> h_beta;
  std::vector<float> h_running_mean;
  std::vector<float> h_running_variance;

  float *d_gamma;
  float *d_beta;
  float *d_running_mean;
  float *d_running_variance;

  float *d_gamma_grad;
  float *d_beta_grad;

  float *d_input_cache;
  float *d_mean_cache;
  float *d_variance_cache;

  int channels;
  float momentum;
  float epsilon;

  BatchNormLayer()
      : d_gamma(nullptr), d_beta(nullptr), d_running_mean(nullptr),
        d_running_variance(nullptr), d_gamma_grad(nullptr),
        d_beta_grad(nullptr), d_input_cache(nullptr), d_mean_cache(nullptr),
        d_variance_cache(nullptr) {}

  BatchNormLayer(int channels_, float epsilon_ = 1e-5f, float momentum_ = 0.9f)
      : d_gamma(nullptr), d_beta(nullptr), d_running_mean(nullptr),
        d_running_variance(nullptr), d_gamma_grad(nullptr),
        d_beta_grad(nullptr), d_input_cache(nullptr), d_mean_cache(nullptr),
        d_variance_cache(nullptr), channels(channels_), momentum(momentum_),
        epsilon(epsilon_) {
    h_gamma.resize(channels, 1.0f);
    h_beta.resize(channels, 0.0f);
    h_running_mean.resize(channels, 0.0f);
    h_running_variance.resize(channels, 1.0f);
  }

  void copy_to_device() {
    CUDA_CHECK(cudaMalloc(&d_gamma, channels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), channels * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_beta, channels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), channels * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_running_mean, channels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_running_mean, h_running_mean.data(),
                          channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_running_variance, channels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_running_variance, h_running_variance.data(),
                          channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_gamma_grad, channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_grad, channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma_grad, 0, channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta_grad, 0, channels * sizeof(float)));
  }

  void copy_to_host() {
    CUDA_CHECK(cudaMemcpy(h_gamma.data(), d_gamma, channels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_beta.data(), d_beta, channels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_running_mean.data(), d_running_mean,
                          channels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_running_variance.data(), d_running_variance,
                          channels * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void forward(float *&d_input, float *&d_output, int batch_size,
               int num_points, bool is_training) {
    // allocate output space
    CUDA_CHECK(cudaMalloc(&d_output,
                          batch_size * channels * num_points * sizeof(float)));

    // cache
    if (d_mean_cache != nullptr) {
      CUDA_CHECK(cudaFree(d_mean_cache));
    }
    CUDA_CHECK(cudaMalloc(&d_mean_cache, channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mean_cache, 0, channels * sizeof(float)));

    if (d_variance_cache != nullptr) {
      CUDA_CHECK(cudaFree(d_variance_cache));
    }
    CUDA_CHECK(cudaMalloc(&d_variance_cache, channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_variance_cache, 0, channels * sizeof(float)));

    if (d_input_cache != nullptr) {
      CUDA_CHECK(cudaFree(d_input_cache));
    }
    CUDA_CHECK(cudaMalloc(&d_input_cache,
                          batch_size * channels * num_points * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input_cache, d_input,
                          batch_size * channels * num_points * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // calculate
    int block_size = 256;
    dim3 grid(channels, batch_size);
    dim3 block(256);

    compute_mean_kernel<<<grid, block, block_size * sizeof(float)>>>(
        d_input, d_mean_cache, batch_size, channels, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    compute_variance_kernel<<<grid, block, block_size * sizeof(float)>>>(
        d_input, d_mean_cache, d_variance_cache, batch_size, channels,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (is_training) {
      update_running_mean_variance_kernel<<<(channels + 255) / 256, 256>>>(
          d_running_mean, d_running_variance, d_mean_cache, d_variance_cache,
          momentum, channels);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      CUDA_CHECK(cudaMemcpy(d_mean_cache, d_running_mean,
                            channels * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(d_variance_cache, d_running_variance,
                            channels * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    }

    dim3 bn_block_dim(1024);
    dim3 bn_grid_dim(channels, batch_size);
    batchnorm_kernel<<<bn_grid_dim, bn_block_dim>>>(
        d_input, d_output, d_mean_cache, d_variance_cache, d_gamma, d_beta,
        epsilon, channels, batch_size, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void backward(float *&d_output_grad, float *&d_input_grad, int batch_size,
                int num_points) {
    CUDA_CHECK(cudaMalloc(&d_input_grad,
                          batch_size * channels * num_points * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_input_grad, 0,
                          batch_size * channels * num_points * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_gamma_grad, 0, channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta_grad, 0, channels * sizeof(float)));

    int threads_per_block = 256;
    int blocks_per_grid = channels;
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    batchnorm_backward_kernel<<<blocks_per_grid, threads_per_block,
                                shared_mem_size>>>(
        d_input_cache, d_output_grad, d_mean_cache, d_variance_cache, d_gamma,
        d_input_grad, d_gamma_grad, d_beta_grad, epsilon, channels, batch_size,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void update_parameters(float learning_rate) {
    int size = channels;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_gamma, d_gamma_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    sgd_update_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_beta, d_beta_grad, learning_rate, 1.0f, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

struct STNkd {
  int m_k;
  std::string module_name;

  std::vector<ConvolutionalLayer> conv;
  std::vector<FullyConnectedLayer> fc;
  std::vector<BatchNormLayer> bn;

  // cache for backward pass
  float *d_conv1_output = nullptr;
  float *d_bn1_output = nullptr;
  float *d_relu1_output = nullptr;
  float *d_conv2_output = nullptr;
  float *d_bn2_output = nullptr;
  float *d_relu2_output = nullptr;
  float *d_conv3_output = nullptr;
  float *d_bn3_output = nullptr;
  float *d_relu3_output = nullptr;
  float *d_maxpool_output = nullptr;
  float *d_fc1_output = nullptr;
  float *d_bn4_output = nullptr;
  float *d_relu4_output = nullptr;
  float *d_fc2_output = nullptr;
  float *d_bn5_output = nullptr;
  float *d_relu5_output = nullptr;
  float *d_fc3_output = nullptr;
  float *d_add_iden_output = nullptr;

  float *d_fc3_input_grad = nullptr;
  float *d_relu5_input_grad = nullptr;
  float *d_bn5_input_grad = nullptr;
  float *d_fc2_input_grad = nullptr;
  float *d_relu4_input_grad = nullptr;
  float *d_bn4_input_grad = nullptr;
  float *d_fc1_input_grad = nullptr;
  float *d_maxpool_input_grad = nullptr;
  float *d_relu3_input_grad = nullptr;
  float *d_bn3_input_grad = nullptr;
  float *d_conv3_input_grad = nullptr;
  float *d_relu2_input_grad = nullptr;
  float *d_bn2_input_grad = nullptr;
  float *d_conv2_input_grad = nullptr;
  float *d_relu1_input_grad = nullptr;
  float *d_bn1_input_grad = nullptr;
  float *d_conv1_input_grad = nullptr;

  STNkd() {}

  STNkd(int k)
      : m_k(k), module_name(k == 3 ? "stn" : "fstn"),
        conv({ConvolutionalLayer(k, 64), ConvolutionalLayer(64, 128),
              ConvolutionalLayer(128, 1024)}),
        fc({FullyConnectedLayer(1024, 512), FullyConnectedLayer(512, 256),
            FullyConnectedLayer(256, m_k * m_k)}),
        bn({BatchNormLayer(64), BatchNormLayer(128), BatchNormLayer(1024),
            BatchNormLayer(512), BatchNormLayer(256)}) {}

  void copy_to_device() {
    for (auto &layer : conv) {
      layer.copy_to_device();
    }
    for (auto &layer : bn) {
      layer.copy_to_device();
    }
    for (auto &layer : fc) {
      layer.copy_to_device();
    }
  }

  void copy_to_host(std::map<std::string, std::vector<float>> &params) {
    for (auto &layer : conv) {
      layer.copy_to_host();
    }
    for (auto &layer : fc) {
      layer.copy_to_host();
    }
    for (auto &layer : bn) {
      layer.copy_to_host();
    }

    for (int i = 0; i < conv.size(); ++i) {
      params["feat." + module_name + ".conv" + std::to_string(i + 1) +
             ".weight"] = conv[i].h_weight;
      params["feat." + module_name + ".conv" + std::to_string(i + 1) +
             ".bias"] = conv[i].h_bias;
    }
    for (int i = 0; i < fc.size(); ++i) {
      params["feat." + module_name + ".fc" + std::to_string(i + 1) +
             ".weight"] = fc[i].h_weight;
      params["feat." + module_name + ".fc" + std::to_string(i + 1) + ".bias"] =
          fc[i].h_bias;
    }
    for (int i = 0; i < bn.size(); ++i) {
      params["feat." + module_name + ".bn" + std::to_string(i + 1) +
             ".weight"] = bn[i].h_gamma;
      params["feat." + module_name + ".bn" + std::to_string(i + 1) + ".bias"] =
          bn[i].h_beta;
      params["feat." + module_name + ".bn" + std::to_string(i + 1) +
             ".running_mean"] = bn[i].h_running_mean;
      params["feat." + module_name + ".bn" + std::to_string(i + 1) +
             ".running_var"] = bn[i].h_running_variance;
    }
  }

  void forward(float *&d_stn_input, float *&d_stn_output, int batch_size,
               int num_points) {
    conv[0].forward(d_stn_input, d_conv1_output, batch_size, num_points);

    bn[0].forward(d_conv1_output, d_bn1_output, batch_size, num_points, true);

    CUDA_CHECK(cudaMalloc(&d_relu1_output,
                          batch_size * 64 * num_points * sizeof(float)));
    relu_kernel<<<(batch_size * 64 * num_points + 255) / 256, 256>>>(
        d_bn1_output, d_relu1_output, batch_size, 64, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    conv[1].forward(d_relu1_output, d_conv2_output, batch_size, num_points);

    bn[1].forward(d_conv2_output, d_bn2_output, batch_size, num_points, true);

    CUDA_CHECK(cudaMalloc(&d_relu2_output,
                          batch_size * 128 * num_points * sizeof(float)));
    relu_kernel<<<(batch_size * 128 * num_points + 255) / 256, 256>>>(
        d_bn2_output, d_relu2_output, batch_size, 128, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    conv[2].forward(d_relu2_output, d_conv3_output, batch_size, num_points);

    bn[2].forward(d_conv3_output, d_bn3_output, batch_size, num_points, true);

    CUDA_CHECK(cudaMalloc(&d_relu3_output,
                          batch_size * 1024 * num_points * sizeof(float)));
    relu_kernel<<<(batch_size * 1024 * num_points + 255) / 256, 256>>>(
        d_bn3_output, d_relu3_output, batch_size, 1024, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMalloc(&d_maxpool_output, batch_size * 1024 * sizeof(float)));
    dim3 max_pool_grid(batch_size, 1024);
    dim3 max_pool_block(256);
    max_pool_kernel<<<max_pool_grid, max_pool_block, 256 * sizeof(float)>>>(
        d_relu3_output, d_maxpool_output, num_points, 1024);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fc[0].forward(d_maxpool_output, d_fc1_output, batch_size);

    bn[3].forward(d_fc1_output, d_bn4_output, batch_size, 1, true);

    CUDA_CHECK(cudaMalloc(&d_relu4_output, batch_size * 512 * sizeof(float)));
    relu_kernel<<<(batch_size * 512 + 255) / 256, 256>>>(
        d_bn4_output, d_relu4_output, batch_size, 512, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fc[1].forward(d_relu4_output, d_fc2_output, batch_size);

    bn[4].forward(d_fc2_output, d_bn5_output, batch_size, 1, true);

    CUDA_CHECK(cudaMalloc(&d_relu5_output, batch_size * 256 * sizeof(float)));
    relu_kernel<<<(batch_size * 256 + 255) / 256, 256>>>(
        d_bn5_output, d_relu5_output, batch_size, 256, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fc[2].forward(d_relu5_output, d_fc3_output, batch_size);

    // add iden
    CUDA_CHECK(
        cudaMalloc(&d_add_iden_output, batch_size * m_k * m_k * sizeof(float)));
    add_identity_matrix_kernel<<<((batch_size * m_k * m_k + 255) / 256), 256>>>(
        d_fc3_output, d_add_iden_output, m_k, batch_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_stn_output = d_add_iden_output;
  }

  void backward(float *&d_stn_output_grad, float *&d_stn_input_grad,
                int batch_size, int num_points) {

    fc[2].backward(d_stn_output_grad, d_fc3_input_grad, batch_size);

    CUDA_CHECK(
        cudaMalloc(&d_relu5_input_grad, batch_size * 256 * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 256 + 255) / 256, 256>>>(
        d_bn5_output, d_fc3_input_grad, d_relu5_input_grad, batch_size, 256, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[4].backward(d_relu5_input_grad, d_bn5_input_grad, batch_size, 1);

    fc[1].backward(d_bn5_input_grad, d_fc2_input_grad, batch_size);

    CUDA_CHECK(
        cudaMalloc(&d_relu4_input_grad, batch_size * 512 * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 512 + 255) / 256, 256>>>(
        d_bn4_output, d_fc2_input_grad, d_relu4_input_grad, batch_size, 512, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[3].backward(d_relu4_input_grad, d_bn4_input_grad, batch_size, 1);

    fc[0].backward(d_bn4_input_grad, d_fc1_input_grad, batch_size);

    CUDA_CHECK(cudaMalloc(&d_maxpool_input_grad,
                          batch_size * 1024 * num_points * sizeof(float)));
    dim3 max_pool_backward_grid(batch_size, 1024);
    dim3 max_pool_backward_block(256);
    size_t shared_mem_size = 2 * max_pool_backward_block.x * sizeof(float);
    max_pool_backward_kernel<<<max_pool_backward_grid, max_pool_backward_block,
                               shared_mem_size>>>(
        d_relu3_output, d_fc1_input_grad, d_maxpool_input_grad, num_points,
        1024);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&d_relu3_input_grad,
                          batch_size * 1024 * num_points * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 1024 * num_points + 255) / 256, 256>>>(
        d_bn3_output, d_maxpool_input_grad, d_relu3_input_grad, batch_size,
        1024, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[2].backward(d_relu3_input_grad, d_bn3_input_grad, batch_size,
                   num_points);

    conv[2].backward(d_bn3_input_grad, d_conv3_input_grad, batch_size,
                     num_points);

    CUDA_CHECK(cudaMalloc(&d_relu2_input_grad,
                          batch_size * 128 * num_points * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 128 * num_points + 255) / 256, 256>>>(
        d_bn2_output, d_conv3_input_grad, d_relu2_input_grad, batch_size, 128,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[1].backward(d_relu2_input_grad, d_bn2_input_grad, batch_size,
                   num_points);

    conv[1].backward(d_bn2_input_grad, d_conv2_input_grad, batch_size,
                     num_points);

    CUDA_CHECK(cudaMalloc(&d_relu1_input_grad,
                          batch_size * 64 * num_points * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 64 * num_points + 255) / 256, 256>>>(
        d_bn1_output, d_conv2_input_grad, d_relu1_input_grad, batch_size, 64,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[0].backward(d_relu1_input_grad, d_bn1_input_grad, batch_size,
                   num_points);

    conv[0].backward(d_bn1_input_grad, d_conv1_input_grad, batch_size,
                     num_points);

    d_stn_input_grad = d_conv1_input_grad;
  }

  void update_parameters(float learning_rate) {
    for (auto &layer : conv) {
      layer.update_parameters(learning_rate);
    }
    for (auto &layer : bn) {
      layer.update_parameters(learning_rate);
    }
    for (auto &layer : fc) {
      layer.update_parameters(learning_rate);
    }
  }

  void free_cache() {
    // free
    CUDA_CHECK(cudaFree(d_fc3_input_grad));
    CUDA_CHECK(cudaFree(d_relu5_input_grad));
    CUDA_CHECK(cudaFree(d_bn5_input_grad));
    CUDA_CHECK(cudaFree(d_fc2_input_grad));
    CUDA_CHECK(cudaFree(d_relu4_input_grad));
    CUDA_CHECK(cudaFree(d_bn4_input_grad));
    CUDA_CHECK(cudaFree(d_fc1_input_grad));
    CUDA_CHECK(cudaFree(d_maxpool_input_grad));
    CUDA_CHECK(cudaFree(d_relu3_input_grad));
    CUDA_CHECK(cudaFree(d_bn3_input_grad));
    CUDA_CHECK(cudaFree(d_conv3_input_grad));
    CUDA_CHECK(cudaFree(d_relu2_input_grad));
    CUDA_CHECK(cudaFree(d_bn2_input_grad));
    CUDA_CHECK(cudaFree(d_conv2_input_grad));
    CUDA_CHECK(cudaFree(d_relu1_input_grad));
    CUDA_CHECK(cudaFree(d_bn1_input_grad));
    CUDA_CHECK(cudaFree(d_conv1_input_grad));

    CUDA_CHECK(cudaFree(d_conv1_output));
    CUDA_CHECK(cudaFree(d_bn1_output));
    CUDA_CHECK(cudaFree(d_relu1_output));
    CUDA_CHECK(cudaFree(d_conv2_output));
    CUDA_CHECK(cudaFree(d_bn2_output));
    CUDA_CHECK(cudaFree(d_relu2_output));
    CUDA_CHECK(cudaFree(d_conv3_output));
    CUDA_CHECK(cudaFree(d_bn3_output));
    CUDA_CHECK(cudaFree(d_relu3_output));
    CUDA_CHECK(cudaFree(d_maxpool_output));
    CUDA_CHECK(cudaFree(d_fc1_output));
    CUDA_CHECK(cudaFree(d_bn4_output));
    CUDA_CHECK(cudaFree(d_relu4_output));
    CUDA_CHECK(cudaFree(d_fc2_output));
    CUDA_CHECK(cudaFree(d_bn5_output));
    CUDA_CHECK(cudaFree(d_relu5_output));
    CUDA_CHECK(cudaFree(d_fc3_output));
    CUDA_CHECK(cudaFree(d_add_iden_output));
  }
};

struct PointNetEncoder {
  STNkd stn3d;
  STNkd stnkd;

  std::vector<ConvolutionalLayer> conv;
  std::vector<BatchNormLayer> bn;

  // cache for backward pass
  float *d_encoder_input_cache = nullptr;

  float *d_stn3d_output = nullptr;
  float *d_bmm1_output = nullptr;
  float *d_conv1_output = nullptr;
  float *d_bn1_output = nullptr;
  float *d_relu1_output = nullptr;
  float *d_stnkd_output = nullptr;
  float *d_bmm2_output = nullptr;
  float *d_conv2_output = nullptr;
  float *d_bn2_output = nullptr;
  float *d_relu2_output = nullptr;
  float *d_conv3_output = nullptr;
  float *d_bn3_output = nullptr;
  float *d_maxpool_output = nullptr;

  float *d_maxpool_input_grad = nullptr;
  float *d_bn3_input_grad = nullptr;
  float *d_conv3_input_grad = nullptr;
  float *d_relu2_input_grad = nullptr;
  float *d_bn2_input_grad = nullptr;
  float *d_conv2_input_grad = nullptr;
  float *d_bmm2_inputA_grad = nullptr; // stnkd and bmm2
  float *d_bmm2_inputB_grad = nullptr; // stnkd and bmm2
  float *d_stnkd_input_grad = nullptr; // stnkd and bmm2
  float *d_acc2_input_grad = nullptr;  // stnkd and bmm2
  float *d_relu1_input_grad = nullptr;
  float *d_bn1_input_grad = nullptr;
  float *d_conv1_input_grad = nullptr;
  float *d_bmm1_inputA_grad = nullptr; // stn3d and bmm1
  float *d_bmm1_inputB_grad = nullptr; // stn3d and bmm1
  float *d_stn3d_input_grad = nullptr; // stn3d and bmm1
  float *d_acc1_input_grad = nullptr;  // stn3d and bmm1

  PointNetEncoder()
      : stn3d(3), stnkd(64),
        conv({ConvolutionalLayer(3, 64), ConvolutionalLayer(64, 128),
              ConvolutionalLayer(128, 1024)}),
        bn({BatchNormLayer(64), BatchNormLayer(128), BatchNormLayer(1024)}) {}

  void copy_to_device() {
    stn3d.copy_to_device();
    stnkd.copy_to_device();
    for (auto &layer : conv) {
      layer.copy_to_device();
    }
    for (auto &layer : bn) {
      layer.copy_to_device();
    }
  }

  void copy_to_host(std::map<std::string, std::vector<float>> &params) {
    stn3d.copy_to_host(params);
    stnkd.copy_to_host(params);

    for (auto &layer : conv) {
      layer.copy_to_host();
    }
    for (auto &layer : bn) {
      layer.copy_to_host();
    }

    for (int i = 0; i < conv.size(); ++i) {
      params["feat.conv" + std::to_string(i + 1) + ".weight"] =
          conv[i].h_weight;
      params["feat.conv" + std::to_string(i + 1) + ".bias"] = conv[i].h_bias;
    }
    for (int i = 0; i < bn.size(); ++i) {
      params["feat.bn" + std::to_string(i + 1) + ".weight"] = bn[i].h_gamma;
      params["feat.bn" + std::to_string(i + 1) + ".bias"] = bn[i].h_beta;
      params["feat.bn" + std::to_string(i + 1) + ".running_mean"] =
          bn[i].h_running_mean;
      params["feat.bn" + std::to_string(i + 1) + ".running_var"] =
          bn[i].h_running_variance;
    }
  }

  void forward(float *&d_encoder_input, float *&d_encoder_output,
               int batch_size, int num_points) {
    // cache input, [B, 3, N]
    CUDA_CHECK(cudaMalloc(&d_encoder_input_cache,
                          batch_size * 3 * num_points * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_encoder_input_cache, d_encoder_input,
                          batch_size * 3 * num_points * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    // [B, 3, N] -> [B, 3, 3]
    stn3d.forward(d_encoder_input, d_stn3d_output, batch_size, num_points);
    // [B, 3, N] * [B, 3, 3] -> [B, 3, N]
    CUDA_CHECK(cudaMalloc(&d_bmm1_output,
                          batch_size * 3 * num_points * sizeof(float)));
    dim3 bmm1_block(256);
    dim3 bmm1_grid(batch_size, 3,
                   ((num_points + bmm1_block.x - 1) / bmm1_block.x));
    batch_matrix_multiply_kernel<<<bmm1_grid, bmm1_block>>>(
        d_encoder_input, d_stn3d_output, d_bmm1_output, batch_size, 3,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    conv[0].forward(d_bmm1_output, d_conv1_output, batch_size, num_points);

    bn[0].forward(d_conv1_output, d_bn1_output, batch_size, num_points, true);

    CUDA_CHECK(cudaMalloc(&d_relu1_output,
                          batch_size * 64 * num_points * sizeof(float)));
    relu_kernel<<<(batch_size * 64 * num_points + 255) / 256, 256>>>(
        d_bn1_output, d_relu1_output, batch_size, 64, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // stnkd and bmm
    stnkd.forward(d_relu1_output, d_stnkd_output, batch_size, num_points);

    CUDA_CHECK(cudaMalloc(&d_bmm2_output,
                          batch_size * 64 * num_points * sizeof(float)));
    dim3 bmm2_block(256);
    dim3 bmm2_grid(batch_size, 64,
                   ((num_points + bmm2_block.x - 1) / bmm2_block.x));
    batch_matrix_multiply_kernel<<<bmm2_grid, bmm2_block>>>(
        d_relu1_output, d_stnkd_output, d_bmm2_output, batch_size, 64,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    conv[1].forward(d_bmm2_output, d_conv2_output, batch_size, num_points);

    bn[1].forward(d_conv2_output, d_bn2_output, batch_size, num_points, true);

    CUDA_CHECK(cudaMalloc(&d_relu2_output,
                          batch_size * 128 * num_points * sizeof(float)));
    relu_kernel<<<(batch_size * 128 * num_points + 255) / 256, 256>>>(
        d_bn2_output, d_relu2_output, batch_size, 128, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    conv[2].forward(d_relu2_output, d_conv3_output, batch_size, num_points);

    bn[2].forward(d_conv3_output, d_bn3_output, batch_size, num_points, true);

    CUDA_CHECK(
        cudaMalloc(&d_maxpool_output, batch_size * 1024 * sizeof(float)));
    dim3 max_pool_grid(batch_size, 1024);
    dim3 max_pool_block(256);
    max_pool_kernel<<<max_pool_grid, max_pool_block, 256 * sizeof(float)>>>(
        d_bn3_output, d_maxpool_output, num_points, 1024);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_encoder_output = d_maxpool_output;
  }

  void backward(float *&d_encoder_output_grad, float *&d_encoder_input_grad,
                int batch_size, int num_points) {
    CUDA_CHECK(cudaMalloc(&d_maxpool_input_grad,
                          batch_size * 1024 * num_points * sizeof(float)));
    dim3 max_pool_backward_grid(batch_size, 1024);
    dim3 max_pool_backward_block(256);
    size_t shared_mem_size = 2 * max_pool_backward_block.x * sizeof(float);
    max_pool_backward_kernel<<<max_pool_backward_grid, max_pool_backward_block,
                               shared_mem_size>>>(
        d_bn3_output, d_encoder_output_grad, d_maxpool_input_grad, num_points,
        1024);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[2].backward(d_maxpool_input_grad, d_bn3_input_grad, batch_size,
                   num_points);

    conv[2].backward(d_bn3_input_grad, d_conv3_input_grad, batch_size,
                     num_points);

    CUDA_CHECK(cudaMalloc(&d_relu2_input_grad,
                          batch_size * 128 * num_points * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 128 * num_points + 255) / 256, 256>>>(
        d_bn2_output, d_conv3_input_grad, d_relu2_input_grad, batch_size, 128,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[1].backward(d_relu2_input_grad, d_bn2_input_grad, batch_size,
                   num_points);

    conv[1].backward(d_bn2_input_grad, d_conv2_input_grad, batch_size,
                     num_points);

    // stnkd backward
    CUDA_CHECK(cudaMalloc(&d_bmm2_inputA_grad,
                          batch_size * 64 * num_points * sizeof(float)));
    dim3 bmm2_grid(batch_size, 64, 1);
    batch_matrix_multiply_backward_inputA_kernel<<<bmm2_grid, num_points>>>(
        d_conv2_input_grad, d_bmm2_output, d_bmm2_inputA_grad, batch_size, 64,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMalloc(&d_bmm2_inputB_grad, batch_size * 64 * 64 * sizeof(float)));
    batch_matrix_multiply_backward_inputB_kernel<<<bmm2_grid, 64>>>(
        d_conv2_input_grad, d_relu1_output, d_bmm2_inputB_grad, batch_size, 64,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    stnkd.backward(d_bmm2_inputB_grad, d_stnkd_input_grad, batch_size,
                   num_points);
    // grad add
    CUDA_CHECK(cudaMalloc(&d_acc2_input_grad,
                          batch_size * 64 * num_points * sizeof(float)));
    accumulate_gradients_kernel<<<(batch_size * 64 * num_points + 255) / 256,
                                  256>>>(d_stnkd_input_grad, d_bmm2_inputA_grad,
                                         d_acc2_input_grad,
                                         batch_size * 64 * num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&d_relu1_input_grad,
                          batch_size * 64 * num_points * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 64 * num_points + 255) / 256, 256>>>(
        d_bn1_output, d_acc2_input_grad, d_relu1_input_grad, batch_size, 64,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[0].backward(d_relu1_input_grad, d_bn1_input_grad, batch_size,
                   num_points);

    conv[0].backward(d_bn1_input_grad, d_conv1_input_grad, batch_size,
                     num_points);

    // stn3d backward
    CUDA_CHECK(cudaMalloc(&d_bmm1_inputA_grad,
                          batch_size * 3 * num_points * sizeof(float)));
    dim3 bmm1_grid(batch_size, 3, 1);
    batch_matrix_multiply_backward_inputA_kernel<<<bmm2_grid, num_points>>>(
        d_conv1_input_grad, d_bmm1_output, d_bmm1_inputA_grad, batch_size, 3,
        num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMalloc(&d_bmm1_inputB_grad, batch_size * 3 * 3 * sizeof(float)));
    batch_matrix_multiply_backward_inputB_kernel<<<bmm1_grid, 3>>>(
        d_conv1_input_grad, d_encoder_input_cache, d_bmm1_inputB_grad,
        batch_size, 3, num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    stn3d.backward(d_bmm1_inputB_grad, d_stn3d_input_grad, batch_size,
                   num_points);

    // grad add
    CUDA_CHECK(cudaMalloc(&d_acc1_input_grad,
                          batch_size * 3 * num_points * sizeof(float)));
    accumulate_gradients_kernel<<<(batch_size * 3 * num_points + 255) / 256,
                                  256>>>(d_stn3d_input_grad, d_bmm1_inputA_grad,
                                         d_acc1_input_grad,
                                         batch_size * 3 * num_points);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    d_encoder_input_grad = d_acc1_input_grad;
  }

  void update_parameters(float learning_rate) {
    stn3d.update_parameters(learning_rate);
    stnkd.update_parameters(learning_rate);
    for (auto &layer : conv) {
      layer.update_parameters(learning_rate);
    }
    for (auto &layer : bn) {
      layer.update_parameters(learning_rate);
    }
  }

  void free_cache() {
    stn3d.free_cache();
    stnkd.free_cache();
    // free
    CUDA_CHECK(cudaFree(d_maxpool_input_grad));
    CUDA_CHECK(cudaFree(d_bn3_input_grad));
    CUDA_CHECK(cudaFree(d_conv3_input_grad));
    CUDA_CHECK(cudaFree(d_relu2_input_grad));
    CUDA_CHECK(cudaFree(d_bn2_input_grad));
    CUDA_CHECK(cudaFree(d_conv2_input_grad));
    CUDA_CHECK(cudaFree(d_bmm2_inputA_grad));
    CUDA_CHECK(cudaFree(d_bmm2_inputB_grad));
    // CUDA_CHECK(cudaFree(d_stnkd_input_grad));
    CUDA_CHECK(cudaFree(d_acc2_input_grad));
    CUDA_CHECK(cudaFree(d_relu1_input_grad));
    CUDA_CHECK(cudaFree(d_bn1_input_grad));
    CUDA_CHECK(cudaFree(d_conv1_input_grad));
    CUDA_CHECK(cudaFree(d_bmm1_inputA_grad));
    CUDA_CHECK(cudaFree(d_bmm1_inputB_grad));
    // CUDA_CHECK(cudaFree(d_stn3d_input_grad));
    CUDA_CHECK(cudaFree(d_acc1_input_grad));

    // CUDA_CHECK(cudaFree(d_stn3d_output));
    CUDA_CHECK(cudaFree(d_bmm1_output));
    CUDA_CHECK(cudaFree(d_conv1_output));
    CUDA_CHECK(cudaFree(d_bn1_output));
    CUDA_CHECK(cudaFree(d_relu1_output));
    // CUDA_CHECK(cudaFree(d_stnkd_output));
    CUDA_CHECK(cudaFree(d_bmm2_output));
    CUDA_CHECK(cudaFree(d_conv2_output));
    CUDA_CHECK(cudaFree(d_bn2_output));
    CUDA_CHECK(cudaFree(d_relu2_output));
    CUDA_CHECK(cudaFree(d_conv3_output));
    CUDA_CHECK(cudaFree(d_bn3_output));
    CUDA_CHECK(cudaFree(d_maxpool_output));

    CUDA_CHECK(cudaFree(d_encoder_input_cache));
  }
};

struct Classifier {
  PointNetEncoder pointnetencoder;

  std::vector<FullyConnectedLayer> fc;
  std::vector<BatchNormLayer> bn;

  int num_classes;

  // cache forward outputs for backward pass
  float *d_encoder_output = nullptr;
  float *d_fc1_output = nullptr;
  float *d_bn1_output = nullptr;
  float *d_relu1_output = nullptr;
  float *d_fc2_output = nullptr;
  float *d_bn2_output = nullptr;
  float *d_relu2_output = nullptr;
  float *d_fc3_output = nullptr;

  float *d_fc3_input_grad = nullptr;
  float *d_relu2_input_grad = nullptr;
  float *d_bn2_input_grad = nullptr;
  float *d_fc2_input_grad = nullptr;
  float *d_relu1_input_grad = nullptr;
  float *d_bn1_input_grad = nullptr;
  float *d_fc1_input_grad = nullptr;
  float *d_encoder_input_grad = nullptr;

  Classifier(int num_classes_ = 10)
      : pointnetencoder(),
        fc({FullyConnectedLayer(1024, 512), FullyConnectedLayer(512, 256),
            FullyConnectedLayer(256, num_classes_)}),
        bn({BatchNormLayer(512), BatchNormLayer(256)}),
        num_classes(num_classes_) {}

  void copy_to_device() {
    pointnetencoder.copy_to_device();
    for (auto &layer : fc) {
      layer.copy_to_device();
    }
    for (auto &layer : bn) {
      layer.copy_to_device();
    }
  }

  void copy_to_host(std::map<std::string, std::vector<float>> &params) {
    pointnetencoder.copy_to_host(params);

    for (auto &layer : fc) {
      layer.copy_to_host();
    }
    for (auto &layer : bn) {
      layer.copy_to_host();
    }

    for (int i = 0; i < fc.size(); ++i) {
      params["fc" + std::to_string(i + 1) + ".weight"] = fc[i].h_weight;
      params["fc" + std::to_string(i + 1) + ".bias"] = fc[i].h_bias;
    }
    for (int i = 0; i < bn.size(); ++i) {
      params["bn" + std::to_string(i + 1) + ".weight"] = bn[i].h_gamma;
      params["bn" + std::to_string(i + 1) + ".bias"] = bn[i].h_beta;
      params["bn" + std::to_string(i + 1) + ".running_mean"] =
          bn[i].h_running_mean;
      params["bn" + std::to_string(i + 1) + ".running_var"] =
          bn[i].h_running_variance;
    }
  }

  void forward(float *&d_classifier_input, float *&d_classifier_output,
               int batch_size, int num_points) {

    pointnetencoder.forward(d_classifier_input, d_encoder_output, batch_size,
                            num_points);

    fc[0].forward(d_encoder_output, d_fc1_output, batch_size);

    bn[0].forward(d_fc1_output, d_bn1_output, batch_size, 1, true);

    CUDA_CHECK(cudaMalloc(&d_relu1_output, batch_size * 512 * sizeof(float)));
    relu_kernel<<<(batch_size * 512 + 255) / 256, 256>>>(
        d_bn1_output, d_relu1_output, batch_size, 512, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fc[1].forward(d_relu1_output, d_fc2_output, batch_size);

    bn[1].forward(d_fc2_output, d_bn2_output, batch_size, 1, true);

    CUDA_CHECK(cudaMalloc(&d_relu2_output, batch_size * 256 * sizeof(float)));
    relu_kernel<<<(batch_size * 256 + 255) / 256, 256>>>(
        d_bn2_output, d_relu2_output, batch_size, 256, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fc[2].forward(d_relu2_output, d_fc3_output, batch_size);

    d_classifier_output = d_fc3_output;
  }

  void backward(float *&d_classifier_output_grad,
                float *&d_classifier_input_grad, int batch_size,
                int num_points) {

    fc[2].backward(d_classifier_output_grad, d_fc3_input_grad, batch_size);

    CUDA_CHECK(
        cudaMalloc(&d_relu2_input_grad, batch_size * 256 * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 256 + 255) / 256, 256>>>(
        d_bn2_output, d_fc3_input_grad, d_relu2_input_grad, batch_size, 256, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[1].backward(d_relu2_input_grad, d_bn2_input_grad, batch_size, 1);

    fc[1].backward(d_bn2_input_grad, d_fc2_input_grad, batch_size);

    CUDA_CHECK(
        cudaMalloc(&d_relu1_input_grad, batch_size * 512 * sizeof(float)));
    relu_backward_kernel<<<(batch_size * 512 + 255) / 256, 256>>>(
        d_bn1_output, d_fc2_input_grad, d_relu1_input_grad, batch_size, 512, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bn[0].backward(d_relu1_input_grad, d_bn1_input_grad, batch_size, 1);

    fc[0].backward(d_bn1_input_grad, d_fc1_input_grad, batch_size);

    pointnetencoder.backward(d_fc1_input_grad, d_encoder_input_grad, batch_size,
                             num_points);

    d_classifier_input_grad = d_encoder_input_grad;
  }

  void update_parameters(float learning_rate) {
    pointnetencoder.update_parameters(learning_rate);
    for (auto &layer : fc) {
      layer.update_parameters(learning_rate);
    }
    for (auto &layer : bn) {
      layer.update_parameters(learning_rate);
    }
  }

  void free_cache() {
    pointnetencoder.free_cache();
    // free
    CUDA_CHECK(cudaFree(d_fc3_input_grad));
    CUDA_CHECK(cudaFree(d_relu2_input_grad));
    CUDA_CHECK(cudaFree(d_bn2_input_grad));
    CUDA_CHECK(cudaFree(d_fc2_input_grad));
    CUDA_CHECK(cudaFree(d_relu1_input_grad));
    CUDA_CHECK(cudaFree(d_bn1_input_grad));
    CUDA_CHECK(cudaFree(d_fc1_input_grad));

    CUDA_CHECK(cudaFree(d_fc1_output));
    CUDA_CHECK(cudaFree(d_bn1_output));
    CUDA_CHECK(cudaFree(d_relu1_output));
    CUDA_CHECK(cudaFree(d_fc2_output));
    CUDA_CHECK(cudaFree(d_bn2_output));
    CUDA_CHECK(cudaFree(d_relu2_output));
    CUDA_CHECK(cudaFree(d_fc3_output));
  }
};

// Point cloud sampling function (uniform sampling)
std::vector<std::vector<float>>
sample_point_clouds(const std::vector<std::vector<float>> &list_of_points,
                    int num_points = 1024) {
  int point_dim = 3;
  int target_size = num_points * point_dim;

  std::vector<std::vector<float>> sampled_point_clouds;

  for (const auto &points : list_of_points) {
    int n_points = points.size() / point_dim;
    std::vector<float> sampled_points(target_size);

    if (n_points >= num_points) {
      float step = static_cast<float>(n_points) / num_points;
      for (int i = 0; i < num_points; ++i) {
        int idx = static_cast<int>(i * step);
        for (int d = 0; d < point_dim; ++d) {
          sampled_points[i * point_dim + d] = points[idx * point_dim + d];
        }
      }
    } else {
      std::cerr << "Warning: Point cloud with less points than required!\n";
      std::cerr << "Falling back to random repeated sampling.\n";
      std::uniform_int_distribution<> dis(0, n_points - 1);
      std::random_device rd;
      std::mt19937 gen(rd());
      for (int i = 0; i < num_points; ++i) {
        int idx = dis(gen);
        for (int d = 0; d < point_dim; ++d) {
          sampled_points[i * point_dim + d] = points[idx * point_dim + d];
        }
      }
    }
    sampled_point_clouds.push_back(std::move(sampled_points));
  }
  return sampled_point_clouds;
}

///////////////////////// Main Function ///////////////////////////

int main(int argc, char *argv[]) {
  // used to save the final parameters
  std::string dir = argv[1];
  std::map<std::string, std::vector<float>> params;

  // load training data
  std::string file_path = "./data/train_point_clouds.h5";
  std::vector<std::vector<float>> list_of_points;
  std::vector<int> list_of_labels;
  read_h5_file(file_path, list_of_points, list_of_labels);

  // sampling the data
  int num_points = 256;
  auto sample_of_points = sample_point_clouds(list_of_points, num_points);

  // initialize the model
  int num_classes = 10;
  Classifier classifier(num_classes);
  classifier.copy_to_device();

  // set hyperparameters
  int num_epochs = 30;
  float learning_rate = 1e-2;
  int batch_size = 32;
  double previous_loss = std::numeric_limits<double>::max();
  double tolerance = 1e-4; // minimum improvement
  int no_improvement_count = 0;
  int patience = 3; // allow at most n epochs without improvement

  // start timing
  auto start = std::chrono::high_resolution_clock::now();

  int num_samples = list_of_labels.size();
  int num_batches = (num_samples + batch_size - 1) / batch_size;

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    double epoch_loss_acc = 0.0;

    for (int batch = 0; batch < num_batches; ++batch) {
      // calculate batch index [ , )
      int start_idx = batch * batch_size;
      int end_idx = std::min(start_idx + batch_size, num_samples);
      int current_batch_size = end_idx - start_idx;

      // assemble batch data
      std::vector<float> batch_input(current_batch_size * 3 * num_points);
      std::vector<int> batch_labels(current_batch_size);

      for (int i = 0; i < current_batch_size; ++i) {
        std::copy(sample_of_points[start_idx + i].begin(),
                  sample_of_points[start_idx + i].end(),
                  batch_input.begin() + num_points * 3 * i);
        batch_labels[i] = list_of_labels[start_idx + i];
      }

      // used for mini-batch timing
      auto mini_batch_start = std::chrono::high_resolution_clock::now();

      // convert batch input to channel first array
      float *d_origin_input;
      CUDA_CHECK(
          cudaMalloc(&d_origin_input, batch_input.size() * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_origin_input, batch_input.data(),
                            batch_input.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

      float *d_channel_first_input;
      CUDA_CHECK(cudaMalloc(&d_channel_first_input,
                            batch_input.size() * sizeof(float)));
      dim3 block(256);
      dim3 grid((num_points * current_batch_size + block.x - 1) / block.x);
      convert_to_channel_first_kernel<<<grid, block>>>(
          d_origin_input, d_channel_first_input, num_points,
          current_batch_size);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // model forward
      float *d_logits_output;
      classifier.forward(d_channel_first_input, d_logits_output,
                         current_batch_size, num_points);

      // softmax
      float *d_softmax_output;
      CUDA_CHECK(cudaMalloc(&d_softmax_output,
                            current_batch_size * num_classes * sizeof(float)));
      int softmax_threads = 256;
      int softmax_blocks = current_batch_size;
      softmax_kernel<<<softmax_blocks, softmax_threads,
                       softmax_threads * sizeof(float)>>>(
          d_logits_output, d_softmax_output, num_classes, current_batch_size);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // compute loss
      int *d_batch_labels;
      CUDA_CHECK(cudaMalloc(&d_batch_labels, sizeof(int) * current_batch_size));
      CUDA_CHECK(cudaMemcpy(d_batch_labels, batch_labels.data(),
                            sizeof(int) * current_batch_size,
                            cudaMemcpyHostToDevice));
      float *d_loss_output;
      CUDA_CHECK(
          cudaMalloc(&d_loss_output, current_batch_size * sizeof(float)));
      int block_size = 256;
      int num_blocks = (current_batch_size + block_size - 1) / block_size;
      cross_entropy_loss_kernel<<<num_blocks, block_size>>>(
          d_softmax_output, d_batch_labels, d_loss_output, current_batch_size,
          num_classes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // copy loss to host
      std::vector<float> h_loss(current_batch_size);
      CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss_output,
                            sizeof(float) * current_batch_size,
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < current_batch_size; ++i) {
        epoch_loss_acc += h_loss[i];
      }

      // loss backward
      float *d_loss_input_grad;
      CUDA_CHECK(cudaMalloc(&d_loss_input_grad,
                            current_batch_size * num_classes * sizeof(float)));
      cross_entropy_loss_backward_kernel<<<num_blocks, block_size>>>(
          d_softmax_output, d_batch_labels, d_loss_input_grad,
          current_batch_size, num_classes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // softmax backward
      float *d_softmax_input_grad;
      CUDA_CHECK(cudaMalloc(&d_softmax_input_grad,
                            current_batch_size * num_classes * sizeof(float)));
      softmax_backward_kernel<<<current_batch_size, 256, 256 * sizeof(float)>>>(
          d_loss_input_grad, d_softmax_output, d_softmax_input_grad,
          current_batch_size, num_classes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      printVector_GPU(d_softmax_input_grad,current_batch_size*10);
      // model backward
      float *d_model_input_grad;
      classifier.backward(d_softmax_input_grad, d_model_input_grad,
                          current_batch_size, num_points);

      // update parameters
      classifier.update_parameters(learning_rate);

      // free
      CUDA_CHECK(cudaFree(d_origin_input));
      classifier.free_cache();

      // check
      auto mini_batch_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> t = mini_batch_end - mini_batch_start;
      std::cout << t.count() << std::endl;

      printf("epoch: %02d - batch: %04d - loss: %f\n", epoch + 1, batch + 1,
             epoch_loss_acc);
    }
    // early stop
    double epoch_loss = epoch_loss_acc / num_samples;

    if (previous_loss - epoch_loss > tolerance) {
      no_improvement_count = 0;
    } else {
      no_improvement_count++;
      learning_rate *= 0.5; // lower the learning rate
    }

    if (no_improvement_count >= patience) {
      // std::cout << "Early stopping triggered at epoch " << epoch + 1
      //           << std::endl;
      break;
    }
    previous_loss = epoch_loss;
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << std::fixed << std::setprecision(4) << diff.count() << std::endl;

  // save
  classifier.copy_to_host(params);
  save_model_params_and_buffers_to_txt(params, dir);

  // exit
  CUDA_CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
