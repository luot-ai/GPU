# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os
import sys
import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl

ch = 1024
ch_half = 512
ch_quarter = 256
numPoints  = 32
batchSize  = 1000
IC  = 3
OC1 = 64
OC2 = 128
OC3 = ch
FC_OC1 = ch_half
FC_OC2 = ch_quarter
FC_OC3 = 9
fstn_IC = 64
fstn_OC1 = 64
fstn_OC2 = 128
fstn_OC3 = ch
fstn_FC_OC1 = ch_half
fstn_FC_OC2 = ch_quarter
fstn_FC_OC3 = fstn_IC * fstn_IC 
encoderIC1 = IC
encoderOC2 = 128
encoderOC3 = ch

def read_params(dir, device='cuda'):
    # 列出所有txt文件
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]
    params = {}
    
    for fileName in files:
        data = []
        with open(os.path.join(dir, fileName), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                value = float(line)
                data.append(value)
        
        # 将数据转换为 PyTorch tensor 并移动到指定的设备（GPU或CPU）
        tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
        
        modelName = fileName.replace(".txt", "")
        params[modelName] = tensor_data
    
    return params

import h5py
import numpy as np

def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            # 读取点云数据
            points = hf[k]["points"][:].astype(np.float32)
        
            # 如果点云的点数超过32，均匀采样32个点
            sampled_indices = np.linspace(0, points.shape[0] - 1, 32).astype(int)
            sampled_points = points[sampled_indices]
            # 将采样后的点云数据添加到 list_of_points
            list_of_points.append(sampled_points)  # 将点云数据展平为一维

            # 获取标签并添加到 list_of_labels
            list_of_labels.append(hf[k].attrs["label"])

    return list_of_points, list_of_labels


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2)
    ]

# 1D 卷积+批归一化+激活内核
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_bn_ru_kernel(
        a_ptr, b_ptr, conv_bias, c_ptr,
        bn_weight, bn_bias, bn_mean, bn_var,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    """计算卷积和批归一化的内核"""
    
    # 获取程序 ID 用于计算分配的 C 块。
    bid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 为 A 和 B 的第一个块创建指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + bid * M * K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 初始化 accumulator
    offs_conv_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    conv_bias_ptrs = conv_bias + offs_conv_bias[None, :]
    conv_bias_line = tl.load(conv_bias_ptrs, mask=offs_conv_bias[None, :] < N)
    conv_bias_matrix = tl.broadcast_to(conv_bias_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator += conv_bias_matrix

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_bn_mean = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    bn_mean_ptrs = bn_mean + offs_bn_mean[None, :]
    bn_mean_line = tl.load(bn_mean_ptrs, mask=offs_bn_mean[None, :] < N)
    bn_mean_matrix = tl.broadcast_to(bn_mean_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator -= bn_mean_matrix

    offs_bn_var = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    bn_var_ptrs = bn_var + offs_bn_var[None, :]
    bn_var_line = tl.load(bn_var_ptrs, mask=offs_bn_var[None, :] < N)
    bn_var_matrix = tl.broadcast_to(bn_var_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator /= tl.sqrt(bn_var_matrix + 1e-5) 

    offs_bn_weight = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    bn_weight_ptrs = bn_weight + offs_bn_weight[None, :]
    bn_weight_line = tl.load(bn_weight_ptrs, mask=offs_bn_weight[None, :] < N)
    bn_weight_matrix = tl.broadcast_to(bn_weight_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator *= bn_weight_matrix

    offs_bn_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    bn_bias_ptrs = bn_bias + offs_bn_bias[None, :]
    bn_bias_line = tl.load(bn_bias_ptrs, mask=offs_bn_bias[None, :] < N)
    bn_bias_matrix = tl.broadcast_to(bn_bias_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator += bn_bias_matrix

    # 激活函数
    if ACTIVATION == "relu":
        accumulator = relu(accumulator)

    # 将结果存储到输出矩阵 C 中
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + bid * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# ReLU 激活函数实现
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)
# conv_bn_ru 主函数
def conv_bn_ru(a, b, cvb, bn_weight, bnb, bnm, bnv, batch_size, width, in_channels, out_channels, activation="relu"):
    # 分配输出 tensor，形状为 (width, out_channels)
    conv_bn_ru_output = torch.empty((batch_size, width, out_channels), device='cuda', dtype=torch.float32)

    # 1D 内核启动配置，计算 grid 大小
    grid = lambda META: (triton.cdiv(width, META['BLOCK_SIZE_M']) * triton.cdiv(out_channels, META['BLOCK_SIZE_N']), batch_size,)
    
    # 启动 Triton 内核
    conv_bn_ru_kernel[grid](
        a, b, cvb, conv_bn_ru_output,
        bn_weight, bnb, bnm, bnv,
        width, out_channels, in_channels,
        in_channels, 1,
        1, in_channels,
        out_channels, 1,
        ACTIVATION=activation
    )
    return conv_bn_ru_output

@triton.jit
def max_along_dim_kernel(
    x_ptr, result_ptr,
    batchsize, channel, N,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前线程的 ID
    b = tl.program_id(axis=0)  # 批量索引
    c = tl.program_id(axis=1)  # 通道索引

    # 每个线程负责 N 维度中的一个元素
    offs_n = tl.arange(0, BLOCK_SIZE)  # N 维度中的偏移量
    x = x_ptr + (b * channel * N + c + offs_n * channel)  # 每个线程计算 x 中的数据位置
    x_vals = tl.load(x)  # 加载数据

    # 使用 warp reduce 来计算最大值
    max_val = tl.max(x_vals)  # 在当前线程组内计算最大值

    # 将最大值写回到结果数组
    result_ptr_b_c = result_ptr + (b * channel + c)  
    tl.store(result_ptr_b_c, max_val)  

# 处理数据的主函数
def max_along_dim(x, batchsize, channel, N, block_size=32):
    result = torch.zeros((batchsize, channel), device=x.device, dtype=torch.float32)

    # 配置 Triton 内核的 grid 和 block
    grid = (batchsize, channel)  # 批次和通道维度的网格大小

    # 启动 Triton 内核
    max_along_dim_kernel[grid](
        x, result, batchsize, channel, N,
        BLOCK_SIZE=block_size,
    )

    return result

@triton.jit
def log_softmax_kernel(input_ptr, output_ptr, batchsize, features, BLOCK_SIZE: tl.constexpr):
    # Get the batch index (iter_batch)
    batch_index = tl.program_id(0)
    
    # The block range for each thread within a batch
    feature_index = tl.arange(0, BLOCK_SIZE)
    
    # Pointer to the starting location of the current batch in input/output tensors
    input_batch_ptr = input_ptr + batch_index * features
    output_batch_ptr = output_ptr + batch_index * features

    # Read the input values for the current batch
    input_values = tl.load(input_batch_ptr + feature_index)

    # Compute the sum of exponentials for the current batch
    exp_values = tl.exp(input_values)
    exp_sum = tl.sum(exp_values)

    # Compute log_softmax
    log_exp_values = tl.log(exp_values / exp_sum)
    
    # Write the result back to the output tensor
    tl.store(output_batch_ptr + feature_index, log_exp_values)
def log_softmax(input, batchsize, features):
    # Define the block size for processing
    BLOCK_SIZE = 1024  # Depending on your GPU architecture, adjust this value
    
    # Allocate output tensor
    output = input.new_zeros(input.shape, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (batchsize,)
    log_softmax_kernel[grid](input, output, batchsize, features, BLOCK_SIZE)
    
    return output

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def fc_kernel(
        a_ptr, b_ptr, conv_bias, c_ptr,
        batch_size, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    """计算卷积和批归一化的内核"""
    
    # 获取程序 ID 用于计算分配的 C 块。
    bid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 为 A 和 B 的第一个块创建指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + bid * M * K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 初始化 accumulator
    offs_conv_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    conv_bias_ptrs = conv_bias + offs_conv_bias[None, :]
    conv_bias_line = tl.load(conv_bias_ptrs, mask=offs_conv_bias[None, :] < N)
    conv_bias_matrix = tl.broadcast_to(conv_bias_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator += conv_bias_matrix


    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 将结果存储到输出矩阵 C 中
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + bid * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def fc(x, conv_weight, conv_bias, batch_size, width, in_channels, out_channels):
    # 分配输出 tensor，形状为 (width, out_channels)
    fc_output = torch.empty((batch_size, width, out_channels), device='cuda', dtype=torch.float32)

    # 1D 内核启动配置，计算 grid 大小
    grid = lambda META: (triton.cdiv(width, META['BLOCK_SIZE_M']) * triton.cdiv(out_channels, META['BLOCK_SIZE_N']), batch_size,)
    
    # 启动 Triton 内核
    fc_kernel[grid](
        x, conv_weight, conv_bias, fc_output,
        batch_size, width, out_channels, in_channels,
        in_channels, 1,
        1, in_channels,
        out_channels, 1
    )
    return fc_output

@triton.jit
def add_iden_kernel(input_ptr, iden_ptr, output_ptr, channel:tl.constexpr):
    pid_b = tl.program_id(axis=0)  # 批次 ID

    # 计算每个线程的偏移量并读取输入矩阵
    if(channel == 3):
        offs_iden = tl.arange(0, 16)
    else:
        offs_iden = tl.arange(0, channel * channel)
    input_ptrs = input_ptr + pid_b * channel * channel + offs_iden
    input_matrix = tl.load(input_ptrs, mask= offs_iden < channel * channel)

    # 读取身份矩阵
    iden_ptrs = iden_ptr + offs_iden
    iden_matrix = tl.load(iden_ptrs, mask= offs_iden < channel * channel)

    # 计算输出矩阵
    output_matrix = input_matrix + iden_matrix

    # 将结果存储到输出矩阵
    output_ptrs = output_ptr + pid_b * channel * channel + offs_iden
    tl.store(output_ptrs, output_matrix, mask= offs_iden < channel * channel )


def add_iden(input_tensor, batch_size, channel):
    """
    输入：形状为 (batch_size, channel, channel) 的输入张量
    输出：形状为 (batch_size, channel, channel) 的输出张量，已加上身份矩阵
    """
    # 分配输出 tensor
    output_tensor = torch.empty_like(input_tensor)

    # 创建身份矩阵，并将其存储到显存中
    iden = torch.eye(channel, device='cuda', dtype=torch.float32)

    # 1D 内核启动配置，计算 grid 大小
    grid = lambda META: (batch_size, )  # 每个批次的内核只需要一个线程

    # 启动 Triton 内核
    add_iden_kernel[grid](
        input_tensor, iden, output_tensor, channel
    )

    return output_tensor

@triton.jit
def max_kernel(input_ptr, output_ptr, batch_size: tl.constexpr, num_elements: tl.constexpr):
    # 获取当前程序 ID，表示处理的批次
    pid_b = tl.program_id(axis=0)

    # 每个线程负责计算一行的最大值
    # 每行包含 num_elements 个元素（在此案例中为 10）
    # 假设 input_ptr 是一个 (batch_size, num_elements) 形状的 tensor

    # 读取该批次的数据
    offs =  tl.arange(0, 16)
    input_ptrs = input_ptr + offs + pid_b * num_elements
    data = tl.load(input_ptrs, mask = offs < 10, other= -sys.float_info.max)

    # 使用 Triton 的 reduce_max 来求最大值
    max_index = tl.argmax(data,axis = 0)

    # 将结果存储到输出 tensor 中
    output_ptrs = output_ptr + pid_b
    tl.store(output_ptrs, max_index)

def compute_max(input_tensor, batch_size, num_elements):
    """
    输入：形状为 (batch_size, num_elements) 的输入张量
    输出：形状为 (batch_size,) 的输出张量，存储每行的最大值
    """
    # 分配输出 tensor
    output_tensor = torch.empty(batch_size, device='cuda', dtype=torch.float32)

    # 设置 Triton 内核启动配置
    grid = lambda META: (batch_size, )

    # 启动 Triton 内核
    max_kernel[grid](input_tensor, output_tensor, batch_size, num_elements)

    return output_tensor.cpu()

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + pid_b * M * K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) + pid_b * K * N

    # 迭代计算C矩阵的一个块。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_b * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
def bmm(a, b, batch_size,M,K,N):
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), batch_size,)
    bmm_kernel[grid](
        a, b, c,
        M, N, K,
        K,  1,
        N,  1,
        N,  1,
    )
    return c

def CBR(x,prefix,idx,IC,OC,activation="relu"):
    cvw = f"{prefix}conv{idx}.weight"
    cvb = f"{prefix}conv{idx}.bias"
    bnw = f"{prefix}bn{idx}.weight"
    bnb = f"{prefix}bn{idx}.bias"
    bnrm = f"{prefix}bn{idx}.running_mean"
    bnrv = f"{prefix}bn{idx}.running_var"
    res = conv_bn_ru(x, params[cvw], params[cvb], params[bnw], params[bnb], params[bnrm], params[bnrv], 1000, 32, IC, OC, activation)
    return res

def CBR3(x,prefix,IC,OC1,OC2,OC3):
    res1 = CBR(x,prefix,1,IC,OC1)
    res2 = CBR(res1,prefix,2,OC1,OC2)
    res3 = CBR(res2,prefix,3,OC2,OC3)
    return res3

def FBR(x,prefix,idx,IC,OC,off):
    bnidx = idx + off
    fcw = f"{prefix}fc{idx}.weight"
    fcb = f"{prefix}fc{idx}.bias"
    bnw = f"{prefix}bn{bnidx}.weight"
    bnb = f"{prefix}bn{bnidx}.bias"
    bnrm = f"{prefix}bn{bnidx}.running_mean"
    bnrv = f"{prefix}bn{bnidx}.running_var"
    res = conv_bn_ru(x, params[fcw], params[fcb], params[bnw], params[bnb], params[bnrm], params[bnrv], 1000, 1, IC, OC, activation="relu")
    return res

def FBR_2_F(x,prefix,IC,OC1,OC2,OC3,off):
    res1 = FBR(x,prefix,1,IC,OC1,off)
    res2 = FBR(res1,prefix,2,OC1,OC2,off)
    fcw = f"{prefix}fc3.weight"
    fcb = f"{prefix}fc3.bias"
    res3 = fc(res2, params[fcw], params[fcb], 1000, 1, OC2, OC3)
    return res3

def stnd(x,prefix,IC,OC1,OC2,OC3,f1,f2,f3):
    CBR3_out = CBR3(x,prefix,IC,OC1,OC2,OC3)
    max_pool_out = max_along_dim(CBR3_out, 1000, OC3, 32, block_size=32)
    fbr2f_out = FBR_2_F(max_pool_out,prefix,OC3,f1,f2,f3,off=3)
    feat = add_iden(fbr2f_out, 1000, IC)
    return feat


def do_inference(list_of_points,list_of_labels,params): #请在本函数下使用triton实现推理操作
    model_input = torch.tensor(np.array(list_of_points), dtype=torch.float32, device='cuda')
    
    trans = stnd(model_input,"feat.stn.",IC,OC1,OC2,OC3,FC_OC1,FC_OC2,FC_OC3)
    x_mul_trans = bmm(model_input,trans,batchSize,numPoints,IC,IC)
    feat_conv_1_out = CBR(x_mul_trans, "feat." , 1 , encoderIC1 , fstn_IC)

    trans_feat = stnd(feat_conv_1_out,"feat.fstn.",fstn_IC,fstn_OC1,fstn_OC2,fstn_OC3,fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3)
    x_mul_trans_feat = bmm(feat_conv_1_out,trans_feat,batchSize,numPoints,fstn_IC,fstn_IC)
    feat_conv_2_out = CBR(x_mul_trans_feat, "feat.", 2 , fstn_IC, encoderOC2)
    feat_conv_3_out = CBR(feat_conv_2_out, "feat.", 3 , encoderOC2, encoderOC3 , "norelu")
    feat_output = max_along_dim(feat_conv_3_out, batchSize, encoderOC3, numPoints, block_size=32)

    fc_3_out = FBR_2_F(feat_output,"",encoderOC3,512,256,10,off=0)
    final_output = compute_max(fc_3_out, batchSize, 10)
    correct_num = 0

    for output, label in zip(final_output, list_of_labels):
        if output == label:
            correct_num += 1

    accuracy_rate = correct_num / 1000
    return accuracy_rate

if __name__ == '__main__':
    dir = "./epoch_300_batchsize_1000/" 
    
    # 读取模型参数
    params = read_params(dir, device='cuda')

    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points,list_of_labels = read_h5_file(dataPath)
    
    # 开始计时
    start = time.time()
    accuracy_rate = do_inference(list_of_points,list_of_labels,params)
    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")