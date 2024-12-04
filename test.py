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
numPoints  = 128
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

def uniform_sample(points, num_sample):
    # 假设每个点云是 Nx3 的矩阵 (N为点的个数，3为坐标)
    num_points = points.shape[0]
    
    if num_points <= num_sample:
        return points
    
    # 计算采样间隔
    sampled_indices = np.linspace(0, num_points - 1, num_sample).astype(int)
    sampled_points = points[sampled_indices]
    return sampled_points

def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            points = hf[k]["points"][:].astype(np.float32)
            points = uniform_sample(points, numPoints) # 均匀采样
            list_of_points.append(points)
            list_of_labels.append(hf[k].attrs["label"])
            # points = hf[k]["points"][:].astype(np.float32)
            # sampled_indices = np.linspace(0, points.shape[0] - 1, numPoints).astype(int)
            # sampled_points = points[sampled_indices]
            # list_of_points.append(sampled_points)  # 将点云数据展平为一维
            # list_of_labels.append(hf[k].attrs["label"])

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


# ReLU 激活函数实现
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_br_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        cvb,
        bnw, 
        bnb, 
        bnrm, 
        bnrv, 
        isCF,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        BN: tl.constexpr,
        RELU: tl.constexpr,
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
    batch_off_b = tl.where(isCF,0,pid_b * K *N)  
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + pid_b * M * K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) + batch_off_b

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 初始化 accumulator
    if isCF == True :
        offs_cvb = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        cvb_ptrs = cvb + offs_cvb[None, :]
        cvb_line = tl.load(cvb_ptrs, mask=offs_cvb[None, :] < N)
        cvb_matrix = tl.broadcast_to(cvb_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator += cvb_matrix

    # 迭代计算C矩阵的一个块。
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if BN == True:
        offs_bnrm = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bnrm_ptrs = bnrm + offs_bnrm[None, :]
        bnrm_line = tl.load(bnrm_ptrs, mask=offs_bnrm[None, :] < N)
        bnrm_matrix = tl.broadcast_to(bnrm_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator -= bnrm_matrix

        offs_bnrv = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bnrv_ptrs = bnrv + offs_bnrv[None, :]
        bnrv_line = tl.load(bnrv_ptrs, mask=offs_bnrv[None, :] < N)
        bnrv_matrix = tl.broadcast_to(bnrv_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator /= tl.sqrt(bnrv_matrix + 1e-5) 

        offs_bnw = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bnw_ptrs = bnw + offs_bnw[None, :]
        bnw_line = tl.load(bnw_ptrs, mask=offs_bnw[None, :] < N)
        bnw_matrix = tl.broadcast_to(bnw_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator *= bnw_matrix

        offs_bnb = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bnb_ptrs = bnb + offs_bnb[None, :]
        bnb_line = tl.load(bnb_ptrs, mask=offs_bnb[None, :] < N)
        bnb_matrix = tl.broadcast_to(bnb_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator += bnb_matrix
    if RELU == True:
        accumulator = relu(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_b * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
def gbr(a, b, batch_size, M, K, N, 
        cvb,
        bnw, 
        bnb, 
        bnrm, 
        bnrv, 
        BN: tl.constexpr,
        RELU: tl.constexpr):
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), batch_size,)
    gemm_br_kernel[grid](
        a, b, c,
        M, N, K,
        K,  1,
        1,  K,
        N,  1,
        cvb, bnw, bnb, bnrm, bnrv, isCF=True, BN=BN, RELU=RELU, 
    )
    return c


@triton.jit
def maxPooling_kernel(
    x_ptr, result_ptr,
    batchsize, channel, N,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(axis=0)  
    c = tl.program_id(axis=1)  

    offs_n = tl.arange(0, BLOCK_SIZE) 
    x = x_ptr + (b * channel * N + c + offs_n * channel)  
    x_vals = tl.load(x)  

    max_val = tl.max(x_vals)  

    result_ptr_b_c = result_ptr + (b * channel + c)  
    tl.store(result_ptr_b_c, max_val)  

def maxPooling(x, batchsize, channel, N, block_size=numPoints):
    result = torch.zeros((batchsize, channel), device=x.device, dtype=torch.float32)
    grid = (batchsize, channel)  
    maxPooling_kernel[grid](
        x, result, batchsize, channel, N,
        BLOCK_SIZE=block_size,
    )
    return result

@triton.jit
def addI_kernel(input_ptr, iden_ptr, output_ptr, channel:tl.constexpr):
    pid_b = tl.program_id(axis=0)  # 批次 ID
    if(channel == 3):
        offs_iden = tl.arange(0, 16)
    else:
        offs_iden = tl.arange(0, channel * channel)
    input_ptrs = input_ptr + pid_b * channel * channel + offs_iden
    input_matrix = tl.load(input_ptrs, mask= offs_iden < channel * channel)
    iden_ptrs = iden_ptr + offs_iden
    iden_matrix = tl.load(iden_ptrs, mask= offs_iden < channel * channel)
    output_matrix = input_matrix + iden_matrix
    output_ptrs = output_ptr + pid_b * channel * channel + offs_iden
    tl.store(output_ptrs, output_matrix, mask= offs_iden < channel * channel )

def matrix_addI(input_tensor, batch_size, channel):
    output_tensor = torch.empty_like(input_tensor)
    iden = torch.eye(channel, device='cuda', dtype=torch.float32)
    grid = lambda META: (batch_size, )  
    addI_kernel[grid](
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
    output_tensor = torch.empty(batch_size, device='cuda', dtype=torch.float32)
    grid = lambda META: (batch_size, )
    max_kernel[grid](input_tensor, output_tensor, batch_size, num_elements)
    return output_tensor.cpu()

def bmm(a, b, batch_size,M,K,N):
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), batch_size,)
    gemm_br_kernel[grid](
        a, b, c,
        M, N, K,
        K,  1,
        N,  1,
        N,  1,
        c,None,None,None,None,isCF=False, BN=False,RELU=False,
    )
    return c

def CBR(x,prefix,idx,IC,OC,activation="relu"):
    cvw = f"{prefix}conv{idx}.weight"
    cvb = f"{prefix}conv{idx}.bias"
    bnw = f"{prefix}bn{idx}.weight"
    bnb = f"{prefix}bn{idx}.bias"
    bnrm = f"{prefix}bn{idx}.running_mean"
    bnrv = f"{prefix}bn{idx}.running_var"
    relu = False
    if(activation == "relu"):
        relu = True
    res = gbr(x,params[cvw],1000,numPoints,IC,OC,params[cvb], params[bnw], params[bnb], params[bnrm], params[bnrv],True,relu)
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
    res = gbr(x,params[fcw],1000,1,IC,OC,params[fcb],params[bnw], params[bnb], params[bnrm], params[bnrv],True,True)
    return res

def FBR_2_F(x,prefix,IC,OC1,OC2,OC3,off):
    res1 = FBR(x,prefix,1,IC,OC1,off)
    res2 = FBR(res1,prefix,2,OC1,OC2,off)
    fcw = f"{prefix}fc3.weight"
    fcb = f"{prefix}fc3.bias"
    res3 = gbr(res2, params[fcw], 1000, 1, OC2, OC3, params[fcb], None,None,None,None,False,False)
    return res3

def stnd(x,prefix,IC,OC1,OC2,OC3,f1,f2,f3):
    CBR3_out = CBR3(x,prefix,IC,OC1,OC2,OC3)
    max_pool_out = maxPooling(CBR3_out, 1000, OC3, numPoints, block_size=numPoints)
    fbr2f_out = FBR_2_F(max_pool_out,prefix,OC3,f1,f2,f3,off=3)
    feat = matrix_addI(fbr2f_out, 1000, IC)
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
    feat_output = maxPooling(feat_conv_3_out, batchSize, encoderOC3, numPoints, block_size=numPoints)

    fc_3_out = FBR_2_F(feat_output,"",encoderOC3,512,256,10,off=0)
    final_output = compute_max(fc_3_out, batchSize, 10)
    correct_num = 0

    for output, label in zip(final_output, list_of_labels):
        if output == label:
            correct_num += 1

    accuracy_rate = correct_num / 1000
    return accuracy_rate

if __name__ == '__main__':
    dir = "./newparams/uniform/chfull/np128/30epoch-1000batch"
    #"./epoch_300_batchsize_1000/" 
    
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