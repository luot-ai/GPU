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
    num_points = points.shape[0]
    if num_points <= num_sample:
        return points
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
    return list_of_points, list_of_labels   


#输入：numpoints,IC,OC 
#输出：blk_m,blk_n,blk_k,grp_size
def get_cf_params():
    return {
        (128, 3, 64): (32, 32, 32, 8),
        (128, 64, 128): (32, 32, 32, 8),
        (128, 128, 1024): (64, 128, 32, 8),
        (1, 1024, 512): (64, 128, 64, 8),
        (1, 512, 256): (64, 64, 64, 8),
        (1, 256, 9): (32, 32, 32, 4),
        (1, 256, 10): (32, 32, 32, 4),
        (1, 256, 4096): (128, 128, 64, 8),   
        (128, 64, 64): (64, 32, 32, 8), } 

#输入：batchsize, M, K, N 
#输出：blk_m, blk_n , blk_k, SM
def get_bmm_params():
    return {
        (1000, 128, 3, 3): (64, 64, 32, 300),  # 推测
        (1000, 128, 64, 64): (64, 64, 32, 350),
        (1000, 128, 3, 64): (64, 64, 32, 350),
        (1000, 128, 64, 128): (32, 32, 32, 350),
        (1000, 128, 128, 1024): (64, 128, 32, 350),
        (1000, 1, 1024, 512): (64, 128, 64, 350),
        (1000, 1, 512, 256): (64, 64, 64, 350),
        (1000, 1, 256, 9): (32, 32, 32, 350),
        (1000, 1, 256, 10): (32, 32, 32, 350),
        (1000, 1, 256, 4096): (128, 128, 64, 350),   
        (1000, 128, 64, 64): (64, 32, 32, 350),}

@triton.jit
def grouped_gemm_kernel(
    A,B,C,M,N,K,batchSize,
    lda,ldb,ldc,
    str_ak,str_bn,str_cn,
    bias,isCF,
    bnrm,bnrv,bnw,bnb,
    BN: tl.constexpr,
    RELU: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles
    for batch in range(batchSize):
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            A_ptr = A + batch * M * K
            B_ptr = B
            if isCF == False:
                B_ptr += batch * K * N
            C_ptr = C + batch * M * N
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_bn_2D = offs_bn[None, :]
            mask_bn = offs_bn_2D < N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            A_ptrs = A_ptr + offs_am[:, None] * lda + offs_k[None, :] * str_ak
            B_ptrs = B_ptr + offs_k[:, None] * ldb + offs_bn[None, :] * str_bn
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a=tl.load(A_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
                b=tl.load(B_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
                accumulator += tl.dot(a,b)
                A_ptrs += BLOCK_SIZE_K
                B_ptrs += BLOCK_SIZE_K * ldb
            if isCF == True:
                bias_vec = tl.load(bias + offs_bn_2D, mask=mask_bn)
                bias_matrix = tl.broadcast_to(bias_vec, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                accumulator += bias_matrix
            if BN == True:
                bnrm_line = tl.load(bnrm + offs_bn_2D, mask= mask_bn)
                bnrm_matrix = tl.broadcast_to(bnrm_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                accumulator -= bnrm_matrix

                bnrv_line = tl.load(bnrv + offs_bn_2D, mask=mask_bn)
                bnw_line = tl.load(bnw + offs_bn_2D, mask=mask_bn)
                gamma_line = bnw_line / tl.sqrt(bnrv_line + 1e-5)
                gamma = tl.broadcast_to(gamma_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                accumulator *= gamma 

                bnb_line = tl.load(bnb + offs_bn_2D, mask=mask_bn)
                bnb_matrix = tl.broadcast_to(bnb_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                accumulator += bnb_matrix
            if RELU == True:
                accumulator = relu(accumulator)
            c = accumulator
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = C_ptr + offs_cm[:, None] * ldc + offs_cn[None, :] * str_cn
            tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M and offs_cn[None, :] < N))
            tile_idx += NUM_SM
        last_problem_end += num_tiles


def bmm(A, B, batchSize, M, N, K):
    device = torch.device('cuda')
    B = B.reshape(batchSize, K, N) 
    assert A[0].shape[1] == B[0].shape[0]
    C = torch.empty((batchSize, M, N), device=device, dtype=torch.float32)
    grid = lambda META: (META['NUM_SM'], )
    (blk_m, blk_n, blk_k, NUM_SM) = get_bmm_params()[(batchSize, M, K, N)]
    
    grouped_gemm_kernel[grid](
        A,B,C,M,N,K,batchSize,
        K,N,N,1,1,1,
        C,False,
        C,C,C,C,
        BN=False,RELU=False,
        BLOCK_SIZE_M = blk_m,
        BLOCK_SIZE_K = blk_k,
        BLOCK_SIZE_N = blk_n,
        NUM_SM=NUM_SM
    )
    return C

# ReLU 激活函数实现
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)

@triton.jit
def br_kernel(
        c_ptr,
        M, N, K,
        stride_cm, stride_cn,
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
    batch = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    blk_row = pid_m * BLOCK_SIZE_M
    blk_col = pid_n * BLOCK_SIZE_N
    arange_M = tl.arange(0, BLOCK_SIZE_M)
    arange_N = tl.arange(0, BLOCK_SIZE_N)

    offs_bn = (blk_col + arange_N) % N
    offs_bn_2D = offs_bn[None, :]
    mask_bn = offs_bn_2D < N

    offs_cm = blk_row + arange_M
    offs_cn = blk_col + arange_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + batch * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    accumulator = tl.load(c_ptrs,c_mask)#tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if BN == True:
        bnrm_ptrs = bnrm + offs_bn_2D
        bnrm_line = tl.load(bnrm_ptrs, mask= mask_bn)
        bnrm_matrix = tl.broadcast_to(bnrm_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator -= bnrm_matrix

        bnrv_ptrs = bnrv + offs_bn_2D
        bnrv_line = tl.load(bnrv_ptrs, mask=mask_bn)
        bnw_ptrs = bnw + offs_bn_2D
        bnw_line = tl.load(bnw_ptrs, mask=mask_bn)
        gamma_line = bnw_line / tl.sqrt(bnrv_line + 1e-5)
        gamma = tl.broadcast_to(gamma_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator *= gamma 

        bnb_ptrs = bnb + offs_bn_2D
        bnb_line = tl.load(bnb_ptrs, mask=mask_bn)
        bnb_matrix = tl.broadcast_to(bnb_line, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator += bnb_matrix

    if RELU == True:
        accumulator = relu(accumulator)

    offs_cm = blk_row + arange_M
    offs_cn = blk_col + arange_N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + batch * M * N
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def gbr(a, b, batch_size, M, K, N, 
        cvb, bnw, bnb, bnrm, bnrv, 
        BN: tl.constexpr, RELU: tl.constexpr):
    device = torch.device('cuda')
    b = b.reshape(K, N) 
    if a.dim() == 2:
        a = a.unsqueeze(1)
    print(a.shape,b.shape)
    assert a[0].shape[1] == b.shape[0]
    g_res = torch.empty((batchSize, M, N), device=device, dtype=torch.float32)
    grid = lambda META: (META['NUM_SM'], )
    (blk_m, blk_n, blk_k, NUM_SM) = get_bmm_params()[(batchSize, M, K, N)]    
    grouped_gemm_kernel[grid](
        a,b,g_res,M,N,K,batchSize,
        K,1,N,1,K,1,
        cvb,True,
        bnrm,bnrv,bnw,bnb,
        BN=BN,RELU=RELU,
        BLOCK_SIZE_M = blk_m,
        BLOCK_SIZE_K = blk_k,
        BLOCK_SIZE_N = blk_n,
        NUM_SM=NUM_SM
    )

    # (blk_m, blk_n, blk_k, grp_size) = get_cf_params()[(M,K,N)]
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), batch_size,)
    # br_kernel[grid](
    #     g_res,
    #     M, N, K,
    #     N, 1,
    #     bnw, bnb, bnrm, bnrv, isCF=True, BN=BN, RELU=RELU, 
    #     BLOCK_SIZE_M = blk_m,
    #     BLOCK_SIZE_N = blk_n,
    #     BLOCK_SIZE_K = blk_k,
    #     GROUP_SIZE_M = grp_size,
    # )
    return g_res

@triton.jit
def maxPooling_kernel(
    x, res,
    channel, N,
    BLOCK_SIZE: tl.constexpr,
):
    curC = tl.program_id(axis=1)  
    batch = tl.program_id(axis=0)  
    bc = batch * channel
    b_off = bc * N
    #取数，这里的维度是B  N  C
    offs_n = tl.arange(0, BLOCK_SIZE) 
    x_idx = x + (b_off + offs_n * channel +curC)  
    x_vals = tl.load(x_idx)  
    #计算，这里的blk_size = numpoints
    max_val = tl.max(x_vals)  
    #存数，res的维度是B C
    tl.store(res + (bc + curC) , max_val)  
def maxPooling(x, batchsize, channel, N):
    max = torch.zeros((batchsize, channel), device=x.device, dtype=torch.float32)
    grid = (batchsize, channel)  
    maxPooling_kernel[grid](
        x, max, channel, N,
        BLOCK_SIZE=N,
    )
    return max

@triton.jit
def addI_kernel(x, I, res, channel:tl.constexpr):
    batch = tl.program_id(axis=0)  
    f_size = channel * channel
    b_off = batch * f_size
    if(channel == 3):
        I_off = tl.arange(0, 16)
    else:
        I_off = tl.arange(0, channel * channel)
    bi_off = b_off + I_off
    l_mask = I_off < f_size

    xM = tl.load(x + bi_off, mask= l_mask)
    IM = tl.load(I + I_off, mask= l_mask)
    resM = xM + IM
    tl.store(res + bi_off, resM, mask= l_mask )
def matrix_addI(x, batch_size, channel):
    res = torch.empty_like(x)
    I = torch.eye(channel, device='cuda', dtype=torch.float32)
    grid = lambda META: (batch_size, )  
    addI_kernel[grid](
        x, I, res, channel
    )
    return res

@triton.jit
def get_label_kernel(x, labels, batch_size: tl.constexpr, N: tl.constexpr):
    batch = tl.program_id(axis=0)
    offs =  tl.arange(0, 16)
    idx = x + batch * N + offs 
    data = tl.load(idx, mask = offs < N, other= -sys.float_info.max)
    label = tl.argmax(data,axis = 0)
    #存数
    tl.store(labels + batch, label)
def get_label(x, batch_size, N):
    labels = torch.empty(batch_size, device='cuda', dtype=torch.float32)
    grid = lambda META: (batch_size, )
    get_label_kernel[grid](x, labels, batch_size, N)
    return labels.cpu()

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
    res = gbr(x,params[cvw],batchSize,numPoints,IC,OC,params[cvb], params[bnw], params[bnb], params[bnrm], params[bnrv],True,relu)
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
    res = gbr(x,params[fcw],batchSize,1,IC,OC,params[fcb],params[bnw], params[bnb], params[bnrm], params[bnrv],True,True)
    return res

def FBR_2_F(x,prefix,IC,OC1,OC2,OC3,off):
    res1 = FBR(x,prefix,1,IC,OC1,off)
    res2 = FBR(res1,prefix,2,OC1,OC2,off)
    fcw = f"{prefix}fc3.weight"
    fcb = f"{prefix}fc3.bias"
    res3 = gbr(res2, params[fcw], batchSize, 1, OC2, OC3, params[fcb], None,None,None,None,False,False)
    return res3

def stnd(x,prefix,IC,OC1,OC2,OC3,f1,f2,f3):
    CBR3_out = CBR3(x,prefix,IC,OC1,OC2,OC3)
    max_pool_out = maxPooling(CBR3_out, batchSize, OC3, numPoints)
    fbr2f_out = FBR_2_F(max_pool_out,prefix,OC3,f1,f2,f3,off=3)
    feat = matrix_addI(fbr2f_out, batchSize, IC)
    return feat


def do_inference(list_of_points,list_of_labels): #请在本函数下使用triton实现推理操作
    input = torch.tensor(np.array(list_of_points), dtype=torch.float32, device='cuda')
    
    trans = stnd(input,"feat.stn.",IC,OC1,OC2,OC3,FC_OC1,FC_OC2,FC_OC3)
    stn3d_bmm = bmm(input,trans,batchSize,numPoints,IC,IC)
    cbr1_output = CBR(stn3d_bmm, "feat." , 1 , encoderIC1 , fstn_IC)

    trans_feat = stnd(cbr1_output,"feat.fstn.",fstn_IC,fstn_OC1,fstn_OC2,fstn_OC3,fstn_FC_OC1,fstn_FC_OC2,fstn_FC_OC3)
    stnkd = bmm(cbr1_output,trans_feat,batchSize,numPoints,fstn_IC,fstn_IC)
    cbr2_output = CBR(stnkd, "feat.", 2 , fstn_IC, encoderOC2)
    cbr3_output = CBR(cbr2_output, "feat.", 3 , encoderOC2, encoderOC3 , "norelu")
    feat_output = maxPooling(cbr3_output, batchSize, encoderOC3, numPoints)

    label_input = FBR_2_F(feat_output,"",encoderOC3,512,256,10,off=0)
    final_output = get_label(label_input, batchSize, 10)
    correct_num = 0

    for output, label in zip(final_output, list_of_labels):
        if output == label:
            correct_num += 1

    accuracy_rate = correct_num / 1000
    return accuracy_rate

if __name__ == '__main__':
    dir = "./newparams/uniform/chfull/np128/30epoch-1000batch"
    # 读取模型参数
    params = read_params(dir, device='cuda')

    # dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径

    # # 读取模型参数
    # params = read_params(dir,device='cuda')

    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points,list_of_labels = read_h5_file(dataPath)
    
    # 开始计时
    start = time.time()
    accuracy_rate = do_inference(list_of_points,list_of_labels)
    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")