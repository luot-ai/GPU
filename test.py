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

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    # 设备张量矩阵指针
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            # 确定 title 坐标
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles


            # do regular gemm here
            # 此处进行常规 gemm
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)


            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)


            # go to the next tile by advancing NUM_SM
            # 通过增加 NUM_SM 来进入下一个 tile
            tile_idx += NUM_SM
        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

def group_gemm_fn(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)


    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]


    # note these are device tensors
    # 注意这些是设备张量
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable
    # 我们使用固定数量的 CTA（线程块），并且它是自动可调节的
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )


    return group_C


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

    offs_am = (blk_row + arange_M) % M
    offs_bn = (blk_col + arange_N) % N
    offs_bn_2D = offs_bn[None, :]
    mask_bn = offs_bn_2D < N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    batch_off_b = tl.where(isCF,0,batch * K *N)  
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + batch * M * K
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn_2D * stride_bn) + batch_off_b

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 初始化 accumulator
    if isCF == True :
        cvb_ptrs = cvb + offs_bn_2D
        cvb_line = tl.load(cvb_ptrs, mask=mask_bn)
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

# def bmm(a, b, batch_size,M,K,N):
#     c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
#     grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), batch_size,)
#     gemm_br_kernel[grid](
#         a, b, c,
#         M, N, K,
#         K,  1,
#         N,  1,
#         N,  1,
#         c,None,None,None,None,isCF=False, BN=False,RELU=False,
#     )
#     return c
def bmm(a, b, batch_size,M,K,N):
    c = group_gemm_fn(a, b)
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
    max_pool_out = maxPooling(CBR3_out, 1000, OC3, numPoints)
    fbr2f_out = FBR_2_F(max_pool_out,prefix,OC3,f1,f2,f3,off=3)
    feat = matrix_addI(fbr2f_out, 1000, IC)
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
    # dir = "./newparams/uniform/chfull/np128/30epoch-1000batch"
    # # 读取模型参数
    # params = read_params(dir, device='cuda')

    dir = os.path.dirname(__file__) # 保存模型参数文件(.txt)的文件夹路径

    # 读取模型参数
    params = read_params(dir,device='cuda')

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