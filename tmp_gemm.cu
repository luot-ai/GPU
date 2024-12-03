__global__ void gemm_64x64_128N_kernel(int M,int N,int K,
float* input_A, float* input_B, float* convBias,float* output, float beta = 0.0f)
{   
    // param-set : variable
    int BM = 64;
    int BN = 64;
    // param-set : fix
    int BK = 8;
    int Tsize = 4; //thread 4*4
    // matrix
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int bI = b * K * N ;
    int bW = b * K * M ;
    int bO = b * M * N ;
    // output arrange
    int warpIdx = tx / 32; // 4x8 threads per Warp
    int twIdx = tx % 32;
    int wx = warpIdx % 2;      // th -> 8x8  warp-> 32x64
    int wy = warpIdx / 2;      // 4x2 warps per Block
    int twx = (twIdx / 2) % 8; 
    int twy = (twIdx / 16) * 2 + (twIdx % 2);

    //shared memory & registers
    __shared__ float W_shared[512];//1024*4B = 4KB
    __shared__ float I_shared[512];//1024*4B = 4KB
    float W_reg[4]={0};
    float I_reg[4]={0};
    float O_reg[4][4] = {0};

    int W_grow = by * BM + tx / BK * 2; // 每BK个threads:连续2行读取1个数
    int W_gcol = 0 + tx % BK;
    int I_grow = 0 + tx / 32; // 32个threads读32个数，重复2次 刚好是一行:INTERLEAVE
    int I_gcol = bx * BN + tx % 32;
    int W_LoadG = INDEX(W_grow, W_gcol, K)+bW;
    int I_LoadG = INDEX(I_grow, I_gcol, N)+bI;
    // OUTERMOST PHASES: K/BK times
    for (int phase = 0; phase < K / BK; phase++)
    {
        //【global -> share】
        int W_srow = tx % BK ; //转置
        int W_scol = tx / BK * 2 ; 
        int I_srow = tx / 32; 
        int I_scol = tx % 32; 
        int W_StoreS = INDEX(W_srow,W_scol,BM);
        int I_StoreS = INDEX(I_srow,I_scol,BN);
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            W_shared[W_StoreS+ldg]=input_A[W_LoadG+ldg*K];
        }
        #pragma unroll
        for (int ldg = 0; ldg < 2; ldg++)
        {
            I_shared[I_StoreS+ldg*32]=input_B[I_LoadG+ldg*32];
        }
        __syncthreads();
        W_LoadG += BK;
        I_LoadG += BK * N;
        // ITERATIONS : BK times
        for (int iter = 0; iter< BK ;iter++)
        {
            //【share -> registers】
            int W_LoadS = INDEX(iter, (wy * 16 + twy * 4), BM);
            int I_LoadS = INDEX(iter, (wx * 32 + twx * 4), BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                W_reg[i] = W_shared[W_LoadS+i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                I_reg[i] = I_shared[I_LoadS+i];
            }
            // calculate
            #pragma unroll
            for (int i = 0; i < Tsize; ++i) {
                #pragma unroll
                for (int j = 0; j < Tsize; ++j) {
                    O_reg[i][j] += W_reg[i] * I_reg[j];
                }
            }
        }
    }
    int O_grow = by * BM + wy * 16 + twy * 4;
    int O_gcol = bx * BN + wx * 32 + twx * 4;
    int O_StoreG = INDEX(O_grow, O_gcol, N)+bO;
    //store to C
    if (beta!= 0.0f)
    {
        #pragma unroll
        for (int i = 0; i<4;i++)
        {
            for (int j = 0 ; j<4 ;j++)
            {
                output[O_StoreG+ i*N+j]+=O_reg[i][j];
            }
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i<4;i++)
        {
            for (int j = 0 ; j<4 ;j++)
            {
                output[O_StoreG+ i*N+j]=O_reg[i][j];
            }
        }
    }
}
