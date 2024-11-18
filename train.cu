// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include<malloc.h>
#include <random>
#include <cmath>
#include <ctime>

#define MAX(i, j) (((i) > (j)) ? (i) : (j))
#define MIN(i, j) (((i) > (j)) ? (j) : (i))
#define TILE_WIDTH 12
#define MASK_WIDTH 5
#define POOL_WIDTH 2
#define BATCH_SIZE 1 //训练时因为存在bp，需要逐张过网络，因此该项需要为1
#define Max_Conv_Input 6
#define Max_Conv_Output 16
#define Max_FC_Input 256
#define Max_Matrix_Width 28
#define Max_Para_num 30720
#define EPOCH 1
__constant__ float mask[MASK_WIDTH * MASK_WIDTH * 6 *16];
__constant__ float FCN[256];
__constant__ float NMS[28 * 28];

using namespace std;


__device__ volatile int g_mutex;
float * D;

// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
                ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
                ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

void writeFloatsToFile(std::string filename, float* numbers, int count) {  
    std::ofstream file(filename);  
  
    if (file.is_open()) {  
        for (int i = 0; i < count; i++) {  
            file << numbers[i] << std::endl;  
        }  
        file.close();  
        //std::cout << filename << " file written" << std::endl;
    } else {  
        //std::cout << "file create fail" << std::endl;  
    }  
}  

// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}


__global__ void Conv2D_convKernel_new(float* M,float* N, float* P, int M_Width, int K_Width,int input_Width,int output_Width)
{
	__shared__ float ds_N[BATCH_SIZE][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
    __shared__ float ds_M[MASK_WIDTH][MASK_WIDTH];
    int offset = (MASK_WIDTH - 1) / 2;
	int bx = blockIdx.x; int by = blockIdx.y;int bz = blockIdx.z;
	int tx = threadIdx.x; int ty = threadIdx.y;int tz = threadIdx.z;
	int Row_o = by * TILE_WIDTH + ty;
	int Col_o = bx * TILE_WIDTH + tx;
	int Row_i = Row_o - offset;
	int Col_i = Col_o - offset;
    int matrix_i = bz/(output_Width);
    int matrix_o = bz%(output_Width);
	float Pvalue = 0;
	if ((Row_i >= 0) && (Row_i < M_Width) &&(Col_i >= 0) && (Col_i < K_Width)) {
		ds_N[tz][ty][tx] = N[tz*(input_Width*M_Width*K_Width)+matrix_i * M_Width * K_Width + Row_i * K_Width +Col_i];
	}else {
		ds_N[tz][ty][tx] = 0.0f;
	}
    if(ty < MASK_WIDTH && tx <MASK_WIDTH && tz == 0){
        ds_M[ty][tx] = M[(matrix_o * input_Width + matrix_i) * MASK_WIDTH * MASK_WIDTH +  ty* MASK_WIDTH + tx];
    }
	__syncthreads();
	if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
		for (int i = 0; i < MASK_WIDTH; i++) {
			for (int j = 0; j < MASK_WIDTH; j++) {
				Pvalue += ds_M[i][j]* ds_N[tz][i + ty][j + tx];
			}
		}
	}
	//__syncthreads();

	if (Row_o < (M_Width-offset) && Col_o < (K_Width-offset) && Row_o < (by+1) * TILE_WIDTH && Col_o < (bx + 1) * TILE_WIDTH && Row_o && Row_o >= offset && Col_o >= offset){
		int add = (Row_o - offset) * (K_Width-offset-offset) + Col_o - offset;
        int new_size = (M_Width-offset-offset) * (K_Width-offset-offset);
        P[tz*(output_Width*input_Width*new_size)+(matrix_i * output_Width + matrix_o) * new_size + add] = Pvalue;
    }
    //__syncthreads();
	//__threadfence();
 
}

__global__ void Conv2D_addKernel_new(float*N,float* P,int M_Width, int K_Width,int input_Width,int output_Width,float * bias,bool relu = true)
{
    __shared__ float ds_N[BATCH_SIZE][6];
	int bx = blockIdx.x; int by = blockIdx.y;int bz = blockIdx.z;
	int tx = threadIdx.x; int tz =threadIdx.z;
	int Row = by;
    int Col = bx;
    int output_index = bz;
    int input_index = tx;
    
    ds_N[tz][input_index] = N[tz*(M_Width*K_Width*output_Width*input_Width)+input_index*(M_Width*K_Width*output_Width)+output_index*(M_Width*K_Width)+Row*K_Width+Col];
    __syncthreads();
    if(tx == 0){
        float Pvalue = ds_N[tz][0];
        for(int i = 1;i<input_Width;++i){
            Pvalue += ds_N[tz][i];
        }
        Pvalue += bias[output_index];
        P[tz*(output_Width*M_Width*K_Width)+output_index*(M_Width*K_Width)+Row*K_Width+Col] = (relu && Pvalue <0)?0:Pvalue;
    }
    //__syncthreads();
    
    
}

void Conv2D_group_new(float* M, float* N, float* P, int M_Width, int K_Width,int input_Width,int output_Width,float* bias, bool nopadding,bool relu) {
    int offset = (MASK_WIDTH-1)/2;
	

	int size_grid_x = (K_Width - 1) / TILE_WIDTH + 1;
	int size_grid_y = (M_Width - 1) / TILE_WIDTH + 1;
	int block_size = TILE_WIDTH + MASK_WIDTH - 1;

	dim3 DimGrid(size_grid_x, size_grid_y,input_Width * output_Width * BATCH_SIZE);
	dim3 DimBlock(block_size, block_size, 1);
    
	Conv2D_convKernel_new << <DimGrid, DimBlock >> > (M,N, D, M_Width, K_Width,input_Width,output_Width);

	//cudaDeviceSynchronize();
    dim3 DimGrid1(K_Width-offset-offset, M_Width-offset-offset, output_Width*BATCH_SIZE);
	dim3 DimBlock1(input_Width, 1, 1);

    Conv2D_addKernel_new << <DimGrid1, DimBlock1 >> > (D,P, M_Width-offset-offset, K_Width-offset-offset,input_Width,output_Width,bias,relu);
    //cudaDeviceSynchronize();
	//cudaFree(D);
}

__global__ void Conv2D_spin(float* I,float* O,int input_Width,int output_Width){
    int ty = threadIdx.y;int tx = threadIdx.x;
    int bx = blockIdx.x;int by = blockIdx.y;

    int y_o = MASK_WIDTH-1-ty;
    int x_o = MASK_WIDTH-1-tx;

    O[bx*output_Width*MASK_WIDTH*MASK_WIDTH+by*MASK_WIDTH*MASK_WIDTH+y_o*MASK_WIDTH+x_o] = I[by*input_Width*MASK_WIDTH*MASK_WIDTH+bx*MASK_WIDTH*MASK_WIDTH+ty*MASK_WIDTH+tx];
    __syncthreads();
}

__global__ void Conv2D_padding(float* I,float* O,float* ref,int M_Width,int K_Width,int Pad_Width,int input_Width,bool relu){
    int ty = threadIdx.y;int tx = threadIdx.x;
    int bx = blockIdx.x;int bz = blockIdx.z;
    int output_M_Width = M_Width+Pad_Width+Pad_Width;
    int output_K_Width = K_Width+Pad_Width+Pad_Width;
    int output_size = output_M_Width*output_K_Width;
    int input_size = M_Width*K_Width;
    int output = ref[bz*input_Width*input_size+bx*input_size+(ty)*K_Width+(tx)];
    bool drop = relu && output==0;
    O[bz*input_Width*output_size+bx*output_size+(ty+Pad_Width)*output_K_Width+(tx+Pad_Width)]=drop?0:I[bz*input_Width*input_size+bx*input_size+(ty)*K_Width+(tx)];
    __syncthreads();
}

__global__ void Conv2D_weight(float* I,float* neuloss,float* ref,float*weight_loss,int M_Width,int K_Width,int input_Width,bool relu){
    __shared__ float d_neuloss[Max_Matrix_Width][Max_Matrix_Width];
    __shared__ float d_I[Max_Matrix_Width][Max_Matrix_Width];
    int ty = threadIdx.y;int tx = threadIdx.x;
    int bx = blockIdx.x;int by = blockIdx.y;int bz = blockIdx.z;
    int matrix_i = bz%input_Width;
    int matrix_o = bz/input_Width;

    int y_i = ty+by;
    int x_i = tx+bx;

    int input_M_Width = M_Width+MASK_WIDTH-1;
    int input_K_Width = K_Width+MASK_WIDTH-1;
    int input_size = input_M_Width*input_K_Width;

    d_neuloss[ty][tx] = (relu && ref[matrix_o*M_Width*K_Width+ty*K_Width+tx]==0)?0:neuloss[matrix_o*M_Width*K_Width+ty*K_Width+tx]; 
    d_I[ty][tx] = I[matrix_i*input_size+y_i*input_K_Width+x_i];
    __syncthreads();

    if(tx==0 && ty==0){
        float Pvalue = 0;
        for(int i =0;i<M_Width;++i){
            for(int j=0;j<K_Width;++j){
                Pvalue += d_neuloss[i][j]*d_I[i][j];
            }
        }
        weight_loss[matrix_o*input_Width*MASK_WIDTH*MASK_WIDTH+matrix_i*MASK_WIDTH*MASK_WIDTH+by*MASK_WIDTH+bx]=Pvalue;
    }

    //__syncthreads();
}

__global__ void Conv2D_bias(float* neuloss,float* ref,float*bias_loss,int M_Width,int K_Width,int input_Width,bool relu){
    __shared__ float d_neuloss[Max_Matrix_Width][Max_Matrix_Width];
    int ty = threadIdx.y;int tx = threadIdx.x;
    int bx = blockIdx.x;
    d_neuloss[ty][tx]=(relu && ref[bx*M_Width*K_Width+ty*K_Width+tx]==0)?0:neuloss[bx*M_Width*K_Width+ty*K_Width+tx];
    __syncthreads();
    if(tx==0 && ty==0){
        float Pvalue = 0;
        for(int i =0;i<M_Width;++i){
            for(int j=0;j<K_Width;++j){
                Pvalue += d_neuloss[i][j];
            }
        }
        bias_loss[bx]=Pvalue;
    }
}

void Conv2D_bp(float* I,float *O,float* weight,float * next_Neu_loss, float * Neu_loss, float* Weight_loss, float* Bias_loss,int M_Width,int K_Width,int input_Width,int output_Width,bool relu) {
    int input_M_Width = M_Width+MASK_WIDTH-1+MASK_WIDTH-1;
    int input_K_Width = K_Width+MASK_WIDTH-1+MASK_WIDTH-1;
    float * O_loss_padding;
    cudaMalloc((void**)&O_loss_padding,BATCH_SIZE*output_Width*input_M_Width*input_K_Width*sizeof(float));
    cudaMemset(O_loss_padding,0,BATCH_SIZE*output_Width*input_M_Width*input_K_Width*sizeof(float));

    dim3 DimGridpad(output_Width, 1, BATCH_SIZE);
	dim3 DimBlockpad(K_Width, M_Width, 1);
    Conv2D_padding<< <DimGridpad, DimBlockpad >> >(next_Neu_loss,O_loss_padding,O,M_Width,K_Width,MASK_WIDTH-1,output_Width,true);

    float* weight_spin;
    cudaMalloc((void**)&weight_spin,BATCH_SIZE*output_Width*input_Width*MASK_WIDTH*MASK_WIDTH*sizeof(float));
    dim3 DimGridspin(input_Width, output_Width, 1);
	dim3 DimBlockspin(MASK_WIDTH, MASK_WIDTH, 1);
    Conv2D_spin<< <DimGridspin, DimBlockspin >> >(weight,weight_spin,input_Width,output_Width);

    float* zero;
    cudaMalloc((void**)&zero,BATCH_SIZE*output_Width*input_Width*sizeof(float));
    cudaMemset(zero,0,BATCH_SIZE*output_Width*input_Width*sizeof(float));
    //calculate neuloss
	Conv2D_group_new(weight_spin,O_loss_padding,Neu_loss,input_M_Width,input_K_Width,output_Width,input_Width,zero,false,false);

    //calculate weightloss
    dim3 DimGridweight(MASK_WIDTH, MASK_WIDTH, input_Width*output_Width);
	dim3 DimBlockweight(K_Width, M_Width, 1);

    Conv2D_weight<< <DimGridweight, DimBlockweight >> >(I,next_Neu_loss,O,Weight_loss,M_Width,K_Width,input_Width,relu);

    //calculate biasloss
    dim3 DimGridbias(output_Width, 1, 1);
	dim3 DimBlockbias(K_Width, M_Width, 1);

    Conv2D_bias<<<DimGridbias, DimBlockbias >>>(next_Neu_loss,O,Bias_loss,M_Width,K_Width,output_Width,relu);

    cudaFree(zero);
    cudaFree(O_loss_padding);
    cudaFree(weight_spin);

}

__global__ void MaxPool_groupKernel_new(float* N, float* P, int M_Width, int K_Width,int input_Width)
{
	__shared__ float ds_N[POOL_WIDTH][POOL_WIDTH][Max_Conv_Output];
	int bx = blockIdx.x; int by = blockIdx.y;int bz=blockIdx.z;
	int tx = threadIdx.x; int ty = threadIdx.y;int tz =threadIdx.z;
	int Row_o = by;
	int Col_o = bx;
    int index = tz;
	int Row_i = by * POOL_WIDTH + ty;
	int Col_i = bx * POOL_WIDTH + tx;
	ds_N[ty][tx][index] = N[bz*input_Width*K_Width*M_Width+index * (K_Width * M_Width) + Row_i * K_Width + Col_i];

	__syncthreads();
	float Pvalue = ds_N[ty][tx][index];
	if (ty == 0 && tx == 0) {
		for (int i = 0; i < POOL_WIDTH; i++) {
			for (int j = 0; j < POOL_WIDTH; j++) {
				Pvalue = Pvalue > ds_N[i][j][index] ?  Pvalue:ds_N[i][j][index];
			}
		}
	}
	//__syncthreads();
	if (ty == 0 && tx == 0)
		P[bz * input_Width*(K_Width * M_Width/POOL_WIDTH/POOL_WIDTH)+index * (K_Width * M_Width/POOL_WIDTH/POOL_WIDTH)+Row_o * (K_Width/POOL_WIDTH) + Col_o] = Pvalue;
}

void MaxPool_group_new(float* N, float* P, int M_Width, int K_Width,int input_Width) {

	int size_grid_x = K_Width / POOL_WIDTH;
	int size_grid_y = M_Width / POOL_WIDTH;
	int block_size = POOL_WIDTH;

	dim3 DimGrid(size_grid_x, size_grid_y, BATCH_SIZE);
	dim3 DimBlock(block_size, block_size, input_Width);

	MaxPool_groupKernel_new << <DimGrid, DimBlock >> > (N, P, M_Width, K_Width,input_Width);
}
__global__ void MaxPool_bp_kernal(float* N,float * next_Neu_loss, float * Neu_loss,int M_Width,int K_Width,int input_Width)
{
	__shared__ float ds_N[POOL_WIDTH][POOL_WIDTH][Max_Conv_Output];
    __shared__ float ds_y[Max_Conv_Output];
    __shared__ float ds_x[Max_Conv_Output];
	int bx = blockIdx.x; int by = blockIdx.y;int bz=blockIdx.z;
	int tx = threadIdx.x; int ty = threadIdx.y;int tz =threadIdx.z;
	int Row_o = by;
	int Col_o = bx;
    int index = tz;
	int Row_i = by * POOL_WIDTH + ty;
	int Col_i = bx * POOL_WIDTH + tx;
	ds_N[ty][tx][index] = N[bz*input_Width*K_Width*M_Width+index * (K_Width * M_Width) + Row_i * K_Width + Col_i];

	__syncthreads();
	float Pvalue = ds_N[ty][tx][index];
    float block_num = M_Width *K_Width/POOL_WIDTH/POOL_WIDTH;
	if (ty == 0 && tx == 0) {
        ds_y[index] = 0;
        ds_x[index] = 0;
		for (int i = 0; i < POOL_WIDTH; i++) {
			for (int j = 0; j < POOL_WIDTH; j++) {
                if(Pvalue < ds_N[i][j][index]){
                    Pvalue = ds_N[i][j][index];
                    ds_y[index] = i;
                    ds_x[index] =j;
                }
			}
		}
	}
	__syncthreads();

    if(ds_y[index]==ty && ds_x[index] == tx){
        Neu_loss[bz*input_Width*K_Width*M_Width+index * (K_Width * M_Width) + Row_i * K_Width + Col_i] = float(1)/block_num * next_Neu_loss[bz * input_Width*int(block_num)+index * int(block_num)+Row_o * (K_Width/POOL_WIDTH) + Col_o];
    }else{
        Neu_loss[bz*input_Width*K_Width*M_Width+index * (K_Width * M_Width) + Row_i * K_Width + Col_i] = 0;
    }
    //__syncthreads();
}

void MaxPool_bp(float* I,float * next_Neu_loss, float * Neu_loss,int M_Width,int K_Width,int input_Width) {

	int size_grid_x = K_Width / POOL_WIDTH;
	int size_grid_y = M_Width / POOL_WIDTH;
	int block_size = POOL_WIDTH;

	dim3 DimGrid(size_grid_x, size_grid_y, BATCH_SIZE);
	dim3 DimBlock(block_size, block_size, input_Width);

	MaxPool_bp_kernal << <DimGrid, DimBlock >> > (I, next_Neu_loss,Neu_loss,M_Width, K_Width,input_Width);
}

__global__ void FCKernel_new(float* N,float* P,float* W,float* B, int M_Width, int K_Width, bool relu = true)
{
    __shared__ float ds_N[Max_FC_Input];
	int bx = blockIdx.x; 
	int tx = threadIdx.x; 
	int bz = blockIdx.z;
	ds_N[tx] = N[bz * M_Width + tx] * W[M_Width * bx+tx];
	__syncthreads();
	if (tx == 0){
        float Pvalue = 0;
        for(int i = 0;i <M_Width;++i){
            Pvalue += ds_N[i];
        }
        Pvalue += B[bx];
        if(relu){
            Pvalue = Pvalue < 0? 0:Pvalue;
        }else{
			//Pvalue = tanh(Pvalue);
		}
        P[bz*K_Width+ bx] = Pvalue;
    }
}

void FC_new(float* N, float* P,float* W,float* B, int M_Width, int K_Width, bool relu = true) {

	dim3 DimGrid(K_Width, 1, BATCH_SIZE);
	dim3 DimBlock(M_Width, 1, 1);

	FCKernel_new << <DimGrid, DimBlock >> > (N,P,W,B, M_Width, K_Width,relu);
}

__global__ void FC_bp_neu_kernal(float* I,float *O,float* weight, float * next_Neu_loss,float * Neu_loss, float* Weight_loss, float* Bias_loss,int input_Width,int output_Witdh,bool relu)
{
	__shared__ float ds_weight[Max_FC_Input];
	__shared__ float ds_next_loss[Max_FC_Input];
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	float nextneuloss = next_Neu_loss[tx];
    float output = O[tx];
	ds_next_loss[tx]=(relu&& output == 0)?0:nextneuloss;
	ds_weight[tx] = weight[tx*input_Width+bx];
	__syncthreads();
	
	if(tx == 0){
		//calculate neuloss
		float neuloss = ds_next_loss[0] * ds_weight[0];
		for(int i =1;i<output_Witdh;++i){
			neuloss += ds_next_loss[i] * ds_weight[i];
		}
		Neu_loss[bx] = neuloss;

	}
	__syncthreads();

	Weight_loss[input_Width * tx + bx] = ds_next_loss[tx] * I[bx];
    __syncthreads();
    if(bx == 0){
	    Bias_loss[tx] = ds_next_loss[tx];
    }
    //__syncthreads();
}

void FC_bp(float* I,float *O,float* weight,float * next_Neu_loss, float * Neu_loss, float* Weight_loss, float* Bias_loss,int input_Width,int output_Witdh,bool relu) {

	dim3 DimGrid0(input_Width, 1, 1);
	dim3 DimBlock0(output_Witdh, 1, 1);

	FC_bp_neu_kernal << <DimGrid0, DimBlock0 >> > (I,O,weight,next_Neu_loss,Neu_loss,Weight_loss,Bias_loss,input_Width,output_Witdh,relu);
}


__global__ void FC_final_bp_kernal(float* I,int label,float* Neu_loss, int Width)
{
    __shared__ float ds_N[Max_FC_Input];
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	ds_N[tx] = I[tx];
	__syncthreads();
    
	if(tx == 0){
		float P = exp(ds_N[0]);
		for(int i = 1;i<Width;++i)
			P += exp(ds_N[i]);
		P = exp(ds_N[bx])/P;
		Neu_loss[bx] = P- ((bx ==  label)?1:0);
	}
    
	//__syncthreads();
}

void FC_final_bp(float* I,int label,float* Neu_loss, int Width) {

	dim3 DimGrid(Width, 1, 1);
	dim3 DimBlock(Width, 1, 1);

	FC_final_bp_kernal << <DimGrid, DimBlock >> > (I,label,Neu_loss,Width);
}

__global__ void NMKernel_new(float*N,float* P,int M_Width, int K_Width)
{
	int bx = blockIdx.x; 
	int by = blockIdx.y; 
	int bz = blockIdx.z;
	P[bz * (M_Width * K_Width) + by * K_Width + bx] = (N[bz * (M_Width * K_Width) + by * K_Width + bx] - 0.5)/0.5;
	//__syncthreads();
}
void Normalize_new(float* N,float* P,int  M_Width, int K_Width) {
	dim3 DimGrid(K_Width, M_Width, BATCH_SIZE);
	dim3 DimBlock(1, 1, 1);

	NMKernel_new << <DimGrid, DimBlock >> > (N,P, M_Width, K_Width);
}

__global__ void FC_final_newKernal(float*N,int* label,int* table,int group_width, int group_num)
{
    __shared__ float ds_N[128];
	int bx = blockIdx.x; 
	int tx = threadIdx.x; 
    ds_N[tx] = N[bx*group_width+tx];
    __syncthreads();
	if(tx == 0){
        float tmp = ds_N[0];
        int flag = 0;
        for(int i =1;i<group_width;++i){
            if(ds_N[i]>tmp){
                tmp = ds_N[i];
                flag = i;
            }
        }
        table[bx] = flag == label[bx] ? 1:0;
    }
	//__syncthreads();
}
void FC_final_new(float* N,int* label,int*table,int group_width, int group_num) {
	dim3 DimGrid(group_num, 1, 1);
	dim3 DimBlock(group_width, 1, 1);
	FC_final_newKernal << <DimGrid, DimBlock >> > (N,label,table,group_width, group_num);
}

__global__ void BP_UPDATE_Kernal(float*N,float*delta,int width,float learning_rate){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int index = bx*(TILE_WIDTH*TILE_WIDTH)+ty*TILE_WIDTH+tx;

    if(index<width){
        N[index] = N[index]+delta[index]*learning_rate;
    }
}

void BP_UPDATE(float*N,float*delta,int width,float learning_rate){
    dim3 DimGrid((width-1)/(TILE_WIDTH*TILE_WIDTH)+1, 1, 1);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    BP_UPDATE_Kernal<<<DimGrid,DimBlock>>>(N,delta,width,learning_rate);
}

void para_init(float*N,int width,float init){
    uniform_real_distribution<double> u(-1*init, init);
    default_random_engine e(time(NULL));

    float rand[Max_Para_num]={0};
    for(int i = 0;i<width;++i){
        rand[i] = u(e);
    }

    cudaMemcpy(N,rand,width*sizeof(float),cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]) {

	std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
	// cout << dir;
	
    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
    auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/train-images-idx3-ubyte");
    // 读取训练集标签
    auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/train-labels-idx1-ubyte");
    // 读取模型参数
    auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    auto conv1_bias = read_param(dir + "/conv1.bias.txt");
    auto conv2_weight = read_param(dir + "/conv2.weight.txt");
    auto conv2_bias = read_param(dir + "/conv2.bias.txt");
    auto fc1_weight = read_param(dir + "/fc1.weight.txt");
    auto fc1_bias = read_param(dir + "/fc1.bias.txt");
    auto fc2_weight = read_param(dir + "/fc2.weight.txt");
    auto fc2_bias = read_param(dir + "/fc2.bias.txt");
    auto fc3_weight = read_param(dir + "/fc3.weight.txt");
    auto fc3_bias = read_param(dir + "/fc3.bias.txt");

    // 打印每一个标签，仅用于调试！
	/*
    for (const auto& label : labels) {
        std::cout << label << " ";
    }
	std::cout< <std::endl;
	*/


   // 进行推理
	// std::cout << images.size() << std::endl;
	// std::cout << images[0].size() << std::endl;
	
	// 参数加载
	// std::cout << fc3_bias.size() << std::endl;
	float correct_num = 0;
    int total_num = 0;
    //std::cout << total_num;
    
    //std::cout << std::endl;
	
    float * d_image,* d_nm,* d_conv1,*d_pool1,*d_conv2,* d_pool2,* d_fc1,* d_fc2,* d_fc3;
    cudaMalloc((void**)&d_image,images.size()* 28*28*sizeof(float));
    cudaMalloc((void**)&d_nm, BATCH_SIZE*28*28*sizeof(float));
    cudaMalloc((void**)&d_conv1, BATCH_SIZE*6*28*28*sizeof(float));
    cudaMalloc((void**)&d_pool1, BATCH_SIZE*6*24*24*sizeof(float));
    cudaMalloc((void**)&d_conv2, BATCH_SIZE*6*16*12*12*sizeof(float));
    cudaMalloc((void**)&d_pool2, BATCH_SIZE*16*8*8*sizeof(float));
    cudaMalloc((void**)&d_fc1, BATCH_SIZE*256*sizeof(float));
    cudaMalloc((void**)&d_fc2, BATCH_SIZE*120*sizeof(float));
    cudaMalloc((void**)&d_fc3, BATCH_SIZE*84*sizeof(float));

    cudaMalloc((void**)&D,BATCH_SIZE*16*6*28*28*sizeof(float));

	float * l_image,* l_nm,* l_conv1,*l_pool1,*l_conv2,* l_pool2,* l_fc1,* l_fc2,* l_fc3,* l_output;
    cudaMalloc((void**)&l_image,images.size()* 28*28*sizeof(float));
    cudaMalloc((void**)&l_nm, BATCH_SIZE*28*28*sizeof(float));
    cudaMalloc((void**)&l_conv1, BATCH_SIZE*6*28*28*sizeof(float));
    cudaMalloc((void**)&l_pool1, BATCH_SIZE*6*24*24*sizeof(float));
    cudaMalloc((void**)&l_conv2, BATCH_SIZE*6*16*12*12*sizeof(float));
    cudaMalloc((void**)&l_pool2, BATCH_SIZE*16*8*8*sizeof(float));
    cudaMalloc((void**)&l_fc1, BATCH_SIZE*256*sizeof(float));
    cudaMalloc((void**)&l_fc2, BATCH_SIZE*120*sizeof(float));
    cudaMalloc((void**)&l_fc3, BATCH_SIZE*84*sizeof(float));
	cudaMalloc((void**)&l_output, BATCH_SIZE*10*sizeof(float));

    float * d_conv1_weight,* d_conv2_weight,* d_conv1_bias,*d_conv2_bias,*d_fc1_weight,* d_fc2_weight,* d_fc3_weight,*d_fc1_bias,* d_fc2_bias,* d_fc3_bias;
    cudaMalloc((void**)&d_conv1_weight, 6*MASK_WIDTH*MASK_WIDTH*sizeof(float));
    cudaMalloc((void**)&d_conv2_weight, 16*6*MASK_WIDTH*MASK_WIDTH*sizeof(float));
    cudaMalloc((void**)&d_conv1_bias, 6*sizeof(float));
    cudaMalloc((void**)&d_conv2_bias,16*sizeof(float));
    cudaMalloc((void**)&d_fc1_weight, 16*4*4*120*sizeof(float));
    cudaMalloc((void**)&d_fc2_weight, 120*84*sizeof(float));
    cudaMalloc((void**)&d_fc3_weight, 84*10*sizeof(float));
    cudaMalloc((void**)&d_fc1_bias, 120*sizeof(float));
    cudaMalloc((void**)&d_fc2_bias, 84*sizeof(float));
    cudaMalloc((void**)&d_fc3_bias, 10*sizeof(float));


	float * l_conv1_weight,* l_conv2_weight,* l_conv1_bias,*l_conv2_bias,*l_fc1_weight,* l_fc2_weight,* l_fc3_weight,*l_fc1_bias,* l_fc2_bias,* l_fc3_bias;
    cudaMalloc((void**)&l_conv1_weight, 6*MASK_WIDTH*MASK_WIDTH*sizeof(float));
    cudaMalloc((void**)&l_conv2_weight, 16*6*MASK_WIDTH*MASK_WIDTH*sizeof(float));
    cudaMalloc((void**)&l_conv1_bias, 6*sizeof(float));
    cudaMalloc((void**)&l_conv2_bias,16*sizeof(float));
    cudaMalloc((void**)&l_fc1_weight, 16*4*4*120*sizeof(float));
    cudaMalloc((void**)&l_fc2_weight, 120*84*sizeof(float));
    cudaMalloc((void**)&l_fc3_weight, 84*10*sizeof(float));
    cudaMalloc((void**)&l_fc1_bias, 120*sizeof(float));
    cudaMalloc((void**)&l_fc2_bias, 84*sizeof(float));
    cudaMalloc((void**)&l_fc3_bias, 10*sizeof(float));
    cudaMemcpy(l_conv1_weight, &conv1_weight[0], 1*6*MASK_WIDTH*MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_conv1_bias, &conv1_bias[0], 1*6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_conv2_weight, &conv2_weight[0], 16*6*MASK_WIDTH*MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_conv2_bias, &conv2_bias[0], 1*16*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc1_weight, &fc1_weight[0], 16*4*4*120 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc1_bias, &fc1_bias[0], 120 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc2_weight, &fc2_weight[0], 84*120 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc2_bias, &fc2_bias[0], 84 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc3_weight, &fc3_weight[0], 84*10 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_fc3_bias, &fc3_bias[0], 10 *sizeof(float), cudaMemcpyHostToDevice);

    float * d_final;
    int *d_table,*d_label;
    cudaMalloc((void**)&d_final, images.size()*10*sizeof(float));
    cudaMalloc((void**)&d_table, images.size()*sizeof(float));
    cudaMalloc((void**)&d_label, images.size()*sizeof(float));
    cudaMemcpy(d_label, &labels[0], images.size() *sizeof(int), cudaMemcpyHostToDevice);

    //load all pics
	for (int i = 0; i < images.size(); ++i) {
		cudaMemcpy(d_image + 28 * 28*i, &images[i][0], 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
	}

    // init paras
    para_init(d_conv1_weight,5*5*6,0.2);
    para_init(d_conv1_bias,6,0.2);
    para_init(d_conv2_weight,5*5*16*6,0.2);
    para_init(d_conv2_bias,16,0.2);
    para_init(d_fc1_weight,256*120,0.2);
    para_init(d_fc1_bias,120,0.2);
    para_init(d_fc2_weight,120*84,0.2);
    para_init(d_fc2_bias,84,0.2);
    para_init(d_fc3_weight,84*10,0.2);
    para_init(d_fc3_bias,10,0.2);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    for(int epoch = 0;epoch<EPOCH;++epoch)
    for (int t = 0; t <images.size(); t+=BATCH_SIZE) {
        // TODO ...在这里实现利用CUDA对图片进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
		// 打印每一张图片，仅用于调试！
        float * image = d_image + t *28*28;
		
		//normalize the pics
        Normalize_new(image,d_nm,28,28);

        //conv1
        Conv2D_group_new(d_conv1_weight, d_nm, d_conv1,28,28,1,6,d_conv1_bias,true,true);
		
        //pool1
        MaxPool_group_new(d_conv1,d_pool1,24,24,6);
		

		//conv2        
        Conv2D_group_new(d_conv2_weight,d_pool1,d_conv2,12,12,6,16,d_conv2_bias,true,true);

        //pool2
        MaxPool_group_new(d_conv2,d_pool2,8,8,16);
    
    	//fc1
        FC_new(d_pool2,d_fc1,d_fc1_weight,d_fc1_bias,256,120,true);
        
		//fc2
        FC_new(d_fc1,d_fc2,d_fc2_weight,d_fc2_bias,120,84,true);


        //fc3
        FC_new(d_fc2,d_fc3,d_fc3_weight,d_fc3_bias,84,10,false);
        cudaMemcpy(d_final+t*10,d_fc3,10*sizeof(float)*BATCH_SIZE,cudaMemcpyDeviceToDevice);
        //total_num +=BATCH_SIZE;

        //backforward
		FC_final_bp(d_fc3,labels[t],l_output,10);
        FC_bp(d_fc2,d_fc3,d_fc3_weight,l_output,l_fc3,l_fc3_weight,l_fc3_bias,84,10,false);
        FC_bp(d_fc1,d_fc2,d_fc2_weight,l_fc3,l_fc2,l_fc2_weight,l_fc2_bias,120,84,true);
        FC_bp(d_pool2,d_fc1,d_fc1_weight,l_fc2,l_fc1,l_fc1_weight,l_fc1_bias,256,120,true);
        MaxPool_bp(d_conv2,l_fc1,l_pool2,8,8,16);
        Conv2D_bp(d_pool1,d_conv2,d_conv2_weight,l_pool2,l_conv2,l_conv2_weight,l_conv2_bias,8,8,6,16,true);
        MaxPool_bp(d_conv1,l_conv2,l_pool1,12,12,6);
        Conv2D_bp(d_nm,d_conv1,d_conv1_weight,l_pool1,l_conv1,l_conv1_weight,l_conv1_bias,24,24,1,6,true);
        

        //update the paras(weight and bias)
        BP_UPDATE(d_conv1_weight,l_conv1_weight,5*5*6,-0.02);
        BP_UPDATE(d_conv1_bias,l_conv1_bias,6,-0.02);
        BP_UPDATE(d_conv2_weight,l_conv2_weight,5*5*16*6,-0.02);
        BP_UPDATE(d_conv2_bias,l_conv2_bias,16,-0.02);
        BP_UPDATE(d_fc1_weight,l_fc1_weight,256*120,-0.01);
        BP_UPDATE(d_fc1_bias,l_fc1_bias,120,-0.01);
        BP_UPDATE(d_fc2_weight,l_fc2_weight,120*84,-0.01);
        BP_UPDATE(d_fc2_bias,l_fc2_bias,84,-0.01);
        BP_UPDATE(d_fc3_weight,l_fc3_weight,84*10,-0.005);
        BP_UPDATE(d_fc3_bias,l_fc3_bias,10,-0.005);

    }

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    //foward test
    images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // 读取测试集标签
    labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    cudaMemcpy(d_label, &labels[0], images.size() *sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < images.size(); ++i) {
		cudaMemcpy(d_image + 28 * 28*i, &images[i][0], 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
	}

    for (int t = 0; t <images.size(); t+=BATCH_SIZE) {
        // TODO ...在这里实现利用CUDA对图片进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
		// 打印每一张图片，仅用于调试！
        float * image = d_image + t *28*28;
		
		//nm
        Normalize_new(image,d_nm,28,28);

        //conv1
        Conv2D_group_new(d_conv1_weight, d_nm, d_conv1,28,28,1,6,d_conv1_bias,true,true);
		
        //pool1
        MaxPool_group_new(d_conv1,d_pool1,24,24,6);
		

		//conv2        
        Conv2D_group_new(d_conv2_weight,d_pool1,d_conv2,12,12,6,16,d_conv2_bias,true,true);

        //pool2
        MaxPool_group_new(d_conv2,d_pool2,8,8,16);
    
    	//fc1
        FC_new(d_pool2,d_fc1,d_fc1_weight,d_fc1_bias,256,120,true);
        
		//fc2
        FC_new(d_fc1,d_fc2,d_fc2_weight,d_fc2_bias,120,84,true);


        //fc3
        FC_new(d_fc2,d_fc3,d_fc3_weight,d_fc3_bias,84,10,false);
        cudaMemcpy(d_final+t*10,d_fc3,10*sizeof(float)*BATCH_SIZE,cudaMemcpyDeviceToDevice);
        total_num +=BATCH_SIZE;
    }
    //calculate final output
    FC_final_new(d_final,d_label,d_table,10,images.size());
    int* table = (int*)(malloc(images.size() * sizeof(int)));
    cudaMemcpy(table,d_table,images.size()*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i<images.size();++i){
        if(int(table[i]) == 1){
            correct_num ++;
        }
    }
    free(table);
    cudaFree(D);
    
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count();

    //store the paras
    float* tmp = (float*)malloc(Max_Para_num*sizeof(float));
    cudaMemcpy(tmp,d_conv1_weight,5*5*6*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/conv1.weight.txt",tmp,5*5*6);

    cudaMemcpy(tmp,d_conv1_bias,6*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/conv1.bias.txt",tmp,6);

    cudaMemcpy(tmp,d_conv2_weight,5*5*16*6*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/conv2.weight.txt",tmp,5*5*16*6);

    cudaMemcpy(tmp,d_conv2_bias,16*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/conv2.bias.txt",tmp,16);

    cudaMemcpy(tmp,d_fc1_weight,256*120*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc1.weight.txt",tmp,256*120);

    cudaMemcpy(tmp,d_fc1_bias,120*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc1.bias.txt",tmp,120);

    cudaMemcpy(tmp,d_fc2_weight,120*84*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc2.weight.txt",tmp,120*84);

    cudaMemcpy(tmp,d_fc2_bias,120*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc2.bias.txt",tmp,84);

    cudaMemcpy(tmp,d_fc3_weight,84*10*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc3.weight.txt",tmp,84*10);

    cudaMemcpy(tmp,d_fc3_bias,10*sizeof(float),cudaMemcpyDeviceToHost);
    writeFloatsToFile(dir +"/fc3.bias.txt",tmp,10);

    

    //while(1);
}