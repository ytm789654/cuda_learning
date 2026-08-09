#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define __M 4096
#define __N 4096
#define __K 4096


// generate rando__M nu__M filling the __Matrix
void generateRandomMatrix(float* __Matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        __Matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // range[-1, 1]
    }
}

__global__ void matmul_baseline(float* A, float* B, float* C,
                                int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int e = 0; e < K; e++)
        sum += A[row * K + e] * B[e * N + col];
    C[row * N + col] = sum;
}

/*
use y control row
use x control col
*/
template <int BLOCK_SIZE>
__global__ void matmul_kernel_v1(float *A, float *B, float *C,
                                 const int M, const int N, const int K)
{
    __shared__ float sdataA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sdataB[BLOCK_SIZE][BLOCK_SIZE];
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    int col = blockIdx.x * blockDim.x + x;
    int row = blockIdx.y * blockDim.y + y;
    float sum = 0.0f;
    for (int k = 0; k < K / BLOCK_SIZE; k++){
        sdataA[y][x] = A[row * K + k * BLOCK_SIZE + x];
        sdataB[y][x] = B[(k * BLOCK_SIZE + y) * N + col];
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; e++)
            sum += sdataA[y][e] * sdataB[e][x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

/*
rearrange BM BK BN
grid is deployed to split C into many BM * BN chunk, each chunk is handled by a block;
block is one way and re-arranged by code instead of 2-dim block to be more flexible
in this kernel, x is used horizonal and y is used vertical.
*/
template <int BM, int BN, int BK, int BLOCK_SIZE, int C_BLOCK_SIZE>
__global__ void matmul_kernel_v2(float* A, float* B, float* C,
                                   int M, int K, int N)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // for tile A, BLOCK_SIZE threads is deployed into 2 dim thread_a_y_step * BK
    constexpr int BLOCK_SIZE_A_X = BK;
    constexpr int BLOCK_SIZE_A_Y = BLOCK_SIZE / BLOCK_SIZE_A_X;
    int thread_a_y = tid / BLOCK_SIZE_A_X;
    int thread_a_y_step = BLOCK_SIZE_A_Y;
    int thread_a_x = tid % BLOCK_SIZE_A_X;

    // for tile B, BLOCK_SIZE threads is deployed into 2 dim BK * thread_b_x_step
    constexpr int BLOCK_SIZE_B_Y = BK;
    constexpr int BLOCK_SIZE_B_X = BLOCK_SIZE / BLOCK_SIZE_B_Y;
    int thread_b_y = tid / BLOCK_SIZE_B_X;
    int thread_b_x = tid % BLOCK_SIZE_B_X;
    int thread_b_x_step = BLOCK_SIZE_B_X;

    // tile C is BM * BN, BLOCK_SIZE threads is deployed into 2 dim C_BLOCK_SIZE * C_BLOCK_SIZE
    // so each thread calc TM * TN vals in tile C
    // TM = BM / C_BLOCK_SIZE, TN = BN / C_BLOCK_SIZE
    int thread_c_y = tid / C_BLOCK_SIZE;
    int thread_c_x = tid % C_BLOCK_SIZE;
    constexpr int TM = BM / C_BLOCK_SIZE;
    constexpr int TN = BN / C_BLOCK_SIZE;

    // acc will not write to C consectively
    // as TM is from C BLOCK num in y axis and TN from C BLOCK num x axis
    // each C BLOCK is C_BLOCK_SIZE * C_BLOCK_SIZE = BLOCK_SIZE, all threads write one C BLOCK!
    // so acc[i][j] is thread write a value in C BLOCK[i][j], within the BLOCK the thread write to (thread_c_y, thread_c_x)
    float acc[TM][TN]= {0.0f};

    /* 
        the k loop, the block will 
        1. read BM * BK from A into As
        2. read BK * BN from B into Bs
        3. accumulate BM * BN
            3.1 BM * BN is split into BLOCK_SIZE * TM * TN (T means thread here, so each thread calc TM * TN for C)
            acc[i][j] is accumulate the (i, j) C_BLOCK 
    */
    for (int k = 0; k < K / BK; k++){
        // read As from GMEM
        for (int i = thread_a_y; i < BM; i += thread_a_y_step){
            int r = r0 + i;
            int c = k * BK + thread_a_x;
            As[i][thread_a_x] = A[r * K + c];
        }

        // read Bs from GMEM
        for (int j = thread_b_x; j < BN; j += thread_b_x_step){
            int r = k * BK + thread_b_y;
            int c = c0 + j;
            Bs[thread_b_y][j] = B[r * N + c];
        }
        __syncthreads();

        for (int i = 0; i < TM; i++){
            int Srow = i * C_BLOCK_SIZE + thread_c_y;
            for (int j = 0; j < TN; j++){
                int Scol = j * C_BLOCK_SIZE + thread_c_x;
                for (int e = 0; e < BK; e++)
                    acc[i][j] += As[Srow][e] * Bs[e][Scol];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++){
        int r = r0 + i * C_BLOCK_SIZE + thread_c_y;
        for (int j = 0; j < TN; j++){
            int c = c0 + j * C_BLOCK_SIZE + thread_c_x;
            C[r * N + c] = acc[i][j];
        }
    }
}

/*
based on V2, V3 use out product to calculate acc.
forget about C_BLOCK first.
Read a col vector with length TM(yes, col) in As and row vector with length TN in Bs to register in thead,
and accumulate whole acc matrix(TM x TN) in each out loop, the e loop.
This can make full use of each shared mem read(No need to read same element in within matrix).
arith tense is related to (TM * TN) / (TM + TN), then when TM = TN it has max value.
*/
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
__global__ void matmul_kernel_v3(float* A, float* B, float* C,
                                   int M, int K, int N){
                                    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // for tile A, BLOCK_SIZE threads is deployed into 2 dim thread_a_y_step * BK
    constexpr int BLOCK_SIZE_A_X = BK;
    constexpr int BLOCK_SIZE_A_Y = BLOCK_SIZE / BLOCK_SIZE_A_X;
    int thread_a_y = tid / BLOCK_SIZE_A_X;
    int thread_a_y_step = BLOCK_SIZE_A_Y;
    int thread_a_x = tid % BLOCK_SIZE_A_X;

    // for tile B, BLOCK_SIZE threads is deployed into 2 dim BK * thread_b_x_step
    constexpr int BLOCK_SIZE_B_Y = BK;
    constexpr int BLOCK_SIZE_B_X = BLOCK_SIZE / BLOCK_SIZE_B_Y;
    int thread_b_y = tid / BLOCK_SIZE_B_X;
    int thread_b_x = tid % BLOCK_SIZE_B_X;
    int thread_b_x_step = BLOCK_SIZE_B_X;

    // constexpr int threads_num = (BM * BN) / (TM * TN);  //total threads
    constexpr int threads_num_row = BN / TN;    // threads per row
    int thread_row_idx = tid / threads_num_row;
    int thread_col_idx = tid % threads_num_row;

    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};

    float acc[TM][TN] = {0.0f};

    for (int k = 0; k < K / BK; k++){
        for (int i = thread_a_y; i < BM; i += thread_a_y_step){
            int r = r0 + i;
            int c = k * BK + thread_a_x;
            As[i][thread_a_x] = A[r * K + c];
        }

        for (int j = thread_b_x; j < BN; j += thread_b_x_step){
            int r = k * BK + thread_b_y;
            int c = c0 + j;
            Bs[thread_b_y][j] = B[r * N + c];
        }
        __syncthreads();

        for (int e = 0; e < BK; e++){
            for (int i = 0; i < TM; i++){
                int thread_r = thread_row_idx * TM + i;
                a_frag[i] = As[thread_r][e];
            }
            for (int j = 0; j < TN; j++){
                int thread_c = thread_col_idx * TN + j;
                b_frag[j] = Bs[e][thread_c];
            }
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++){
        int row = r0 + thread_row_idx * TM + i;
        for (int j = 0; j < TN; j++){
            int col = c0 + thread_col_idx * TN + j;
            C[row * N + col] = acc[i][j];
        }
    }
}

/*
    based on V3, V4 will re-arrange warp distribution.
    consider in the warp angle:
        If threads have the same thread_x, they can share the read in Bs via bank broadcast, 
        because threads in warp execute same instruction, they have same thread_x, they read same col in B.
        Likely, threads with same y can share read in As.
        If the shape of warp is arranged as m x n = 32, the calculation times is fixed, Constant x TM x TN, 
        so consider the shared mem read times:
            each row, read TM from As and broadcast to row, total m x TM
            each col, read TN from Bs and broadcast to col, total n x TN
        then arith tense is (Constant x TM x TN) / (m x TM + n x TN) = Constant / (m + n), m x m = 32 so m=4 n = 8.
    //todo try bit arith instead of * /
*/
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
__global__ void matmul_kernel_v4(float* A, float* B, float* C,
                                   int M, int K, int N){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    // for tile A, BLOCK_SIZE threads is deployed into 2 dim thread_a_y_step * BK
    constexpr int BLOCK_SIZE_A_X = BK;
    constexpr int BLOCK_SIZE_A_Y = BLOCK_SIZE / BLOCK_SIZE_A_X;
    int thread_a_y = tid / BLOCK_SIZE_A_X;
    int thread_a_y_step = BLOCK_SIZE_A_Y;
    int thread_a_x = tid % BLOCK_SIZE_A_X;

    // for tile B, BLOCK_SIZE threads is deployed into 2 dim BK * thread_b_x_step
    constexpr int BLOCK_SIZE_B_Y = BK;
    constexpr int BLOCK_SIZE_B_X = BLOCK_SIZE / BLOCK_SIZE_B_Y;
    int thread_b_y = tid / BLOCK_SIZE_B_X;
    int thread_b_x = tid % BLOCK_SIZE_B_X;
    int thread_b_x_step = BLOCK_SIZE_B_X;

    // thread per row/col within warp
    constexpr int tpr = 8;
    constexpr int tpc = 4;

    constexpr int warp_per_row = BN / (TN * tpr);
    // constexpr int warp_nums = BLOCK_SIZE / warpSize;

    int warp_id = tid / warpSize;
    int warp_lane = tid % warpSize;
    int warp_row_id = warp_id / warp_per_row;
    int warp_col_id = warp_id % warp_per_row;

    // in warp, calc thread x, y
    int thread_warp_y = warp_lane / tpr;
    int thread_warp_x = warp_lane % tpr;

    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};

    float acc[TM][TN] = {0.0f};

    for (int k = 0; k < K / BK; k++){
        for (int i = thread_a_y; i < BM; i += thread_a_y_step){
            int r = r0 + i;
            int c = k * BK + thread_a_x;
            As[i][thread_a_x] = A[r * K + c];
        }

        for (int j = thread_b_x; j < BN; j += thread_b_x_step){
            int r = k * BK + thread_b_y;
            int c = c0 + j;
            Bs[thread_b_y][j] = B[r * N + c];
        }
        __syncthreads();

        for (int e = 0; e < BK; e++){
            for (int i = 0; i < TM; i++){
                int r = (warp_row_id * tpc + thread_warp_y) * TM + i;
                a_frag[i] = As[r][e];
            }
            for (int j = 0; j < TN; j++){
                int c = (warp_col_id * tpr + thread_warp_x) * TN + j;
                b_frag[j] = Bs[e][c];
            }
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++){
        int r = r0 + (warp_row_id * tpc + thread_warp_y) * TM + i;
        for (int j = 0; j < TN; j++){
            int c = c0 + (warp_col_id * tpr + thread_warp_x) * TN + j;
            C[r * N + c] = acc[i][j];
        }
    }
}

/*
    V5: float4 optimization replace single read to reduce instruction.
    As a_frag read from As in col, transpose As into As[BK][BM] to make
    As can be read via float4
*/
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
__global__ void matmul_kernel_v5(float* A, float* B, float* C,
                                   int M, int K, int N){
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Bs[BK][BN];

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;

    int tid = threadIdx.x;

    // for tile A, BLOCK_SIZE threads is deployed into 2 dim thread_a_y_step * (BK / 4)
    constexpr int BLOCK_SIZE_A_X = BK / 4;  // for float4
    constexpr int BLOCK_SIZE_A_Y = BLOCK_SIZE / BLOCK_SIZE_A_X;
    int thread_a_y = tid / BLOCK_SIZE_A_X;
    int thread_a_y_step = BLOCK_SIZE_A_Y;
    int thread_a_x = tid % BLOCK_SIZE_A_X;

    // for tile B, BLOCK_SIZE threads is deployed into 2 dim BK * thread_b_x_step
    constexpr int BLOCK_SIZE_B_Y = BK;
    constexpr int BLOCK_SIZE_B_X = BLOCK_SIZE / BLOCK_SIZE_B_Y;
    int thread_b_y = tid / BLOCK_SIZE_B_X;
    int thread_b_x = tid % BLOCK_SIZE_B_X;
    int thread_b_x_step = BLOCK_SIZE_B_X;

    // constexpr int threads_num = (BM * BN) / (TM * TN);  //total threads
    constexpr int threads_num_row = BN / TN;    // threads per row for C
    int thread_row_idx = tid / threads_num_row;
    int thread_col_idx = tid % threads_num_row;

    float4 a_frag[TM / 4];
    float4 b_frag[TN / 4];

    float acc[TM][TN] = {0.0f};

    for (int k = 0; k < K / BK; k++){
        for (int i = thread_a_y; i < BM; i += thread_a_y_step){
            int r = r0 + i;
            int c = k * BK + thread_a_x * 4;
            // As[i][thread_a_x] = A[r * K + c]
            float4 tmp = FLOAT4(A[r * K + c]);
            As[thread_a_x * 4 + 0][i] = tmp.x;
            As[thread_a_x * 4 + 1][i] = tmp.y;
            As[thread_a_x * 4 + 2][i] = tmp.z;
            As[thread_a_x * 4 + 3][i] = tmp.w;
        }

        for (int j = thread_b_x; j * 4 < BN; j += thread_b_x_step){
            int r = k * BK + thread_b_y;
            int c = c0 + j * 4;
            FLOAT4(Bs[thread_b_y][j * 4]) = FLOAT4(B[r * N + c]);
        }

        __syncthreads();

        for (int e = 0; e < BK; e++){
            for (int i = 0; i < TM / 4; i++){
                a_frag[i] = FLOAT4(As[e][thread_row_idx * TM]);
            }

            for (int j = 0; j < TN / 4; j++){
                b_frag[j] = FLOAT4(Bs[e][thread_col_idx * TN]);
            }

            for (int i = 0; i < TM / 4; i++)
                for (int j = 0; j < TN / 4; j++){
                    acc[i*4+0][j*4+0] += a_frag[i].x*b_frag[j].x;  acc[i*4+0][j*4+1] += a_frag[i].x*b_frag[j].y;
                    acc[i*4+0][j*4+2] += a_frag[i].x*b_frag[j].z;  acc[i*4+0][j*4+3] += a_frag[i].x*b_frag[j].w;
                    acc[i*4+1][j*4+0] += a_frag[i].y*b_frag[j].x;  acc[i*4+1][j*4+1] += a_frag[i].y*b_frag[j].y;
                    acc[i*4+1][j*4+2] += a_frag[i].y*b_frag[j].z;  acc[i*4+1][j*4+3] += a_frag[i].y*b_frag[j].w;
                    acc[i*4+2][j*4+0] += a_frag[i].z*b_frag[j].x;  acc[i*4+2][j*4+1] += a_frag[i].z*b_frag[j].y;
                    acc[i*4+2][j*4+2] += a_frag[i].z*b_frag[j].z;  acc[i*4+2][j*4+3] += a_frag[i].z*b_frag[j].w;
                    acc[i*4+3][j*4+0] += a_frag[i].w*b_frag[j].x;  acc[i*4+3][j*4+1] += a_frag[i].w*b_frag[j].y;
                    acc[i*4+3][j*4+2] += a_frag[i].w*b_frag[j].z;  acc[i*4+3][j*4+3] += a_frag[i].w*b_frag[j].w;
                }
        }

        __syncthreads();
    }
    
    for (int i = 0; i < TM; i++){
        int r = r0 + thread_row_idx * TM + i;
        for (int j = 0; j * 4 < TN; j += 4){
            int c = c0 + thread_col_idx * TN + j;
            FLOAT4(C[r * N + c]) = make_float4(acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
        }
    }
}
int main()
{
    // seed the random generator so data differs between runs
    srand((unsigned int)time(NULL));

    int __Matrix_size_A = __M * __K;
    int __Matrix_size_B = __K * __N;
    int __Matrix_size_C = __M * __N;
    float *h_A, *h_B, *h_C, *h_C_Cublas;
    h_A = new float[__Matrix_size_A];
    h_B = new float[__Matrix_size_B];
    h_C = new float[__Matrix_size_C];
    h_C_Cublas = new float[__Matrix_size_C];
    generateRandomMatrix(h_A, __M, __K);
    generateRandomMatrix(h_B, __K, __N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, __Matrix_size_A * sizeof(float));
    cudaMalloc(&d_B, __Matrix_size_B * sizeof(float));
    cudaMalloc(&d_C, __Matrix_size_C * sizeof(float));
    
    cudaMemcpy(d_A, h_A, __Matrix_size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, __Matrix_size_B * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // the matrix stored in array is row major order, but CUBLAS uses col major order
    // this means when pass row major A and B to function, the function is handling A.trans and B.trans
    // then we can calc C^T = B^T x A^T without changing the order of A,B and get C^T as result
    // C^T is still col major, this means it has same mem sequence as C.
    // Conclusion: only need to replace A and B in param and adjust NMK pos, we can get C without any trans.
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, // no trans for A,B, instead calc B^T x A^T
                __N, __M, __K,                  // B^T is N x K, A^T is K x M, result is N x M
                &alpha,
                d_B, __N,                   // B^T has N rows
                d_A, __K,                   // A^T has K rows
                &beta,
                d_C, __N);           // C^T has N rows
    cudaMemcpy(h_C_Cublas, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);
    const int check_row = __M / 2 + 1;
    const int check_col = __N / 2 + 1;
    printf("check with CUBLAS, in row %d, col %d value is %f\n", check_row, check_col, h_C_Cublas[check_row * __N + check_col]);

    // launch baseline
    // {
    //     const int base_line_block_size = 32;
    //     const dim3 base_line_block(base_line_block_size, base_line_block_size);
    //     const dim3 base_line_grid(__M / base_line_block_size, __N / base_line_block_size);
    //     matmul_baseline<<<base_line_grid, base_line_block>>>(d_A, d_B, d_C, __M, __N, __K);
    //     cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

    //     if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
    //         printf("baseline check pass \n");
    //     else
    //         printf("v0 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
    //     cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    // }

    // launch V1
    // {
    //     constexpr int block_dim = 32;
    //     const int BLOCK_NUM_X = __M / block_dim;
    //     const int BLOCK_NUM_Y = __N / block_dim;

    //     dim3 grid (BLOCK_NUM_X, BLOCK_NUM_Y);
    //     dim3 block (block_dim, block_dim);
    //     matmul_kernel_v1<block_dim><<<grid, block>>>(d_A, d_B, d_C, __M, __N, __K);
    //     cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

    //     if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
    //         printf("v1 check pass \n");
    //     else
    //         printf("v1 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
    //     cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    // }

    // launch V2
    // tile: BM x BN, each block loads BM x BK of A and BK x BN of B per k-step
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int BS = 256;      // 1D block threads (16 x 16)
    constexpr int CBS = 16;      // C sub-tile size: TM = BM/CBS, TN = BN/CBS
    dim3 grid2(__N / BN, __M / BM);
    dim3 block2(BS);

    {
        matmul_kernel_v2<BM, BN, BK, BS, CBS><<<grid2, block2>>>(d_A, d_B, d_C, __M, __K, __N);
        cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
            printf("v2 check pass \n");
        else
            printf("v2 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
        cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    }

    constexpr int TM = 4;
    constexpr int TN = 4;
    
    // launch V3
    {
        matmul_kernel_v3<BM, BN, BK, TM, TN, BS><<<grid2, block2>>>(d_A, d_B, d_C, __M, __K, __N);
        cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
            printf("v3 check pass \n");
        else
            printf("v3 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
        cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    }
    
    // launch V4
    {
        matmul_kernel_v4<BM, BN, BK, TM, TN, BS><<<grid2, block2>>>(d_A, d_B, d_C, __M, __K, __N);
        cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
            printf("v4 check pass \n");
        else
            printf("v4 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
        cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    }

    // launch V5
    {
        matmul_kernel_v5<BM, BN, BK, TM, TN, BS><<<grid2, block2>>>(d_A, d_B, d_C, __M, __K, __N);
        cudaMemcpy(h_C, d_C, __Matrix_size_C * sizeof(float), cudaMemcpyDeviceToHost);

        if (h_C[check_row * __N + check_col] - h_C_Cublas[check_row * __N + check_col] < 1e-4)
            printf("v5 check pass \n");
        else
            printf("v5 check error! expect %f but get %f \n", h_C_Cublas[check_row * __N + check_col], h_C[check_row * __N + check_col]);
        cudaMemset(d_C, 0, __Matrix_size_C * sizeof(float));
    }
    // ncu --set detailed -f -o matmul.ncu-rep 1.exe
    // nvcc -o 1.exe matmul.cu -lcublas
}