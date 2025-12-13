#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define debug_with_msg(msg)                  \
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) \
        printf(msg)

#define BLOCK_NUM 256
#define BLOCK_SIZE 1024

const int d_model = 1024;

// get maximun vals in M x N Matrix, return int 1D vector out
// block will contain 1024 threads to handle each row, a 2-round warp shufle will be executed to find the max val
__global__ void max_kernel(float *A, float *out, size_t M, size_t N)
{
    int row_idx = blockIdx.x;
    float *row_base;
    __shared__ float sdata[32];
    while(row_idx < M)
    {
        row_base = A + row_idx * N;
        float val = -FLT_MAX;
        int col = threadIdx.x;
        while(col < N){
            val = fmaxf(val, row_base[col]);
            col += blockDim.x;
        }
        int thread_idx = threadIdx.x;
        int warpId = thread_idx / warpSize;
        int warpLane = thread_idx % warpSize;
        unsigned shfl_mask = 0XFFFFFFFF;

        for (int offset = warpSize / 2; offset > 0; offset = offset >> 1){
            float tmp = __shfl_down_sync(shfl_mask, val, offset);
            val = max(val, tmp);
        }

        if(warpLane == 0)
            sdata[warpId] = val;
        __syncthreads();

        if (warpId == 0)
        {
            val = (warpLane < BLOCK_SIZE / warpSize)?sdata[warpLane]:-FLT_MAX;
            for (int offset = warpSize / 2; offset > 0; offset = offset/2){
                float tmp = fmaxf(val, __shfl_down_sync(shfl_mask, val, offset));
                val = fmaxf(val, tmp);
            }
            if (warpLane == 0)
                out[row_idx] = val;
        }
        row_idx += gridDim.x;
    }
}

// A: M x K, B: K x N, C: M x N
// grid size is 16 x 16
// block size is 32 x 32, each block handle 32 x 32 size block and write result to C every time, and then move to another block in C
__global__ void matmul(float *A, float *B, float *C, size_t M, size_t K, size_t N)
{
    __shared__ float sdataA[32][32];
    __shared__ float sdataB[32][32];
    // id x control row in C, and block id y control col
    int rowBlockId = blockIdx.x;
    int colBlockId = blockIdx.y;
    while(rowBlockId * blockDim.x < M)
    {
        while(colBlockId * blockDim.y < N)
        {
            float temp = 0.0f;
            for (int k = 0; k * blockDim.y < K; k++)
            {
                int x = threadIdx.x;
                int y = threadIdx.y;
                int rowA = rowBlockId * blockDim.x + x;
                int colA = k * blockDim.y + y;
                sdataA[x][y] = (rowA < M && colA < K)?A[rowA * K + colA]:0.0f;
                int rowB = k * blockDim.y + x;
                int colB = colBlockId * blockDim.y + y;
                sdataB[x][y] = (rowB < K && colB < N)?B[rowB * N + colB]:0.0f;
                __syncthreads();
                for (int e = 0; e < blockDim.y; e++)
                    temp += sdataA[x][e] * sdataB[e][y];
                __syncthreads();
            }
            int rowC = rowBlockId * blockDim.x + threadIdx.x;
            int colC = colBlockId * blockDim.y + threadIdx.y;
            if(rowC < M && colC < N)
                C[rowC * N + colC] = temp;
            colBlockId += gridDim.y;
        }
        colBlockId = blockIdx.y;
        rowBlockId += gridDim.x;
    }
}

// generate random num filling the matrix
void generateRandomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // range[-1, 1]
    }
}

// compare two matrixs
bool compareMatrices(const float* C1, const float* C2, int rows, int cols, float epsilon = 1e-4f) {
    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabs(C1[i] - C2[i]);
        if (diff > epsilon) {
            printf("Mismatch at index %d: C1[%d] = %f, C2[%d] = %f, diff = %f\n", 
                   i, i, C1[i], i, C2[i], diff);
            return false;
        }
    }
    return true;
}

int main()
{
    // code block for test max_kernel
    #ifdef test_max_kernel
    {
        float *h_A, *d_A;
        float *h_out, *d_out;

        const int testMatrixHeight = 1024;
        const int testMatrixWidth = 4094;

        h_A = new float[testMatrixHeight * testMatrixWidth];
        for (int i = 0; i < testMatrixHeight; i++)
        {
            for (int j = 0; j < testMatrixWidth; j++)
                h_A[i * testMatrixWidth + j] = 1.0f;
            h_A[i * testMatrixWidth + i] = 2 * float(i) + 1.0f;
        }
        h_out = new float[testMatrixHeight];

        // allocate mem in GPU
        cudaMalloc(&d_A, testMatrixHeight * testMatrixWidth * sizeof(float));
        cudaMalloc(&d_out, testMatrixHeight * sizeof(float));
        cudaCheckErrors("cudaMalloc failure");

        // copy data to GPU
        cudaMemcpy(d_A, h_A, testMatrixHeight * testMatrixWidth * sizeof(float), cudaMemcpyHostToDevice);
        max_kernel<<<BLOCK_NUM, BLOCK_SIZE>>>(d_A, d_out, testMatrixHeight, testMatrixWidth);
        printf("kernel launch finished\n");
        cudaMemcpy(h_out, d_out, testMatrixHeight * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("kernel copy back failure");
        for (int i = 0; i < testMatrixHeight; i++)
            //if(h_out[i] != 2 * float(i) + 1.0f)
            printf("Mismatch in row %d , actual is %f but expect %f \n", i, h_out[i], 2 * float(i) + 1.0f);
    }
    #endif

    #define test_mat_mul
    #ifdef test_mat_mul
    {
        int M = 512;
        int K = 256;
        int N = 1024;

        // 1. init cuda
        int deviceId;
        cudaGetDevice(&deviceId);
        printf("Using CUDA device %d\n", deviceId);

        // 2. allocate host mem
        float* h_A = new float[M * K];
        float* h_B = new float[K * N];
        float* h_C_custom = new float[M * N];
        float* h_C_cublas = new float[M * N];

        // 3. generate data
        generateRandomMatrix(h_A, M, K);
        generateRandomMatrix(h_B, K, N);

        // 4. allocate device mem
        float* d_A, *d_B, *d_C_custom, *d_C_cublas;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C_custom, M * N * sizeof(float));
        cudaMalloc(&d_C_cublas, M * N * sizeof(float));

        // 5. copy from H to D
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // 6. call kernel
        dim3 blockDim(32, 32); // block size: 32x32
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // grim size
        matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C_custom, M, K, N);
        cudaDeviceSynchronize();

        // 7. call cublas
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
                    N, M, K,                  // B^T is N x K, A^T is K x M, result is N x M
                    &alpha,
                    d_B, N,                   // B^T has N rows
                    d_A, K,                   // A^T has K rows
                    &beta,
                    d_C_cublas, N);           // C^T has N rows

        // 8. copy from D2H
        cudaMemcpy(h_C_custom, d_C_custom, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // 9. comprare
        bool passed = compareMatrices(h_C_custom, h_C_cublas, M, N);
        if (passed) {
            printf("Test PASSED! Custom kernel matches cuBLAS result.\n");
        } else {
            printf("Test FAILED! Custom kernel differs from cuBLAS result.\n");
        }

        // 10. release
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_custom);
        cudaFree(d_C_cublas);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C_custom;
        delete[] h_C_cublas;
    }
    #endif
    return 0;
}