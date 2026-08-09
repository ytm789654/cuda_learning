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

int main()
{
    // code block for test max_kernel
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
    return 0;
}