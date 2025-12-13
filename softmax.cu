#include <stdio.h>
#include <float.h>


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

/*
using 32 thread per block to handle each row
*/
__global__ void safe_softmax_kernel(float *A, float *out, int M, int N)
{
    for (int row_idx = blockDim.x; row_idx < M; row_idx += blockDim.x)
    {
        __shared__ float row_max_val;
        __shared__ float exp_sum;
        
        // reduce to get row max
        int col_idx = threadIdx.x;
        float val = -FLT_MAX;
        while(col_idx < N)
        {
            val = fmaxf(val, A[row_idx * N + col_idx]);
            col_idx += blockDim.x;
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0)
            row_max_val = val;
        __syncthreads();

        // reduce to get safe exp sum
        col_idx = threadIdx.x;
        val = 0.0f;
        while(col_idx < N)
        {
            val += exp(A[row_idx * N + col_idx] - row_max_val);
            col_idx += blockDim.x;
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = val + __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0)
            exp_sum = val;
        __syncthreads();
        
        // divide exp_sum and write back to out
        col_idx = threadIdx.x;
        while (col_idx < N){
            out[row_idx * N + col_idx] = A[row_idx * N + col_idx] / exp_sum;
            col_idx += blockDim.x;
        }
    }
}

int main()
{
    // code block for test safe_softmax
    #define test_safe_softmax
    #ifdef test_safe_softmax
    {
        float *h_A, *d_A;
        float *h_out, *d_out;

        const int testMatrixHeight = 1024;
        const int testMatrixWidth = 4094;

        h_A = new float[testMatrixHeight * testMatrixWidth];
        for (int i = 0; i < testMatrixHeight; i++)
            for (int j = 0; j < testMatrixWidth; j++)
                h_A[i * testMatrixWidth + j] = 1.0f;
        h_out = new float[testMatrixHeight * testMatrixWidth];

        // allocate mem in GPU
        cudaMalloc(&d_A, testMatrixHeight * testMatrixWidth * sizeof(float));
        cudaMalloc(&d_out, testMatrixHeight * testMatrixWidth * sizeof(float));
        cudaCheckErrors("cudaMalloc failure");

        // copy data to device
        cudaMemcpy(d_A, h_A, testMatrixHeight * testMatrixWidth * sizeof(float), cudaMemcpyHostToDevice);

        // launch kernel
        dim3 grid_size = 256;
        dim3 block_size = 32;
        safe_softmax_kernel<<<grid_size, block_size>>>(d_A, d_out, testMatrixHeight, testMatrixWidth);
        printf("kernel launch finished\n");

        // copy result back to host
        cudaMemcpy(h_out, d_out, testMatrixHeight * testMatrixWidth * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("kernel copy back failure");
        printf("result copy finished\n");
        
        for (int i = 0; i < testMatrixHeight; i++)
            for (int j = 0; j < testMatrixWidth; j++)
                if(h_out[i * testMatrixWidth + j] - 1.0f/testMatrixWidth > 0.0001)
                    printf("error at row %d col %d, expect %f, but get %f\n", i, j, 1.0f/testMatrixWidth, h_out[i * testMatrixWidth + j]);
        printf("Run success! result[32][32] is %f ", h_out[32 * testMatrixWidth + 32]);
    }
    #endif
    return 0;
}