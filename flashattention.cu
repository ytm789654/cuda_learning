#include <stdio.h>
#include <float.h>

/*
calc O = scale * softmax(Q x K^T) x V, scale = 1/sqrt(d)
Q,K,V is N * d, O is N * d
L is 1-d vector with length N
use block size = 32 per block to handle 32 rows in Q in out loop
*/
#define BLOCK_SIZE 32

__global__ void flash_attention_kernel(const float *Q, const float *K, const float *V,
                                       const int N, const int d, const float scale,
                                       const int Bc, const int Tr, // for simple, assume Bc = Br = BLOCK_SIZE and Tr = Tc, and N = Bc * x
                                       float *O, float *L)
{
    // The reason for shared memory:
    // K_j, V_j can be re used in block, as each row will do dot product to each col
    // Q_i only used in specific thread, but used shared mem can let warp read data from memory.
    // S is too big to stored in thread register(related to Bc * Bc), so use shared memory to store.
    extern __shared__ float sram[]; //dynamic shared mem
    int tile_size = Bc * d;

    float *Q_i = sram;  //place Q_i at first, use Bc * d float
    float *K_j = &sram[tile_size];  //then place K_j, also use Bc * d
    float *V_j = &sram[tile_size * 2]; //then V_j, Bc * d
    float *O_i = &sram[tile_size * 3];  // then O_i, use Bc * d
    float *S = &sram[tile_size * 4]; //store S, Bc x Bc, and then update to P = exp(S - max(S))

    // out loop, each block handle one Q tile once
    for (int tile_id = blockIdx.x; tile_id < Tr; tile_id += gridDim.x)
    {
        // read Q_i from mem
        // init O_i as zero
        for (int row_bias = 0; row_bias < Bc; row_bias++)
        {
            int col = threadIdx.x;
            while(col < d){
                Q_i[row_bias * d + col] = Q[(tile_id * Bc + row_bias) * d + col];
                O_i[row_bias * d + col] = 0.0f;
                col += Bc;
            }
        }
        float m_j = -FLT_MAX;
        float prev_m_j = -FLT_MAX;
        float l_j = 0.0f;
        float prev_l_j = 0.0f;

        for (int j = 0; j < Tr; j++)
        {
            // read K_j, V_j from mem
            for (int row_bias = 0; row_bias < Bc; row_bias++)
            {
                int col = threadIdx.x;
                while(col < d){
                    K_j[row_bias * d + col] = K[(j * Bc + row_bias) * d + col];
                    V_j[row_bias * d + col] = V[(j * Bc + row_bias) * d + col];
                    col += Bc;
                }
            }
            __syncthreads();
            //calc S = Q x K^T, each thread calc one row in S
            //S[x][y] = Q_i[x] dot K_j[y]
            for (int K_jr = 0; K_jr < Bc; K_jr++)
            {
                float qk = 0.0f;
                for (int col = 0; col < d; col++)
                    qk += Q_i[threadIdx.x * d + col] * K_j[K_jr * d + col];
                qk = qk * scale;
                m_j = m_j > qk ? m_j : qk;
                S[threadIdx.x * Bc + K_jr] = qk;
            }

            //calc P = exp(S-rowmax(S)), rowmax(S) = m_j, store P into S
            //accumulate l_j at the same time
            l_j = 0.0f;
            for (int col = 0; col < Bc; col++)
            {
                float p = __expf(S[threadIdx.x * Bc + col] - m_j);
                l_j += p;
                S[threadIdx.x * Bc + col] = p;
            }
            //calc PV and accumulate into O_i
            for (int col = 0; col < d; col++)   //for each col in V
            {
                float pv = 0.0f;
                for (int V_jr = 0; V_jr < Bc; V_jr++)
                    pv += S[threadIdx.x * Bc + V_jr] * V_j[V_jr * d + col];
                O_i[threadIdx.x * d + col] = O_i[threadIdx.x * d + col] * __expf(prev_m_j - m_j) + pv;
            }
            
            //update l, m
            l_j += prev_l_j * __expf(prev_m_j - m_j);   //in fact init l_j in exp(m_j - prev_m_j) is the same, but this step follow algo
            prev_m_j = m_j;
            prev_l_j = l_j;
        }
        // write O_i to O, write m + log(l) to L
        // due to O rely on l_j and l_j stored in each thread, so each thread handle one row in O_i
        for (int col = 0; col < d; col++)
            O[(tile_id * Bc + threadIdx.x) * d + col] = O_i[threadIdx.x * d + col] / l_j;
        L[tile_id * Bc + threadIdx.x] = m_j + __logf(l_j);
    }
}

int main()
{
    return 0;
}