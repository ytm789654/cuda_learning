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
        float prev_qk_row_max = -FLT_MAX;
        float qk_row_max = -FLT_MAX;
        float prev_l = 0;
        float l;

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
            // calc S = Q x K^T
            // use noob iteration first, guess should be many bank conflicts.
            int Qi_row = threadIdx.x;
            for (int K_jr = 0; K_jr < Bc; K_jr++)
            {
                float qk_dot_sum = 0.0f;
                for (int c = 0; c < d; c++)
                    qk_dot_sum += Q_i[Qi_row * d + c] * K_j[K_jr * d + c];
                qk_dot_sum *= scale;
                qk_row_max = qk_dot_sum > qk_row_max ? qk_dot_sum : qk_row_max;
                S[Qi_row * Bc + K_jr] = qk_dot_sum;
            }

            //calc P = exp(S - row_max(S))
            l = prev_l * __expf(prev_qk_row_max - qk_row_max);  //seems prev_l can be hidden.
            for (int c = 0; c < Bc; c++)
            {
                float exp_safe_S = __expf(S[Qi_row * Bc + c] - qk_row_max);
                S[Qi_row * Bc + c] = exp_safe_S;
                l += exp_safe_S;
            }

            //calc P x V_j, update O_i
            for (int c = 0; c < d; c++)    // for col in V_j
            {
                float pv_sum = 0.0f;
                for (int V_jr = 0; V_jr < Bc; V_jr++)   // for row in V_j
                    pv_sum += S[Qi_row * Bc + V_jr] * V_j[V_jr *d + c];    // stride for S is Bc, not d !!!
                O_i[Qi_row * d + c] = O_i[Qi_row * d + c] * __expf(prev_qk_row_max - qk_row_max) + pv_sum;    //scale O_i and add P x V
            }
            
            // calc m and l and update prev_m prev_l
            // qk_row_max = fmaxf(qk_row_max, prev_qk_row_max); no need to update row_max, it has been updated when calc S
            prev_qk_row_max = qk_row_max;
            prev_l = l;
        }
        // write O_i to O, write m + log(l) to L
        for (int col = 0; col < d; col++)
            // due to O rely on l, l stored in each thread, so let each thread write one row to O
            // better to update O_i, divide O_i by l, and then use warp to write back to memory.
            O[(tile_id * Bc + threadIdx.x) * d + col] = O_i[threadIdx.x * d + col] / l;
        L[tile_id * Bc + threadIdx.x] = qk_row_max + __logf(l);
    }
}

int main()
{
    return 0;
}