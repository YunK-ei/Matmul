#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

using namespace std;

// error checking macro
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


// problem scale limit: M = N or Results Mismatch
#define M 4096
#define N 4096
#define K 2048
#define RANDOM_MIN 1
#define RANDOM_MAX 8

// block config

// depricated
// #define BLOCK_SIZE 16
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

void random_init(float *A, float *B, int dimm0, int dimm1, int dimm2){
    srand(unsigned(time(0)));
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm2; j++)
            A[i*dimm2+j] = RANDOM_MIN + (int)random() % (RANDOM_MAX - RANDOM_MIN + 1);

    for(int i=0; i<dimm2; i++)
        for(int j=0; j<dimm1; j++)
            B[i*dimm1+j] = RANDOM_MIN + (int)random() % (RANDOM_MAX - RANDOM_MIN + 1);
}

void zero_init(float *mat, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++)
            mat[i*dimm1+j] = 0;
}

// layout
// A: dimm0 * dimm2
// B: dimm2 * dimm1
// C: dimm0 * dimm1
void matmul_ref(float *A, float *B, float *C, int dimm0, int dimm1, int dimm2){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++)
            for(int k=0; k<dimm2; k++)
                C[i*dimm1 + j] += A[i*dimm2 + k] * B[k*dimm0 + j];
}

void matprint(float *A, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++){
        for(int j=0; j<dimm1; j++)
            printf("%f ", A[i*dimm1+j]);
        printf("\n");
    }
}

// __global__ void naive_matmul(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
//     // grid-stride loop
//     int row_stride_loop = 0;
//     while(row_stride_loop * gridDim.x * blockDim.x < dimm0){
//         int col_stride_loop = 0;
//         int row_offset = row_stride_loop * gridDim.x * blockDim.x;
//         int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
//         while(col_stride_loop * gridDim.y * blockDim.y < dimm1){
//             int col_offset = col_stride_loop * gridDim.y * blockDim.y;
//             int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
//             float tmp = 0;
//             // if(col < dimm0 && row < dimm1){
//             //     for(int i=0; i<dimm2; i++)
//             //         tmp += A[col*dimm2+i] * B[i*dimm1+row];
//             //     C[col*dimm1+row] = tmp;
//             // }
//             if(row < dimm0 && col < dimm1){
//                 for(int i=0; i<dimm2; i++)
//                     tmp += A[row*dimm2+i] * B[i*dimm1+col];
//                 C[row*dimm1+col] = tmp;
//             }
//             col_stride_loop++;
//         }
//         row_stride_loop++;
//     }
// }


// __global__ void coalesced_matmul(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
//     int row_stride_loop = 0;
//     while(row_stride_loop * gridDim.x * blockDim.x < dimm1){
//         int col_stride_loop = 0;
//         int row_offset = row_stride_loop * gridDim.x * blockDim.x;
//         int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
//         while(col_stride_loop * gridDim.y * blockDim.y < dimm0){
//             int col_offset = col_stride_loop * gridDim.y * blockDim.y;
//             int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
//             float tmp = 0;
//             // if(col < dimm0 && row < dimm1){
//             //     for(int i=0; i<dimm2; i++)
//             //         tmp += A[col*dimm2+i] * B[i*dimm1+row];
//             //     C[col*dimm1+row] = tmp;
//             // }
//             if(col < dimm0 && row < dimm1){
//                 for(int i=0; i<dimm2; i++)
//                     tmp += A[col*dimm2+i] * B[i*dimm1+row];
//                 C[col*dimm1+row] = tmp;
//             }
//             col_stride_loop++;
//         }
//         row_stride_loop++;
//     }
// }


// limit: smem tile_size > block_size
#define MI 128
#define NI 128
#define KI 16

constexpr auto UNROLL_FACTOR_OX{NI/BLOCK_SIZE_X};
constexpr auto UNROLL_FACTOR_OY{MI/BLOCK_SIZE_Y};

// inner product
__global__ void shared_matmul(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
    __shared__ float smemA[MI][KI];
    __shared__ float smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    // int loop_A = MI * KI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    // int loop_B = KI * NI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
    assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
    while(row_stride_loop * gridDim.x * (NI) < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * (NI);
        int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * (MI) < dimm0){
            int col_offset = col_stride_loop * gridDim.y * (MI);
            int col = blockIdx.y * MI + threadIdx.y + col_offset;
            float tmp[(MI)/BLOCK_SIZE_Y][(NI)/BLOCK_SIZE_X] = {{0}};
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    #pragma unroll
                    for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
                        // if(tiy_A < (MI) && tix_A < (KI)) 
                        smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A) * dimm2 + i * (KI) + tix_A];
                    }
                    
                    #pragma unroll
                    for(int j=0; j<(KI) * (NI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        // Bugs from improper use of MACRO expand
                        // int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        // int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(NI);
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(NI);
                        // if(tiy_B < (KI) && tix_B < (NI))
                        smemB[tiy_B][tix_B] = B[(i * (KI) + tiy_B) * dimm1 + row_offset + blockIdx.x * (NI) + tix_B];
                    }   
                
                    __syncthreads();
                    // compute tile block in smem
                    #pragma unroll UNROLL_FACTOR_OX
                    for(int outer_x=0; outer_x < (NI)/BLOCK_SIZE_X; outer_x++){
                        #pragma unroll UNROLL_FACTOR_OY
                        for(int outer_y=0; outer_y < (MI)/BLOCK_SIZE_Y; outer_y++){
                            for(int inner_loop=0; inner_loop<KI; inner_loop++)
                                tmp[outer_y][outer_x] += smemA[threadIdx.y + outer_y * blockDim.y][inner_loop] * smemB[inner_loop][threadIdx.x + outer_x * blockDim.x];
                        }
                    }
                    __syncthreads();
                }
                // write smemC back to global memory
                #pragma unroll UNROLL_FACTOR_OX
                for(int outer_x=0; outer_x < NI/BLOCK_SIZE_X; outer_x++){
                    #pragma unroll UNROLL_FACTOR_OY
                    for(int outer_y=0; outer_y < MI/BLOCK_SIZE_Y; outer_y++){
                        C[(col + outer_y * blockDim.y)*dimm1+ row + outer_x * blockDim.x] = tmp[outer_y][outer_x];
                    }
                }
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
}





#define Mfrag 8
#define Nfrag 8

constexpr auto UNROLL_FACTOR_M{Mfrag};
constexpr auto UNROLL_FACTOR_N{Nfrag};

// Outer Product
// limit Mfrag = MI/BLOCK_SIZE_Y, Nfrag = NI/BLOCK_SIZE_X
__global__ void shared_matmul_thread_tile(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
    __shared__ float smemA[MI][KI];
    __shared__ float smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    // int loop_A = MI * KI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    // int loop_B = KI * NI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
    assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
    while(row_stride_loop * gridDim.x * (NI) < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * (NI);
        int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * (MI) < dimm0){
            int col_offset = col_stride_loop * gridDim.y * (MI);
            int col = blockIdx.y * MI + threadIdx.y + col_offset;
            float fragM[Mfrag], fragN[Nfrag];
            float tmp[Mfrag * Nfrag] = {0};
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory

                    #pragma unroll
                    for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
                        // if(tiy_A < (MI) && tix_A < (KI)) 
                        smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A) * dimm2 + i * (KI) + tix_A];
                    }
                    
                    #pragma unroll
                    for(int j=0; j<(KI) * (NI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        // Bugs from improper use of MACRO expand
                        // int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        // int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(NI);
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(NI);
                        // if(tiy_B < (KI) && tix_B < (NI))
                        smemB[tiy_B][tix_B] = B[(i * (KI) + tiy_B) * dimm1 + row_offset + blockIdx.x * (NI) + tix_B];
                    }   
                
                    __syncthreads();
                    assert(MI / BLOCK_SIZE_Y == Mfrag);
                    assert(NI / BLOCK_SIZE_X == Nfrag);
                    // compute tile block in smem
                    for(int j=0; j<KI; j++){
                        #pragma unroll UNROLL_FACTOR_M
                        for(int k=0; k<Mfrag; k++)
                            fragM[k] = smemA[Mfrag * threadIdx.y + k][j];
                        #pragma unroll UNROLL_FACTOR_N
                        for(int k=0; k<Nfrag; k++)
                            fragN[k] = smemB[j][Nfrag * threadIdx.x + k];


                        #pragma unroll UNROLL_FACTOR_M
                        for(int m=0; m<Mfrag; m++){
                            #pragma unroll UNROLL_FACTOR_N
                            for(int n=0; n<Nfrag; n++){
                                tmp[m * Nfrag + n] += fragM[m] * fragN[n];
                            }
                        }
                    }
                }

                #pragma unroll UNROLL_FACTOR_M
                for(int m=0; m<Mfrag; m++){
                    #pragma unroll UNROLL_FACTOR_N
                    for(int n=0; n<Nfrag; n++){
                        C[(col_offset + blockIdx.y * MI + m + threadIdx.y * Mfrag)*dimm1 + row_offset + blockIdx.x * NI + n + threadIdx.x * Nfrag] = tmp[m * Nfrag + n];
                    }
                }
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
}

#define WARP_X 8
#define WARP_Y 4
#define NUM_WARP (BLOCK_SIZE_X * BLOCK_SIZE_Y / 32)
#define NUM_WARP_X (NI/(WARP_X * Nfrag))
#define NUM_WARP_Y (MI/(WARP_Y * Mfrag))

__global__ void shared_matmul_warp_tile(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
    assert(NI % (Nfrag * WARP_X) == 0);
    assert(MI % (Mfrag * WARP_Y) == 0);
    assert (WARP_X * WARP_Y == 32);
    // force alignment
    assert (BLOCK_SIZE_X * Nfrag * BLOCK_SIZE_Y * Mfrag == NI * MI);
    __shared__ float smemA[MI][KI];
    __shared__ float smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    // int loop_A = MI * KI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    // int loop_B = KI * NI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
    assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
    while(row_stride_loop * gridDim.x * (NI) < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * (NI);
        int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * (MI) < dimm0){
            int col_offset = col_stride_loop * gridDim.y * (MI);
            int col = blockIdx.y * MI + threadIdx.y + col_offset;
            float fragM[Mfrag], fragN[Nfrag];
            float tmp[Mfrag * Nfrag] = {0};
            int warpIdx = (threadIdx.y * blockDim.x + threadIdx.x)/32;
            // thread cordinate inside the warp
            int warpx = (threadIdx.y * blockDim.x + threadIdx.x)%32%WARP_X;
            int warpy = (threadIdx.y * blockDim.x + threadIdx.x)%32/WARP_X;
            // warp tile cordinate inside the tile
            int wtilex = warpIdx % (NUM_WARP_X);
            int wtiley = warpIdx / (NUM_WARP_X);
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    #pragma unroll
                    for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
                        // if(tiy_A < (MI) && tix_A < (KI)) 
                        smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A) * dimm2 + i * (KI) + tix_A];
                    }
                    
                    #pragma unroll
                    for(int j=0; j<(KI) * (NI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        // Bugs from improper use of MACRO expand
                        // int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        // int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(NI);
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(NI);
                        // if(tiy_B < (KI) && tix_B < (NI))
                        smemB[tiy_B][tix_B] = B[(i * (KI) + tiy_B) * dimm1 + row_offset + blockIdx.x * (NI) + tix_B];
                    }   
                
                    __syncthreads();
                    // eliminate the limit
                    // assert(MI / BLOCK_SIZE_Y == Mfrag);
                    // assert(NI / BLOCK_SIZE_X == Nfrag);
                    // compute tile block in smem
                    for(int j=0; j<KI; j++){
                        #pragma unroll UNROLL_FACTOR_M
                        for(int k=0; k<Mfrag; k++)
                            fragM[k] = smemA[Mfrag * (warpy + wtiley * WARP_Y) + k][j];
                        #pragma unroll UNROLL_FACTOR_N
                        for(int k=0; k<Nfrag; k++)
                            fragN[k] = smemB[j][Nfrag * (warpx + wtilex * WARP_X) + k];

                        // tmp[0] += fragM[0] * fragN[0];
                        // tmp[1] += fragM[0] * fragN[1];
                        // tmp[2] += fragM[1] * fragN[0];
                        // tmp[3] += fragM[1] * fragN[1];

                        #pragma unroll UNROLL_FACTOR_M
                        for(int m=0; m<Mfrag; m++){
                            #pragma unroll UNROLL_FACTOR_N
                            for(int n=0; n<Nfrag; n++){
                                tmp[m * Nfrag + n] += fragM[m] * fragN[n];
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll UNROLL_FACTOR_M
                for(int m=0; m<Mfrag; m++){
                    #pragma unroll UNROLL_FACTOR_N
                    for(int n=0; n<Nfrag; n++){
                        C[(col_offset + blockIdx.y * MI + m + (warpy + wtiley * WARP_Y) * Mfrag)*dimm1 + row_offset + blockIdx.x * NI + n + (warpx + wtilex * WARP_X) * Nfrag] = tmp[m * Nfrag + n];
                    }
                }
                // C[(col_offset + blockIdx.y * MI + 0)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp0;
                // C[(col_offset + blockIdx.y * MI + 0)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp1;
                // C[(col_offset + blockIdx.y * MI + 1)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp2;
                // C[(col_offset + blockIdx.y * MI + 1)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp3;
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
}

__global__ void shared_matmul_thread_tile_load_opt(const float *A, const float *B, float *C, int dimm0, int dimm1, int dimm2){
    // allocate 4 consecutive elements as a tile for a thread
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // num of threads for loading a row
    const int A_TILE_THREAD_PER_ROW = KI / 4;
    const int B_TILE_THREAD_PER_ROW = NI / 4;

    // start row number in a stride that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = (BLOCK_SIZE_X * BLOCK_SIZE_Y) / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = (BLOCK_SIZE_X * BLOCK_SIZE_Y) / B_TILE_THREAD_PER_ROW;
    __shared__ float smemA[MI][KI];
    __shared__ float smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    // int loop_A = MI * KI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    // int loop_B = KI * NI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
    assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
    while(row_stride_loop * gridDim.x * (NI) < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * (NI);
        int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * (MI) < dimm0){
            int col_offset = col_stride_loop * gridDim.y * (MI);
            int col = blockIdx.y * MI + threadIdx.y + col_offset;
            float fragM[Mfrag], fragN[Nfrag];
            float tmp[Mfrag * Nfrag] = {0};
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    #pragma unroll
                    for(int j=0; j<MI; j+=A_TILE_ROW_STRIDE){
                        // if(tiy_A < (MI) && tix_A < (KI))
                        #pragma unroll
                        for(int k=0; k<4; k++){
                            assert(A_TILE_ROW_START+j < MI);
                            assert(A_TILE_COL+k < KI); 
                            smemA[A_TILE_ROW_START+j][A_TILE_COL+k] = A[(col_offset + blockIdx.y * (MI) + A_TILE_ROW_START + j) * dimm2 + i * (KI) + A_TILE_COL + k];
                        }
                    }
                    
                    #pragma unroll
                    for(int j=0; j<KI; j+=B_TILE_ROW_STRIDE){
                        // if(tiy_B < (KI) && tix_B < (NI))
                        #pragma unroll
                        for(int k=0; k<4; k++){
                            assert(B_TILE_ROW_START+j < KI);
                            assert(B_TILE_COL+k < NI);
                            smemB[B_TILE_ROW_START+j][B_TILE_COL+k] = B[(i * (KI) + B_TILE_ROW_START+j) * dimm1 + row_offset + blockIdx.x * (NI) + B_TILE_COL + k];
                        }
                    }   
                
                    __syncthreads();
                    assert(MI / BLOCK_SIZE_Y == Mfrag);
                    assert(NI / BLOCK_SIZE_X == Nfrag);
                    // compute tile block in smem
                    for(int j=0; j<KI; j++){
                        #pragma unroll UNROLL_FACTOR_M
                        for(int k=0; k<Mfrag; k++)
                            fragM[k] = smemA[Mfrag * threadIdx.y + k][j];
                        #pragma unroll UNROLL_FACTOR_N
                        for(int k=0; k<Nfrag; k++)
                            fragN[k] = smemB[j][Nfrag * threadIdx.x + k];


                        #pragma unroll UNROLL_FACTOR_M
                        for(int m=0; m<Mfrag; m++){
                            #pragma unroll UNROLL_FACTOR_N
                            for(int n=0; n<Nfrag; n++){
                                tmp[m * Nfrag + n] += fragM[m] * fragN[n];
                            }
                        }
                    }
                    __syncthreads();
                }

                #pragma unroll UNROLL_FACTOR_M
                for(int m=0; m<Mfrag; m++){
                    #pragma unroll UNROLL_FACTOR_N
                    for(int n=0; n<Nfrag; n++){
                        C[(col_offset + blockIdx.y * MI + m + threadIdx.y * Mfrag)*dimm1 + row_offset + blockIdx.x * NI + n + threadIdx.x * Nfrag] = tmp[m * Nfrag + n];
                    }
                }
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
}


void check_result(float *ref, float *res, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++){
            if(ref[i*dimm1+j] != res[i*dimm1+j]){
                printf("Mismatch Error: ref[%d][%d] = %f, res[%d][%d] = %f\n", i, j, ref[i*dimm1+j], i, j, res[i*dimm1+j]);
                return;
            }
        }
    printf("Check Result: PASS\n");
}

int main(){
    // row major
    float *A = (float *) malloc(M*K*sizeof(float));
    float *B = (float *) malloc(K*N*sizeof(float));
    float *C = (float *) malloc(M*N*sizeof(float));
    float *ref = (float *) malloc(M*N*sizeof(float));
    float *d_A, *d_B, *d_C;
    // init & ref result
    random_init(A, B, M, N, K);
    zero_init(ref, M, N);
    // zero_init(C, M, N);
    matmul_ref(A, B, ref, M, N, K);

    // // print for debug
    // matprint(A, M, K);
    // printf("\n");
    // matprint(B, K, N);
    // printf("\n");
    // matprint(ref, M, N);
    // printf("\n");
    // copy data to the device
    cudaMalloc(&d_A, M*K*sizeof(int));
    cudaMalloc(&d_B, K*N*sizeof(int));
    cudaMalloc(&d_C, M*N*sizeof(int));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, A, M*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // block config
    dim3 grid(32, 32);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // compile for ncu-ui, set iter = 1
    // int iter = 10;
    int iter = 1;
    float time_ms = 0.f;
    long ops = (long)M * N * K * 2;
    double gops;
    // //==========================================================
    // // start timer
    // // kernel launch
    // // warmup
    // naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // cudaEventRecord(start);
    // for(int i = 0; i < iter; i++){
    //     naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_ms, start, stop);
    // cudaCheckErrors("kernel launch failure");
    // gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    // printf("naive matmul:%f Gops\n", gops);
    // cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");
    // // check result
    // check_result(ref, C, M, N);

    // //==========================================================
    // // start timer
    // // kernel launch
    // // warmup
    // coalesced_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // cudaEventRecord(start);
    // for(int i = 0; i < iter; i++){
    //     coalesced_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_ms, start, stop);
    // cudaCheckErrors("kernel launch failure");
    // gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    // printf("coalesced matmul:%f Gops\n", gops);
    // cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");
    // // check result
    // check_result(ref, C, M, N);

    //==========================================================
    // start timer
    // kernel launch
    // warmup
    shared_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        shared_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaCheckErrors("kernel launch failure");
    gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("shared matmul:%f Gops\n", gops);
    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);

    //==========================================================
    // start timer
    // kernel launch
    // warmup
    shared_matmul_thread_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        shared_matmul_thread_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaCheckErrors("kernel launch failure");
    gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("shared matmul thread tile:%f Gops\n", gops);
    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);

    //==========================================================
    // start timer
    // kernel launch
    // warmup
    shared_matmul_warp_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        shared_matmul_warp_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaCheckErrors("kernel launch failure");
    gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("shared matmul warp tile:%f Gops\n", gops);
    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);

    //==========================================================
    // start timer
    // kernel launch
    // warmup
    shared_matmul_thread_tile_load_opt<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        shared_matmul_thread_tile_load_opt<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaCheckErrors("kernel launch failure");
    gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3);
    printf("shared matmul thread tile load opt:%f Gops\n", gops);
    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);

    free(A);
    free(B);
    free(C);
    free(ref);
    return 0;
}