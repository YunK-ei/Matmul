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

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// problem scale limit: M = N or Results Mismatch
#define M 2048
#define N 2048
#define K 512
#define RANDOM_MIN 1
#define RANDOM_MAX 1

// block config

// depricated
// #define BLOCK_SIZE 16
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

void random_init(int *A, int *B, int dimm0, int dimm1, int dimm2){
    srand(unsigned(time(0)));
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm2; j++)
            A[i*dimm2+j] = RANDOM_MIN + (int)random() % (RANDOM_MAX - RANDOM_MIN + 1);

    for(int i=0; i<dimm2; i++)
        for(int j=0; j<dimm1; j++)
            B[i*dimm1+j] = RANDOM_MIN + (int)random() % (RANDOM_MAX - RANDOM_MIN + 1);
}

void zero_init(int *mat, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++)
            mat[i*dimm1+j] = 0;
}

// layout
// A: dimm0 * dimm2
// B: dimm2 * dimm1
// C: dimm0 * dimm1
void matmul_ref(int *A, int *B, int *C, int dimm0, int dimm1, int dimm2){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++)
            for(int k=0; k<dimm2; k++)
                C[i*dimm1 + j] += A[i*dimm2 + k] * B[k*dimm0 + j];
}

void matprint(int *A, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++){
        for(int j=0; j<dimm1; j++)
            printf("%d ", A[i*dimm1+j]);
        printf("\n");
    }
}

__global__ void naive_matmul(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    // grid-stride loop
    int row_stride_loop = 0;
    while(row_stride_loop * gridDim.x * blockDim.x < dimm0){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * blockDim.x;
        int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * blockDim.y < dimm1){
            int col_offset = col_stride_loop * gridDim.y * blockDim.y;
            int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
            int tmp = 0;
            // if(col < dimm0 && row < dimm1){
            //     for(int i=0; i<dimm2; i++)
            //         tmp += A[col*dimm2+i] * B[i*dimm1+row];
            //     C[col*dimm1+row] = tmp;
            // }
            if(row < dimm0 && col < dimm1){
                for(int i=0; i<dimm2; i++)
                    tmp += A[row*dimm2+i] * B[i*dimm1+col];
                C[row*dimm1+col] = tmp;
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }
}


__global__ void coalesced_matmul(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    int row_stride_loop = 0;
    while(row_stride_loop * gridDim.x * blockDim.x < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * blockDim.x;
        int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * blockDim.y < dimm0){
            int col_offset = col_stride_loop * gridDim.y * blockDim.y;
            int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
            int tmp = 0;
            // if(col < dimm0 && row < dimm1){
            //     for(int i=0; i<dimm2; i++)
            //         tmp += A[col*dimm2+i] * B[i*dimm1+row];
            //     C[col*dimm1+row] = tmp;
            // }
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<dimm2; i++)
                    tmp += A[col*dimm2+i] * B[i*dimm1+row];
                C[col*dimm1+row] = tmp;
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }
}


// limit: smem tile_size > block_size
#define MI BLOCK_SIZE_Y * 4
#define NI BLOCK_SIZE_X * 4
#define KI 32

constexpr auto UNROLL_FACTOR_OX{NI/BLOCK_SIZE_X};
constexpr auto UNROLL_FACTOR_OY{MI/BLOCK_SIZE_Y};

// inner product
__global__ void shared_matmul(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    __shared__ int smemA[MI][KI];
    __shared__ int smemB[KI][NI];
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
            int tmp[(MI)/BLOCK_SIZE_Y][(NI)/BLOCK_SIZE_X] = {{0}};
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
                        // if(tiy_A < (MI) && tix_A < (KI)) 
                        smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A) * dimm2 + i * (KI) + tix_A];
                    }
                    
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



// __global__ void shared_matmul_no_conflict(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
//     __shared__ int smemA[MI][KI+1];
//     __shared__ int smemB[KI][NI+1];
//     int row_stride_loop = 0;
//     // grid-stride loop
//     while(row_stride_loop * gridDim.x * blockDim.x < dimm0){
//         int col_stride_loop = 0;
//         int row_offset = row_stride_loop * gridDim.x * blockDim.x;
//         int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
//         while(col_stride_loop * gridDim.y * blockDim.y < dimm1){
//             int col_offset = col_stride_loop * gridDim.y * blockDim.y;
//             int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
//             int tmp = 0;
//             if(col < dimm0 && row < dimm1){
//                 for(int i=0; i<K/KI; i++){
//                     // move data to shared memory
                    // if(threadIdx.x < KI)
                    //     smemA[threadIdx.y][threadIdx.x] = A[col * dimm2 + i * KI + threadIdx.x];
                    // if(threadIdx.y < KI)
                    //     smemB[threadIdx.y][threadIdx.x] = B[(i * KI + threadIdx.y) * dimm1 + row_offset + blockIdx.x * blockDim.x + threadIdx.x];
                
//                     __syncthreads();
//                     // compute tile block in smem
//                     for(int i=0; i<KI; i++)
//                         tmp += smemA[threadIdx.y][i] * smemB[i][threadIdx.x];
//                     __syncthreads();
//                 }
//                 // write smemC back to global memory
//                 C[col*dimm1+row] = tmp;
//             }
//             col_stride_loop++;
//         }
//         row_stride_loop++;
//     }    
// }


// limit Mfrag = MI/BLOCK_SIZE_Y, Nfrag = NI/BLOCK_SIZE_X
#define Mfrag 4
#define Nfrag 4

constexpr auto UNROLL_FACTOR_M{Mfrag};
constexpr auto UNROLL_FACTOR_N{Nfrag};

// Outer Product
__global__ void shared_matmul_thread_tile(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    __shared__ int smemA[MI][KI];
    __shared__ int smemB[KI][NI];
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
            int fragM[Mfrag], fragN[Nfrag];
            int tmp[Mfrag * Nfrag] = {0};
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
                        // if(tiy_A < (MI) && tix_A < (KI)) 
                        smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A) * dimm2 + i * (KI) + tix_A];
                    }
                    
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
                }

                #pragma unroll UNROLL_FACTOR_M
                for(int m=0; m<Mfrag; m++){
                    #pragma unroll UNROLL_FACTOR_N
                    for(int n=0; n<Nfrag; n++){
                        C[(col_offset + blockIdx.y * MI + m + threadIdx.y * Mfrag)*dimm1 + row_offset + blockIdx.x * NI + n + threadIdx.x * Nfrag] = tmp[m * Nfrag + n];
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

// // Outer Product
// // TODO: #pragma unroll
// __global__ void shared_matmul_thread_tile_grid_unroll(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
//     __shared__ int smemA[4][MI][KI];
//     __shared__ int smemB[KI][NI];
//     int row_stride_loop = 0;
//     // grid-stride loop
//     assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
//     assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
//     while(row_stride_loop * gridDim.x * (NI) < dimm1){
//         int col_stride_loop = 0;
//         int row_offset = row_stride_loop * gridDim.x * (NI);
//         int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
//         while(col_stride_loop * gridDim.y * (MI) * 4 < dimm0){
//             int col_offset = col_stride_loop * gridDim.y * (MI);
//             int col = blockIdx.y * MI + threadIdx.y + col_offset;
//             int fragM[4][Mfrag], fragN[Nfrag];
//             int tmp[4][4] = {0};
//             if(col < dimm0 && row < dimm1){
//                 for(int i=0; i<K/KI; i++){
//                     // move data to shared memory
//                     for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
//                         int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
//                         int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
//                         // if(tiy_A < (MI) && tix_A < (KI)) 
//                         smemA[0][tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A + gridDim.y * MI * 0) * dimm2 + i * (KI) + tix_A];
//                         smemA[1][tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A + gridDim.y * MI * 1) * dimm2 + i * (KI) + tix_A];
//                         smemA[2][tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A + gridDim.y * MI * 2) * dimm2 + i * (KI) + tix_A];
//                         smemA[3][tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A + gridDim.y * MI * 3) * dimm2 + i * (KI) + tix_A];
//                     }
                    
//                     for(int j=0; j<(KI) * (NI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
//                         // Bugs from improper use of MACRO expand
//                         // int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
//                         // int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
//                         int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(NI);
//                         int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(NI);
//                         // if(tiy_B < (KI) && tix_B < (NI))
//                         smemB[tiy_B][tix_B] = B[(i * (KI) + tiy_B) * dimm1 + row_offset + blockIdx.x * (NI) + tix_B];
//                     }   
                
//                     __syncthreads();
//                     assert(MI / BLOCK_SIZE_Y == 2);
//                     assert(NI / BLOCK_SIZE_X == 2);
//                     // compute tile block in smem
//                     // #pragma unroll
//                     for(int j=0; j<KI; j++){
//                         fragM[0][0] = smemA[0][Mfrag * threadIdx.y + 0][j];
//                         fragM[0][1] = smemA[0][Mfrag * threadIdx.y + 1][j];
//                         fragM[1][0] = smemA[1][Mfrag * threadIdx.y + 0][j];
//                         fragM[1][1] = smemA[1][Mfrag * threadIdx.y + 1][j];
//                         fragM[2][0] = smemA[2][Mfrag * threadIdx.y + 0][j];
//                         fragM[2][1] = smemA[2][Mfrag * threadIdx.y + 1][j];
//                         fragM[3][0] = smemA[3][Mfrag * threadIdx.y + 0][j];
//                         fragM[3][1] = smemA[3][Mfrag * threadIdx.y + 1][j];
//                         fragN[0] = smemB[j][Nfrag * threadIdx.x + 0];
//                         fragN[1] = smemB[j][Nfrag * threadIdx.x + 1];
//                         tmp[0][0] += fragM[0][0] * fragN[0];
//                         tmp[0][1] += fragM[0][0] * fragN[1];
//                         tmp[0][2] += fragM[0][1] * fragN[0];
//                         tmp[0][3] += fragM[0][1] * fragN[1];
//                         tmp[1][0] += fragM[1][0] * fragN[0];
//                         tmp[1][1] += fragM[1][0] * fragN[1];
//                         tmp[1][2] += fragM[1][1] * fragN[0];
//                         tmp[1][3] += fragM[1][1] * fragN[1];
//                         tmp[2][0] += fragM[2][0] * fragN[0];
//                         tmp[2][1] += fragM[2][0] * fragN[1];
//                         tmp[2][2] += fragM[2][1] * fragN[0];
//                         tmp[2][3] += fragM[2][1] * fragN[1];
//                         tmp[3][0] += fragM[3][0] * fragN[0];
//                         tmp[3][1] += fragM[3][0] * fragN[1];
//                         tmp[3][2] += fragM[3][1] * fragN[0];
//                         tmp[3][3] += fragM[3][1] * fragN[1];
//                     }
//                 }
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 0)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[0][0];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 0)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[0][1];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 0)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[0][2];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 0)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[0][3];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 1)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[1][0];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 1)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[1][1];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 1)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[1][2];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 1)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[1][3];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 2)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[2][0];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 2)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[2][1];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 2)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[2][2];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 2)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[2][3];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 3)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[3][0];
//                 C[(col_offset + blockIdx.y * MI + 0 + gridDim.y * MI * 3)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[3][1];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 3)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp[3][2];
//                 C[(col_offset + blockIdx.y * MI + 1 + gridDim.y * MI * 3)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp[3][3];
//             }
//             col_stride_loop++;
//         }
//         row_stride_loop++;
//     }    
// }

// #define GRID_PARALLEL 2
// constexpr auto UNROLL_FACTOR_GRID{GRID_PARALLEL};

// __global__ void shared_matmul_thread_tile_grid_unroll(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
//     __shared__ int smemA[GRID_PARALLEL][MI][KI];
//     __shared__ int smemB[KI][NI];
//     int row_stride_loop = 0;
//     // grid-stride loop
//     // int loop_A = MI * KI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
//     // int loop_B = KI * NI / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
//     assert((MI) * (KI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0);
//     assert((KI) * (NI) % (BLOCK_SIZE_X * BLOCK_SIZE_Y) == 0); 
//     while(row_stride_loop * gridDim.x * (NI) < dimm1){
//         int col_stride_loop = 0;
//         int row_offset = row_stride_loop * gridDim.x * (NI);
//         int row = blockIdx.x * (NI) + threadIdx.x + row_offset;
//         while(col_stride_loop * gridDim.y * (MI) < dimm0){
//             int col_offset = col_stride_loop * gridDim.y * (MI);
//             int col = blockIdx.y * MI + threadIdx.y + col_offset;
//             int fragM[GRID_PARALLEL][Mfrag], fragN[Nfrag];
//             int tmp[GRID_PARALLEL][64] = {{0}};
//             if(col < dimm0 && row < dimm1){
//                 for(int i=0; i<K/KI; i++){
//                     // move data to shared memory
//                     for(int j=0; j<(MI) * (KI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
//                         int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(KI);
//                         int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(KI);
//                         // if(tiy_A < (MI) && tix_A < (KI))
//                         #pragma unroll UNROLL_FACTOR_GRID
//                         for(int k=0; k<GRID_PARALLEL; k++) 
//                             smemA[k][tiy_A][tix_A] = A[(col_offset + blockIdx.y * (MI) + tiy_A + gridDim.y * MI * k) * dimm2 + i * (KI) + tix_A];
//                     }
                    
//                     for(int j=0; j<(KI) * (NI) / (BLOCK_SIZE_X * BLOCK_SIZE_Y); j++){
//                         // Bugs from improper use of MACRO expand
//                         // int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
//                         // int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
//                         int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/(NI);
//                         int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%(NI);
//                         // if(tiy_B < (KI) && tix_B < (NI))
//                         smemB[tiy_B][tix_B] = B[(i * (KI) + tiy_B) * dimm1 + row_offset + blockIdx.x * (NI) + tix_B];
//                     }   
                
//                     __syncthreads();
//                     assert(MI / BLOCK_SIZE_Y == Mfrag);
//                     assert(NI / BLOCK_SIZE_X == Nfrag);
//                     // compute tile block in smem
//                     for(int j=0; j<KI; j++){
//                         assert(Mfrag == Nfrag);
//                         #pragma unroll UNROLL_FACTOR_M
//                         for(int k=0; k<Mfrag; k++){
//                             #pragma unroll UNROLL_FACTOR_GRID
//                             for(int gr=0; gr<GRID_PARALLEL; gr++){
//                                 fragM[gr][k] = smemA[gr][Mfrag * threadIdx.y + k][j];
//                             }
//                             fragN[k] = smemB[j][Nfrag * threadIdx.x + k];
//                         }

//                         // tmp[0] += fragM[0] * fragN[0];
//                         // tmp[1] += fragM[0] * fragN[1];
//                         // tmp[2] += fragM[1] * fragN[0];
//                         // tmp[3] += fragM[1] * fragN[1];

//                         #pragma unroll UNROLL_FACTOR_M
//                         for(int m=0; m<Mfrag; m++){
//                             #pragma unroll UNROLL_FACTOR_N
//                             for(int n=0; n<Nfrag; n++){
//                                 #pragma unroll UNROLL_FACTOR_GRID
//                                 for(int k=0; k<GRID_PARALLEL; k++)
//                                     tmp[k][m * Nfrag + n] += fragM[k][m] * fragN[n];
//                             }
//                         }
//                     }
//                 }

//                 #pragma unroll UNROLL_FACTOR_M
//                 for(int m=0; m<Mfrag; m++){
//                     #pragma unroll UNROLL_FACTOR_N
//                     for(int n=0; n<Nfrag; n++){
//                         #pragma unroll UNROLL_FACTOR_GRID
//                         for(int k=0; k<GRID_PARALLEL; k++)
//                             C[(col_offset + blockIdx.y * MI + m + threadIdx.y * Mfrag + gridDim.y * MI * k)*dimm1 + row_offset + blockIdx.x * NI + n + threadIdx.x * Nfrag] = tmp[k][m * Nfrag + n];
//                     }
//                 }
//                 // C[(col_offset + blockIdx.y * MI + 0)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp0;
//                 // C[(col_offset + blockIdx.y * MI + 0)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp1;
//                 // C[(col_offset + blockIdx.y * MI + 1)*dimm1 + row_offset + blockIdx.x * NI + 0] = tmp2;
//                 // C[(col_offset + blockIdx.y * MI + 1)*dimm1 + row_offset + blockIdx.x * NI + 1] = tmp3;
//             }
//             col_stride_loop++;
//         }
//         row_stride_loop++;
//     }    
// }

// outer product
__global__ void shared_matmul_doublebuffer(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
}

void check_result(int *ref, int *res, int dimm0, int dimm1){
    for(int i=0; i<dimm0; i++)
        for(int j=0; j<dimm1; j++){
            if(ref[i*dimm1+j] != res[i*dimm1+j]){
                printf("Mismatch Error: ref[%d][%d] = %d, res[%d][%d] = %d\n", i, j, ref[i*dimm1+j], i, j, res[i*dimm1+j]);
                exit(1);
            }
        }
    printf("Check Result: PASS\n");
}

int main(){
    // row major
    int *A = (int *) malloc(M*K*sizeof(int));
    int *B = (int *) malloc(K*N*sizeof(int));
    int *C = (int *) malloc(M*N*sizeof(int));
    int *ref = (int *) malloc(M*N*sizeof(int));
    int *d_A, *d_B, *d_C;
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


    // block config
    dim3 grid(16, 16);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    double iStart, iElaps;
    //==========================================================
    // start timer
    iStart = cpuSecond();
    // kernel launch
    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    iElaps = cpuSecond() - iStart;
    printf("naive kernel time elapsed: %f sec\n", iElaps);

    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);


    //=========================================================
    iStart = cpuSecond();
    // kernel launch
    coalesced_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    iElaps = cpuSecond() - iStart;
    printf("coalesced kernel time elapsed: %f sec\n", iElaps);

    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);


    //=========================================================
    iStart = cpuSecond();
    // kernel launch
    shared_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    iElaps = cpuSecond() - iStart;
    printf("shared kernel time elapsed: %f sec\n", iElaps);

    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);


    // //=========================================================
    // iStart = cpuSecond();
    // // kernel launch
    // shared_matmul_no_conflict<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // cudaDeviceSynchronize();
    // cudaCheckErrors("kernel launch failure");

    // iElaps = cpuSecond() - iStart;
    // printf("no conflict shared kernel time elapsed: %f sec\n", iElaps);

    // cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");
    // // check result
    // check_result(ref, C, M, N);

    //=========================================================
    iStart = cpuSecond();
    // kernel launch
    shared_matmul_thread_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    iElaps = cpuSecond() - iStart;
    printf("thread tile kernel time elapsed: %f sec\n", iElaps);

    cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    // check result
    check_result(ref, C, M, N);

    // //=========================================================
    // iStart = cpuSecond();
    // // kernel launch
    // shared_matmul_thread_tile_grid_unroll<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // cudaDeviceSynchronize();
    // cudaCheckErrors("kernel launch failure");

    // iElaps = cpuSecond() - iStart;
    // printf("thread tile parallel unroll kernel time elapsed: %f sec\n", iElaps);

    // cudaMemcpy(C, d_C, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");
    // // check result
    // check_result(ref, C, M, N);

    free(A);
    free(B);
    free(C);
    free(ref);
    return 0;
}