#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

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
#define RANDOM_MAX 4

// block config

// depricated
// #define BLOCK_SIZE 16
#define BLOCK_SIZE_X 16
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


// limit: smeme tile_size > block_size
#define MI BLOCK_SIZE_Y
#define NI BLOCK_SIZE_X
#define KI 32


__global__ void shared_matmul(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    __shared__ int smemA[MI][KI];
    __shared__ int smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    int loop_A = MI * KI / (blockDim.x * blockDim.y);
    int loop_B = KI * NI / (blockDim.x * blockDim.y); 
    while(row_stride_loop * gridDim.x * blockDim.x < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * blockDim.x;
        int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * blockDim.y < dimm0){
            int col_offset = col_stride_loop * gridDim.y * blockDim.y;
            int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
            int tmp = 0;
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    for(int j=0; j<loop_A; j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/KI;
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%KI;
                        if(tiy_A < MI && tix_A < KI) 
                            smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A) * dimm2 + i * KI + tix_A];
                    }
                    
                    for(int j=0; j<loop_B; j++){
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        if(tiy_B < KI && tix_B < NI)
                            smemB[tiy_B][tix_B] = B[(i * KI + tiy_B) * dimm1 + row_offset + blockIdx.x * blockDim.x + tix_B];
                    }   
                
                    __syncthreads();
                    // compute tile block in smem
                    for(int i=0; i<KI; i++)
                        tmp += smemA[threadIdx.y][i] * smemB[i][threadIdx.x];
                    __syncthreads();
                }
                // write smemC back to global memory
                C[col*dimm1+row] = tmp;
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

__global__ void shared_matmul_unroll(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    __shared__ int smemA[4][MI][KI];
    __shared__ int smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    int loop_A = MI * KI / (blockDim.x * blockDim.y);
    int loop_B = KI * NI / (blockDim.x * blockDim.y); 
    while(row_stride_loop * gridDim.x * blockDim.x < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * blockDim.x;
        int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * blockDim.y  * 4 < dimm0){
            int col_offset = col_stride_loop * gridDim.y * blockDim.y;
            int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
            int tmp0 = 0;
            int tmp1 = 0;
            int tmp2 = 0;
            int tmp3 = 0;
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    for(int j=0; j<loop_A; j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/KI;
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%KI;
                        if(tiy_A < MI && tix_A < KI){ 
                            smemA[0][tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A + gridDim.y * blockDim.y * 0) * dimm2 + i * KI + tix_A];
                            smemA[1][tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A + gridDim.y * blockDim.y * 1) * dimm2 + i * KI + tix_A];
                            smemA[2][tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A + gridDim.y * blockDim.y * 2) * dimm2 + i * KI + tix_A];
                            smemA[3][tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A + gridDim.y * blockDim.y * 3) * dimm2 + i * KI + tix_A];
                        }
                    }
                    
                    for(int j=0; j<loop_B; j++){
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        if(tiy_B < KI && tix_B < NI)
                            smemB[tiy_B][tix_B] = B[(i * KI + tiy_B) * dimm1 + row_offset + blockIdx.x * blockDim.x + tix_B];
                    }   
                
                    __syncthreads();
                    // compute tile block in smem
                    for(int i=0; i<KI; i++){
                        tmp0 += smemA[0][threadIdx.y][i] * smemB[i][threadIdx.x];
                        tmp1 += smemA[1][threadIdx.y][i] * smemB[i][threadIdx.x];
                        tmp2 += smemA[2][threadIdx.y][i] * smemB[i][threadIdx.x];
                        tmp3 += smemA[3][threadIdx.y][i] * smemB[i][threadIdx.x];
                    }
                    __syncthreads();
                }
                // write smemC back to global memory
                C[(col + gridDim.y * blockDim.y * 0)*dimm1+row] = tmp0;
                C[(col + gridDim.y * blockDim.y * 1)*dimm1+row] = tmp1;
                C[(col + gridDim.y * blockDim.y * 2)*dimm1+row] = tmp2;
                C[(col + gridDim.y * blockDim.y * 3)*dimm1+row] = tmp3;
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
}

__global__ void shared_matmul_reg(const int *A, const int *B, int *C, int dimm0, int dimm1, int dimm2){
    __shared__ int smemA[MI][KI];
    __shared__ int smemB[KI][NI];
    int row_stride_loop = 0;
    // grid-stride loop
    int loop_A = MI * KI / (blockDim.x * blockDim.y);
    int loop_B = KI * NI / (blockDim.x * blockDim.y); 
    while(row_stride_loop * gridDim.x * blockDim.x < dimm1){
        int col_stride_loop = 0;
        int row_offset = row_stride_loop * gridDim.x * blockDim.x;
        int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
        while(col_stride_loop * gridDim.y * blockDim.y < dimm0){
            int col_offset = col_stride_loop * gridDim.y * blockDim.y;
            int col = blockIdx.y * blockDim.y + threadIdx.y + col_offset;
            int tmp = 0;
            if(col < dimm0 && row < dimm1){
                for(int i=0; i<K/KI; i++){
                    // move data to shared memory
                    for(int j=0; j<loop_A; j++){
                        int tiy_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/KI;
                        int tix_A = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%KI;
                        if(tiy_A < MI && tix_A < KI) 
                            smemA[tiy_A][tix_A] = A[(col_offset + blockIdx.y * blockDim.y + tiy_A) * dimm2 + i * KI + tix_A];
                    }
                    
                    for(int j=0; j<loop_B; j++){
                        int tiy_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)/NI;
                        int tix_B = (j * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)%NI;
                        if(tiy_B < KI && tix_B < NI)
                            smemB[tiy_B][tix_B] = B[(i * KI + tiy_B) * dimm1 + row_offset + blockIdx.x * blockDim.x + tix_B];
                    }   
                
                    __syncthreads();
                    // compute tile block in smem
                    for(int i=0; i<KI; i++)
                        tmp += smemA[threadIdx.y][i] * smemB[i][threadIdx.x];
                    __syncthreads();
                }
                // write smemC back to global memory
                C[col*dimm1+row] = tmp;
            }
            col_stride_loop++;
        }
        row_stride_loop++;
    }    
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
    zero_init(C, M, N);
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
    shared_matmul_unroll<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel launch failure");

    iElaps = cpuSecond() - iStart;
    printf("parallel unroll kernel time elapsed: %f sec\n", iElaps);

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