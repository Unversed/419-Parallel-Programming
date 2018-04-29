#include<stdlib.h>
#include<stdio.h>
#include <cuda_runtime.h>
#define seed 13
#define block_size 16


void printMatrix(float *matrix, int size){
   int i;
   for(i=0;i<size*size;i++){
      if(i%size == 0 && i != 0) 
         printf("\n");
      printf("%10.1f", matrix[i]);

   }
   printf("\n");
}

__global__ void version_1_matrixMul(float *dev_A, float *dev_B, float* dev_C,
      int N){
   // Each thread computes one elements of h_C
   // by accumulating results into dev_C
   float partial = 0.0;
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   int j = blockIdx.x * blockDim.x + threadIdx.x;
   int k;
   for(k =0; k < N;k++){
      partial += dev_A[i * N + k] * dev_B[k * N + j];
   }
   dev_C[i * N + j] = partial;
}

__global__ void version_2_matrixMul(float *dev_A, float *dev_B, float *dev_C,
      int matrix_size)
{
   __shared__ float A_tile[block_size][block_size];
   __shared__ float B_tile[block_size][block_size];

   float partial = 0.0;
   // block index
   int bx = blockIdx.x;
   int by = blockIdx.y;
   // thread index
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int row = by * blockDim.y + ty;
   int col = bx * blockDim.x + tx;
   // by the block

   int m;
   for( m=0 ; m < matrix_size/blockDim.x; m++){
      A_tile[ty][tx] = dev_A[row * matrix_size + (m * block_size + tx)];
      B_tile[ty][tx] = dev_B[col + (m * block_size + ty) * matrix_size];
      __syncthreads();
      int k;
      for(k=0; k< blockDim.x; k++)
         partial += A_tile[ty][k] * B_tile[k][tx];
      __syncthreads();
      dev_C[row * matrix_size + col] = partial;
   }

}

int main(int argc, char **argv){
   srand(seed);

   if(argc != 2){
      printf("Usage: \n");
      printf("/lab4 <matrixSize>");
      return 1;
   }
   int matrix_size = atoi(argv[1]);

   float *h_A = (float*) malloc(matrix_size * matrix_size * sizeof(float));
   float *h_B = (float*) malloc(matrix_size * matrix_size * sizeof(float));
   float *h_C = (float*) malloc(matrix_size * matrix_size * sizeof(float));

   int i,j;
   for(i=0;i<matrix_size;i++){
      for(j=0;j<matrix_size;j++){
         h_A[i * matrix_size + j] = (float)rand()/((float)RAND_MAX/10.0);
         h_B[i * matrix_size + j] = (float)rand()/((float)RAND_MAX/10.0);
      }
   }
   //printf("This is matrix A: %d\n", matrix_size);
   //printMatrix(h_A, matrix_size);
   //printf("This is matrix B: \n");
   //printMatrix(h_B, matrix_size);

   float *d_A, *d_B, *d_C;
   cudaMalloc((void**) &d_A, matrix_size * matrix_size * sizeof(float));
   cudaMalloc((void**) &d_B, matrix_size * matrix_size * sizeof(float));
   cudaMalloc((void**) &d_C, matrix_size * matrix_size * sizeof(float));

   dim3 Block(block_size, block_size, 1);
   dim3 Grid(matrix_size / Block.x, matrix_size / Block.y, 1);
   float elapsedTime;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   printf("=====This is naive version.======\n");
   cudaEventRecord(start, 0);
   cudaMemcpy(d_A, h_A, matrix_size * matrix_size * sizeof(float),
         cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, matrix_size * matrix_size * sizeof(float),
         cudaMemcpyHostToDevice);
   version_1_matrixMul<<< Grid, Block >>>(d_A, d_B, d_C, matrix_size);
   cudaMemcpy(h_C, d_C, matrix_size * matrix_size * sizeof(float),
         cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventElapsedTime(&elapsedTime, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("For naive version, the elapsed time is %.4f(ms).\n", elapsedTime);

   //printf("This is matrix C: \n");
   //printMatrix(h_C, matrix_size);

   printf("=====This is tiled version.======\n");
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   cudaMemcpy(d_A, h_A, matrix_size * matrix_size * sizeof(float),
         cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, matrix_size * matrix_size * sizeof(float),
         cudaMemcpyHostToDevice);
   version_2_matrixMul<<< Grid, Block >>>(d_A, d_B, d_C, matrix_size);
   cudaMemcpy(h_C, d_C, matrix_size* matrix_size * sizeof(float),
         cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("For tiled version, the elapsed time is %.4f(ms).\n", elapsedTime);

   //printf("This is matrix C: \n");
   //printMatrix(h_C, matrix_size);

   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   free(h_A);
   free(h_B);
   free(h_C);

   return 0;
}
