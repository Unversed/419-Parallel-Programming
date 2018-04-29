#include<stdlib.h>
#include<stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define seed 13

__global__ void matrixMul(float *dev_A, float *dev_B, float *dev_C, int
      matrixWitdh)
{
   __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
   __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

   int bx = blockIdx.x;
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int row = by * TILE_WIDTH + ty;
   int col = bx * TILE_WIDTH + tx;

   float partial = 0.0;
   int m;
   for( m=0 ; m < matrixWitdh/TILE_WIDTH; m++){
      A_tile[ty][tx] = dev_A[row * matrixWitdh + (m * TILE_WIDTH + tx)];
      B_tile[ty][tx] = dev_B[col + (m * TILE_WIDTH + ty) * matrixWitdh];
      __syncthreads();
      int k;
      for(k=0; k< TILE_WIDTH; k++)
         partial += A_tile[ty][k] * B_tile[k][tx];
      __syncthreads();
      dev_C[row * matrixWitdh + col] = partial;
   }

}

int main(int argc, char **argv){
   srand(seed);

   if(argc != 2){
      printf("Usage /lab4_4 <matrixWitdh>");
      return 1;
   }
   int matrixWitdh = atoi(argv[1]);

   float *h_A = (float*) malloc(matrixWitdh * matrixWitdh * sizeof(float));
   float *h_B = (float*) malloc(matrixWitdh * matrixWitdh * sizeof(float));
   float *h_C = (float*) malloc(matrixWitdh * matrixWitdh * sizeof(float));

   int i,j;
   for(i=0;i<matrixWitdh;i++){
      for(j=0;j<matrixWitdh;j++){
         h_A[i * matrixWitdh + j] = (float)rand()/((float)RAND_MAX/10.0);
         h_B[i * matrixWitdh + j] = (float)rand()/((float)RAND_MAX/10.0);
      }
   }

   float *d_A, *d_B, *d_C;
   cudaMalloc((void**) &d_A, matrixWitdh * matrixWitdh * sizeof(float));
   cudaMalloc((void**) &d_B, matrixWitdh * matrixWitdh * sizeof(float));
   cudaMalloc((void**) &d_C, matrixWitdh * matrixWitdh * sizeof(float));

   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 2);
   dim3 dimGrid(matrixWitdh/TILE_WIDTH, matrixWitdh/TILE_WIDTH, 1);

   float elapsedTime;
   cudaEvent_t start, stop;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   cudaMemcpy(d_A, h_A, matrixWitdh * matrixWitdh * sizeof(float),
         cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, matrixWitdh * matrixWitdh * sizeof(float),
         cudaMemcpyHostToDevice);
   matrixMul<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, matrixWitdh);
   cudaMemcpy(h_C, d_C, matrixWitdh* matrixWitdh * sizeof(float),
         cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("For tiled version, the elapsed time is %.4f(ms).\n", elapsedTime);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   free(h_A);
   free(h_B);
   free(h_C);

   return 0;
}
