#include <stdlib.h>
#include <stdio.h>


#define TILE_DIM 32 

__global__ void tileMatMul(float* matA, float* matB, float* matC, int aRows,
      int aCols, 
      int bRows, int bCols, int cRows, int cCols)
{
   int Row = blockIdx.y * TILE_DIM + threadIdx.y;
   int Col = blockIdx.x * TILE_DIM + threadIdx.x;

   __shared__ float sharedMatA[TILE_DIM][TILE_DIM];
   __shared__ float sharedMatB[TILE_DIM][TILE_DIM];

   float cResultValue = 0.0;

   for(int i = 0; i < (aCols-1)/TILE_DIM+1; ++i)
   {
      if(Row < aRows && i*TILE_DIM+threadIdx.x < aCols)
      {
         sharedMatA[threadIdx.y][threadIdx.x] = matA[Row*aCols +
            i*TILE_DIM+threadIdx.x];
      }
      else
         sharedMatA[threadIdx.y][threadIdx.x] = 0.0;

      if(Col < bCols && i*TILE_DIM+threadIdx.y < cRows)
         sharedMatB[threadIdx.y][threadIdx.x] =
            matB[(i*TILE_DIM+threadIdx.y)*bCols+Col];
      else
         sharedMatB[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for(int j = 0; j < TILE_DIM; ++j)
         cResultValue += sharedMatA[threadIdx.y][j] *
            sharedMatB[j][threadIdx.x];
       
       __syncthreads();
   }

   if(Row < cRows && Col < cCols)
      matC[Row*cCols+Col] = cResultValue;

}


int main()
{
   float *hMatA, *hMatB, *hMatC;
   float *dMatA, *dMatB, *dMatC;

   int aRows = 512;
   int aCols = 512; 
   int bRows = 512;
   int bCols = 512; 
   int cRows, cCols;

   hMatA = (float *) malloc(sizeof(float)*aRows*aCols);
   hMatB = (float *) malloc(sizeof(float)*bRows*bCols);

   for(int i = 0; i < aRows*aCols; ++i)
   {
      hMatA[i] = (float)rand()/(float)(RAND_MAX/1.0);
      hMatB[i] = (float)rand()/(float)(RAND_MAX/1.0);
   }

   cRows = aRows;
   cCols = bCols;

   hMatC = (float *) malloc(sizeof(float)*cRows*cCols);

   cudaMalloc((void**)&dMatA, sizeof(float)*aRows*aCols);
   cudaMalloc((void**)&dMatB, sizeof(float)*bRows*bCols);
   cudaMalloc((void**)&dMatC, sizeof(float)*cRows*cCols);

   cudaMemcpy(dMatA, hMatA, sizeof(float)*aRows*aCols, cudaMemcpyHostToDevice);
   cudaMemcpy(dMatB, hMatB, sizeof(float)*bRows*bCols, cudaMemcpyHostToDevice);

   dim3 dimGrid((cCols - 1)/TILE_DIM+1, (cRows - 1)/TILE_DIM+1, 1);
   dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

   tileMatMul<<<dimGrid,dimBlock>>>(dMatA, dMatB, dMatC, aRows, aCols, bRows,
         bCols, cRows, cCols);

   cudaThreadSynchronize();

   cudaMemcpy(hMatC, dMatC, sizeof(float)*cRows*cCols, cudaMemcpyDeviceToHost);

   for(int q = 0; q < 100; ++q)
   {
      printf("Result matrix #%d: %f\n",q, hMatC[q]);
   }

   cudaFree(dMatA);
   cudaFree(dMatB);
   cudaFree(dMatC);

   free(hMatA);
   free(hMatB);
   free(hMatC);

   return 0;

}
