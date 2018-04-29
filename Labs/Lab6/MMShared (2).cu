/*
 ============================================================================
 Name        : MMShared.cu
 Author      : Liam Lefferts
 Version     : 1.0
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

// System includes
#include <stdio.h>
#include<stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>

#define TILE_WIDTH 16

typedef struct 
{
    int numRows;
    int numCols;
    int stride;
    float * elements;
} Matrix;



// Fill matrix elements
void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}



//
// Matrix Multiplication CPU for error checking
//
void matrixmult(float *fa, float *fb, float *fc,int Hight, int Width){
	int row, col, k;
	float Pvalue=0;
	for (row=0; row<Hight; row++){
		for(col=0; col<Width; col++) {
        	Pvalue=0;
			for(k=0; k<Width; k++){
				Pvalue+=fa[row*Width+k]*fb[k*Width+col];
            }
			fc[row*Width+col]=Pvalue;
         }
	}
}




//Compute C=A*B in GPU non shared memory
__global__ void matrixMultiply(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //identify row and column to work on
  	int row=blockIdx.y*blockDim.y+threadIdx.y;
  	int col=blockIdx.x*blockDim.x+threadIdx.x;
  	int i= row*numCColumns+col;
  	float Pvalue=0; int k;
  	if(row<numARows && col<numBColumns){
  		for(k=0; k<numBColumns; k++){
			Pvalue+=A[row*numAColumns+k]*B[k*numBColumns+col];
  		}
  		C[i]=Pvalue;
  	}
}




/* matrixMultiplyShared -
 * Compute C = A*B in GPU shared memory
 *
 * Takes: 
 * Matrix d_A, d_B;              matrices to compute
 * Matrix d_C;                   result matrix 
 *
 * Returns: void
 */
__global__ void 
matrixMultiplyShared(Matrix d_A, Matrix d_B, Matrix d_C) 
{  
// Each thread block computes one sub-matrix sub_C of d_C
    Matrix sub_C;
    sub_C.numCols = sub_C.numRows = TILE_WIDTH;
    sub_C.stride = d_C.stride;
    sub_C.elements = &d_C.elements[d_C.stride 
                                    * TILE_WIDTH * blockIdx.y 
                                    + TILE_WIDTH * blockIdx.x];
  
  // Each thread computes one element of sub_C by accumulating results into Pvalue
  float Pvalue = 0.0;

  // Loop over all the sub-matrices of d_A and d_B that are required to compute sub_C
  // Multiply each pair of sub-matrices together and accumulate the results
  for (int m = 0; m < (d_A.numCols / TILE_WIDTH); ++m) {
    //Write sub-matrix sub_A
    Matrix sub_A;
    sub_A.numCols = sub_A.numRows = TILE_WIDTH;
    sub_A.stride = d_A.stride;
    sub_A.elements = &d_A.elements[d_A.stride 
                                    * TILE_WIDTH * blockIdx.y 
                                    + TILE_WIDTH * blockIdx.x];

   //Write sub-matrix sub_B
    Matrix sub_B;
    sub_B.numCols = sub_B.numRows = TILE_WIDTH;
    sub_B.stride = d_B.stride;
    sub_B.elements = &d_B.elements[d_B.stride 
                                    * TILE_WIDTH * blockIdx.y 
                                    + TILE_WIDTH * blockIdx.x];


    // Shared memory used to store sum_A and sum_B respectively
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    // Each thread loads one element of sum_A and sum_B from device to shared memory
    sharedA[threadIdx.y][threadIdx.x] = 
            sum_A.elements[threadIdx.y * A.stride + threadIdx.x];
    sharedB[threadIdx.y][threadIdx.x] = 
            sum_B.elements[threadIdx.y * B.stride + threadIdx.x];
    
    
    // Synchronize to ensure sub-matrices loaded
    __syncthreads();
    
    // Multiplicatio logic
    for (int i = 0; i < TILE_WIDTH; ++i)
      Pvalue += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
      
    // Synchronize to ensures sub-matricies computed
    __syncthreads();
  }

  // Each thread writes one sub_C element to global device memory
  sub_C.elements[threadIdx.y * sub_C.stride + threadIdx.x] = Pvalue;
}

/* printMatricesCheck
 * Prints matrices check status to stdout
 *
 * Takes: 
 * float * A, B;                 matrices to compare 
 * int numCRows, numCColumns;    matrix dimensions
 *
 * Returns: void
 */
void
printMatricesCheck(float * A, float * B, int numCRows, int numCColumns)
{
   float accum = 0;
   for (int i = 0; i < numCRows * numCColumns; ++i)
   {
    accum+=abs(A[i]-B[i]);
    if(accum!=0)
      printf("FAILED\n");
   }
   
   if (accum == 0)
   { 
      printf("Matrices match...\n");
      printf("\nSUCCESSFUL!\n");
	
   // print MM result
      for (int i = 0; i < numCRows * numCColumns; ++i)
      {
         if(i % numCColumns)
            printf("\n");
         printf("%lf ", A[i]);

      }
   }
}

int main(int argc, char ** argv) 
{

   int   numARows, numAColumns, 
         numBRows, numBColumns,
         numCRows, numCColumns;
   
   if(argc != 4)
   {
      printf("Usage: %s <matrix A row count>, <matrix A column count>, <matrix B column count>", argv[0]);
      exit(1);
   }

   // number of Matrix A rows must equal Matrix C rows
   numCRows    = numARows     = atoi(argv[1]);
   // number of Matrix A columns must equal Matrix B rows 
   numAColumns = numBRows     = atoi(argv[2]);
   // number of Matrix B columns must equal Matric C columns 
   numCColumns = numBColumns  = atoi(argv[3]); 
   
   printf("A[%d x %d] * B[%d x %d] = C[%d x %d]", 
   numARows, numAColumns, 
   numBRows, numBColumns, 
   numCRows, numCColumns);
   
   //Dimensions requirment check MM
	if(numAColumns != numBRows)
   {
		printf("numAColumns != numBRows, This matrix cannot be multiplied");
      exit(1);
	}
   
   //alloc host memory
	float *hostA = new float[numARows*numAColumns]; //input matrix A
   float *hostB = new float[numBRows*numBColumns]; //input matrix B
   float *hostC = new float[numCRows*numCColumns]; //output matrix C
   float *hostD = new float[numCRows*numCColumns]; //output matrix D
   float *hostE = new float[numCRows*numCColumns]; //output matrix E
	
   initialize (hostA, numARows*numAColumns);
   initialize (hostB, numBRows*numBColumns);
	
   //do MM on CPU for timing
	matrixmult(hostA, hostB, hostC, numCRows, numCColumns);

	
	//device variables
	float * deviceA;
	float * deviceB;
	float * deviceC;

   //Determine matrix memory sizes
	unsigned int size_A = numARows * numAColumns;
    	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int size_B = numBRows * numBColumns;
    	unsigned int mem_size_B = sizeof(float) * size_B;
   unsigned int size_C = numCRows * numCColumns;
    	unsigned int mem_size_C = sizeof(float) * size_C;

   //Holds error value
   cudaError_t err; 
   
   //Allocate GPU memory
   err = cudaMalloc((void**) &deviceA, mem_size_A);
   printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
   err = cudaMalloc((void**) &deviceB, mem_size_B);
   printf("CUDA malloc B: %s\n", cudaGetErrorString(err)); 	
   err = cudaMalloc((void**) &deviceC, mem_size_C);
   printf("CUDA malloc C: %s\n", cudaGetErrorString(err));


   //Copy memory to the GPU
	err = cudaMemcpy(deviceA, hostA, mem_size_A, cudaMemcpyHostToDevice);
   printf("Copy A off host: %s\n", cudaGetErrorString(err));
   err = cudaMemcpy(deviceB, hostB, mem_size_B, cudaMemcpyHostToDevice);
   printf("Copy B off host: %s\n", cudaGetErrorString(err));


   //Initialize grid and block dimensions
   dim3 threads(TILE_WIDTH, TILE_WIDTH);
   dim3 grid(numBColumns / threads.x, numARows / threads.y);
	
	//MM without shared memory
   matrixMultiply<<< grid, threads>>>(deviceA, deviceB, deviceC);

   //Wait for all previously issued device commands before continuing 
   err = cudaDeviceSynchronize();
   printf("Run nsMM kernel: %s\n", cudaGetErrorString(err));
    	
   //Copy the GPU memory back to the CPU here
   err = cudaMemcpy(hostE, deviceC, mem_size_C, cudaMemcpyDeviceToHost);
   printf("Copy C off of device: %s\n", cudaGetErrorString(err));
   
   printMatricesCheck(hostC, hostE, numCRows, numCColumns);
   
    //Initialize matrix structures
    Matrix d_A;
    d_A.numCols = d_A.stride = numAColumns;
    d_A.numRows = numARows;
    d_A.elements = deviceA;
    
    Matrix d_B;
    d_B.numCols = d_B.stride = numBColumns;
    d_B.numRows = numBRows;
    d_B.elements = deviceB;
    
    Matrix d_D;
    d_D.numCols = d_D.stride = numCColumns;
    d_D.numRows = numCRows;
    d_D.elements = deviceC;
    
   //Initialize grid and block dimensions
   dim3 threads(TILE_WIDTH, TILE_WIDTH);
   dim3 grid(numBColumns / threads.x, numARows / threads.y);
   
   //Invoke kernel
	matrixMultiplyShared<<< grid, threads>>>(d_A, d_B, d_D);
   
   //Wait for all previously issued device commands before continuing
   err = cudaDeviceSynchronize();
   printf("Run sMM kernel: %s\n", cudaGetErrorString(err));
    	
	//Copy the GPU memory back to CPU
	err = cudaMemcpy(hostD, d_D.elements, mem_size_C, cudaMemcpyDeviceToHost);
   printf("Copy D off of device: %s\n", cudaGetErrorString(err));

   printMatricesCheck(hostC, hostD, numCRows, numCColumns);
   
   // Free GPU memory
   // Free devices 
	cudaFree(deviceA);
   cudaFree(deviceB);
   cudaFree(deviceC);
   
   //Free Matrices
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
    
   // Free CPU memory
 	free(hostA);
   free(hostB);
  	free(hostC);
   free(hostD);

   return 0;
}
