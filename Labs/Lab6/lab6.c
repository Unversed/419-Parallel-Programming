/*
 ============================================================================
 Name        : MMShared.cu
 Author      : Liam Lefferts
 Version     : 1.0
 ============================================================================
 */

// System includes
#include <stdio.h>
#include<stdlib.h>
#include <time.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>

#define TILE_WIDTH 16

typedef struct {
	int numRows;
	int numCols;
	int stride;
	float * elements;
} Matrix;

/* dtime -
 * utility routine to return the current wall clock time
 */
double 
dtime()
{
   double tseconds = 0.0;
   struct timeval mytime;
   gettimeofday(&mytime,(struct timezone*)0);
   tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
   return( tseconds );
}

// Fill matrix elements
void initialize(float *data, unsigned size) {
	for(unsigned i = 0; i < size; ++i)
		data[i] = ((i % 10) * .1 + 1);
}

//
// Matrix Multiplication CPU for error checking
//
void matrixmult(float *fa, float *fb, float *fc, int Hight, int Width) {
	int row, col, k;
	float Pvalue = 0;
	for(row = 0; row < Hight; row++) {
		for(col = 0; col < Width; col++) {
			Pvalue = 0;
			for(k = 0; k < Width; k++) {
				Pvalue += fa[row * Width + k] * fb[k * Width + col];
			}
			fc[row * Width + col] = Pvalue;
		}
	}
}

//Compute C=A*B in GPU non shared memory
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows,
		int numAColumns, int numBRows, int numBColumns, int numCRows,
		int numCColumns) {
	//identify row and column to work on
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row * numCColumns + col;
	float Pvalue = 0;
	int k;
	if (row < numARows && col < numBColumns) {
		for (k = 0; k < numBColumns; k++) {
			Pvalue += A[row * numAColumns + k] * B[k * numBColumns + col];
		}
		C[i] = Pvalue;
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
__global__ void matrixMultiplyShared(Matrix d_A, Matrix d_B, Matrix d_C) {
// Each thread block computes one matrix tile, tile_C, of d_C
	Matrix tile_C;
	tile_C.numCols = tile_C.numRows = TILE_WIDTH;
	tile_C.stride = d_C.stride;
	tile_C.elements = &d_C.elements[d_C.stride * TILE_WIDTH * blockIdx.y
			+ TILE_WIDTH * blockIdx.x];

// Each thread computes one element of tile_C by accumulating results into Pvalue
	float Pvalue = 0.0;

	// Iterate over matrix tiles of d_A and d_B to compute tile_C
	// Multiply each pair of matrix tiles together
	// accumulate tile_C results
	for (int m = 0; m < (d_A.numCols / TILE_WIDTH); ++m) {

//populate tile_A
		Matrix tile_A;
		tile_A.numCols = tile_A.numRows = TILE_WIDTH;
		tile_A.stride = d_A.stride;
		tile_A.elements = &d_A.elements[d_A.stride * TILE_WIDTH * blockIdx.y
				+ TILE_WIDTH * m];

//populate tile_B
		Matrix tile_B;
		tile_B.numCols = tile_B.numRows = TILE_WIDTH;
		tile_B.stride = d_B.stride;
		tile_B.elements = &d_B.elements[d_B.stride * TILE_WIDTH * m
				+ TILE_WIDTH * blockIdx.x];

		// Shared memory tile_A
		__shared__
		float shared_A[TILE_WIDTH][TILE_WIDTH];
		// Shared memory tile_A
		__shared__
		float shared_B[TILE_WIDTH][TILE_WIDTH];

// Each thread loads one tile_A and tile_B element from device to shared memory 
		shared_A[threadIdx.y][threadIdx.x] = tile_A.elements[threadIdx.y
				* tile_A.stride + threadIdx.x];
		shared_B[threadIdx.y][threadIdx.x] = tile_B.elements[threadIdx.y
				* tile_B.stride + threadIdx.x];

// Synchronize to ensure tile initialization
		__syncthreads();

//Accumulation logic
		for (int tile = 0; tile < TILE_WIDTH; ++tile)
			Pvalue += shared_A[threadIdx.y][tile] * shared_B[tile][threadIdx.x];

// Ensures tile computations
		__syncthreads();
	}

// Each thread writes one sub_C element to global device memory
	tile_C.elements[threadIdx.y * tile_C.stride + threadIdx.x] = Pvalue;
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
void printMatricesCheck(float * A, float * B, int numCRows, int numCColumns) {
   float accum = 0;
	for (int i = 0; i < numCRows * numCColumns; ++i) {
		if ((accum = abs(A[i] - B[i])) >= .1) {
			printf("FAILED at line %d: A=%.4f B=%.4f %f", i, A[i], B[i]);
			break;
		}
	}
   
   if (accum <= .1) printf("SUCCESSFUL");
         
	/* print MM result
	for (int i = 0; i < TILE_WIDTH * TILE_WIDTH; ++i) {
		if (i % TILE_WIDTH == 0)
			printf("\n");

		printf("%.2f ", B[i]);

	}*/
	printf("\n");
}

int main(int argc, char ** argv) {

	int numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns;

	if (argc != 2) {
		printf("Usage: %s <square matrix dimension>\n", argv[0]);
		exit(1);
	}
	numARows = numBRows = numCRows = numAColumns = numBColumns = numCColumns =
			atoi(argv[1]);

	/*
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
	 */

	printf("A[%d x %d] * B[%d x %d] = C[%d x %d]\n", numARows, numAColumns,
			numBRows, numBColumns, numCRows, numCColumns);

	//Dimensions requirement check MM
	if (numAColumns != numBRows) {
		printf("numAColumns != numBRows, This matrix cannot be multiplied\n");
		exit(1);
	}

	//allocate host memory
	float *hostA = new
	float[numARows*numAColumns]; //input matrix A
	float *hostB = new
	float[numBRows*numBColumns]; //input matrix B
	float *hostC = new
	float[numCRows*numCColumns]; //output matrix C
	float *hostD = new
	float[numCRows*numCColumns]; //output matrix D
	float *hostE = new
	float[numCRows*numCColumns]; //output matrix E

	initialize(hostA, numARows * numAColumns);
	initialize(hostB, numBRows * numBColumns);

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
	dim3
	threads_nSM(TILE_WIDTH, TILE_WIDTH);
	dim3
	grid_nSM(numBColumns / threads_nSM.x, numARows / threads_nSM.y);

   //Initialize CUDA measurement 
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   //Start measure of first CUDA operation time   
   cudaEventRecord(start);

	//MM without shared memory
	matrixMultiply<<< grid_nSM, threads_nSM>>>(deviceA, deviceB, deviceC,numARows, numAColumns,numBRows, numBColumns, numCRows, numCColumns);

   //Stop measure of first CUDA operation time
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);

   printf("Secs MatMul Non Shared = %10.3lf\n", milliseconds);
      
	//Wait for all previously issued device commands before continuing 
	err = cudaDeviceSynchronize();
	printf("Run nsMM kernel: %s\n", cudaGetErrorString(err));

	//Copy the GPU memory back to the CPU here
	err = cudaMemcpy(hostE, deviceC, mem_size_C, cudaMemcpyDeviceToHost);
	printf("Copy E off of device: %s\n", cudaGetErrorString(err));

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
	dim3
	threads_sM(TILE_WIDTH, TILE_WIDTH);
	dim3
	grid_sM(numBColumns / threads_sM.x, numARows / threads_sM.y);

   //Start measure of first CUDA operation time   
   cudaEventRecord(start);

	//MM with shared memory
	matrixMultiplyShared<<< grid_sM, threads_sM>>>(d_A, d_B, d_D);
   
   //Stop measure of first CUDA operation time
   cudaEventRecord(stop);
   
   cudaEventSynchronize(stop);
   milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);

   printf("Secs MatMul Shared = %10.3lf\n", milliseconds);
   
	//Wait for all previously issued device commands before continuing
	err = cudaDeviceSynchronize();
	printf("Run sMM kernel: %s\n", cudaGetErrorString(err));

	//Copy the GPU memory back to CPU
	err = cudaMemcpy(hostD, d_D.elements, mem_size_C, cudaMemcpyDeviceToHost);
	printf("Copy D off of device: %s\n", cudaGetErrorString(err));

	printMatricesCheck(hostE, hostC, numCRows, numCColumns);
	printMatricesCheck(hostC, hostD, numCRows, numCColumns);
	printMatricesCheck(hostD, hostE, numCRows, numCColumns);

	// Free GPU memory
	// Free device matrices 
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	//Free device matrix structures 
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_D.elements);

	// Free host result matrices
	free(hostA);
	free(hostB);
	free(hostC);
	free(hostD);

	return 0;
}
