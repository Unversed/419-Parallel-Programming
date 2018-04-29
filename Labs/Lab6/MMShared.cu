/*
 ============================================================================
 Name        : MMShared.cu
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#define TILE_WIDTH 16

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
// Compute C = A * B
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


// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ TODO Insert code to implement matrix multiplication here
    //@@ TODO You have to use shared memory for this MP

}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}


int main(int argc, char ** argv) {

	int numARows=512; // number of rows in the matrix A
    	int numAColumns=512; // number of columns in the matrix A
    	int numBRows=512; // number of rows in the matrix B
    	int numBColumns=512; // number of columns in the matrix B
    	int numCRows=numARows; // number of rows in the matrix C (you have to set this)
    	int numCColumns=numBColumns; // number of columns in the matrix C (you have to set this)

    	//check if you can do the MM
	if(numAColumns != numBRows){
		printf("This matrix cannot be multiplied");
		return -1;
	}
    	//alloc memory
	float *hostA = new float[numARows*numAColumns];
	initialize (hostA, numARows*numAColumns);
	float *hostB = new float[numBRows*numBColumns];
	initialize (hostB, numBRows*numBColumns);
	float * hostC=new float[numCRows*numCColumns];; // The output C matrix
	//do MM on CPU for timing
	matrixmult(hostA, hostB, hostC, numARows, numAColumns);

	
	//device variables
	float * deviceA;
	float * deviceB;
	float * deviceC;

    	//@@ Allocate GPU memory here
	unsigned int size_A = numARows * numAColumns;
    	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int size_B = numBRows * numBColumns;
    	unsigned int mem_size_B = sizeof(float) * size_B;
    	unsigned int size_C = numCRows * numCColumns;
    	unsigned int mem_size_C = sizeof(float) * size_C;

    	cudaMalloc((void**) &deviceA, mem_size_A);
    	cudaMalloc((void**) &deviceB, mem_size_B);
    	cudaMalloc((void**) &deviceC, mem_size_C);


    	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, mem_size_A, cudaMemcpyHostToDevice) ;
    	cudaMemcpy(deviceB, hostB, mem_size_B, cudaMemcpyHostToDevice) ;


    	//@@ Initialize the grid and block dimensions here
    	dim3 threads(0,0,0); //TODO change to correct values
    	dim3 grid(0,0,0); //TODO change to correct values
	//TODO comment the next two lines and uncoment the matrixMultiplyShared lines to execute
	//MM with shared memory
    	matrixMultiply<<< grid, threads>>>(deviceA, deviceB, deviceC,numARows, numAColumns,numBRows, numBColumns, numCRows, numCColumns);
    	cudaDeviceSynchronize();
    	//@@ Copy the GPU memory back to the CPU here
    	cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost) ;
    	/*print MM result
	for (row=0; row<numCRows; row++){
		for(col=0; col<numCColumns; col++) {
			printf("%lf ",hostC[row*numCColumns+col]);
		}
		printf("\n");
	}*/

    	//@@ TODO Launch the GPU Kernel here now for shared Memory
	//Uncomment this next two lines to execute the shared matrix matrix multipliaction code
	//matrixMultiplyShared<<< grid, threads>>>(deviceA, deviceB, deviceC,numARows, numAColumns,numBRows, numBColumns, numCRows, numCColumns);
    	//cudaThreadSynchronize();
    	
	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost) ;

	//verify result  all elements added
	//accum of error and if less than xxx
	//print MM result
	for (row=0; row<numCRows; row++){
		for(col=0; col<numCColumns; col++) {
			printf("%lf ",hostC[row*numCColumns+col]);
		}
		printf("\n");
	}

        //@@ Free the GPU memory here
	cudaFree(deviceA);
    	cudaFree(deviceB);
    	cudaFree(deviceC);


 	free(hostA);
    	free(hostB);
    	free(hostC);

    	return 0;
}
