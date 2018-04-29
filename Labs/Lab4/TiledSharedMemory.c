#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#define Tile_size 2


#define funcCheck(stmt) do {
        cudaError_t err = stmt;
        if (err != cudaSuccess) {
            printf( "Failed to run stmt %d ", __LINE__);
            return -1;
        }
    } while(0)

int numARows;
int numAColumns;
int numBRows;
int numBColumns;
int numCRows;
int numCColumns;


__global__ void matrixMultiplyShared(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    __shared__ float sA[Tile_size][Tile_size];
    __shared__ float sB[Tile_size][Tile_size];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ Tile_size) + 1); k++) {
        if ( (Row < numARows) && (threadIdx.x + (k*Tile_size)) < numAColumns) {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
		} else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
		}
		
		if ( Col < numBColumns && (threadIdx.y + k*Tile_size) < numBRows) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*Tile_size)*numBColumns + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
		
		__syncthreads();

        for (int j = 0; j < Tile_size; ++j)
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
    
	if (Row < numCRows && Col < numCColumns) {
        C[Row*numCColumns + Col] = Cvalue;
    }
}

void Print_Mat(int Row,int Col,float * Mat) {
	for(int i=0;i<Row*Col;i++)
			{
			printf("%f  ",*(Mat+i));

			if((i%Col)==0 )
					printf("\n");
				
			}
}

void matMultiplyOnHost(float * A, float * B, float * C, 
	int numARows, int numAColumns, 
	int numBRows, int numBColumns,
	int numCRows, int numCColumns) {
    for (int i=0; i < numARows; i ++)
    {
        for (int j = 0; j < numAColumns; j++)
        {
            C[i*numCColumns + j ] = 0.0;
            for (int k = 0; k < numCColumns; k++)
                C[i*numCColumns + j ] += A[i*numAColumns + k] * B [k*numBColumns + j];
        }
    }
    return;
}

int main(int argc, char ** argv) {
    float * hostA;
    float * hostB;
    float * hostC;
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;


    printf("\nPlease Enter Rows and Columns of A:");
    scanf("%d %d",&numARows,&numAColumns);

    printf("\nPlease Enter Rows and Columns of B:");
    scanf("%d %d",&numBRows,&numBColumns);

    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    for (int i = 0; i < numARows*numAColumns; i++)
       	hostA[i]=1.0;
	
    for (int i = 0; i < numBRows*numBColumns; i++)
       	hostB[i]=1.0;

    printf("\nMatrix A Values:\n");
    Print_Mat(numARows,numAColumns,hostA);

    printf("\n\nMatrix B Values:\n");
    Print_Mat(numBRows,numBColumns,hostB);

    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    funcCheck(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
    funcCheck(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns));
    funcCheck(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));

    funcCheck(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice));

    dim3 dimGrid((numCColumns/Tile_size) + 1, (numCRows/Tile_size) + 1, 1);
    dim3 dimBlock(Tile_size, Tile_size, 1);

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err1 = cudaPeekAtLastError();

    cudaDeviceSynchronize();

    funcCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

    printf("\nMatrix C From Device\n");
    Print_Mat(numCRows,numCColumns,hostC);

    matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    printf("\nMatrix C From Host\n");
    Print_Mat(numCRows,numCColumns,hostComputedC);

    for (int i=0; i < numCColumns*numCRows; i++)
    {
        if (hostComputedC[i]  != hostC[i] )
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }

    printf("\n Number of Blocks Created:%d \n",((numCColumns/Tile_size) + 1)*((numCColumns/Tile_size) + 1));
    printf("\n Number of Threads Per Block: %d \n",(Tile_size*Tile_size));

    funcCheck(cudaFree(deviceA));
    funcCheck(cudaFree(deviceB));
    funcCheck(cudaFree(deviceC));

    free(hostA);
    free(hostB);
    free(hostC);
    free(hostComputedC);

    return 0;
}