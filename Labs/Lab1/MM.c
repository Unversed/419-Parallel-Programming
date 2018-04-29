#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h> 


/*global variable accesible to all threads*/
long threads_count;
int SIZE, NTHREADS;
int **A, **B, **C, **D; 

/* dtime -
 * utility routine to return the current wall clock time
 */
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}


void init()
{
    int i, j;

    A = (int**)malloc(SIZE * sizeof(int *));
    for(i = 0; i < SIZE; i++)
        A[i] = malloc(SIZE * sizeof(int));

    B = (int**)malloc(SIZE * sizeof(int *));
    for(i = 0; i < SIZE; i++)
        B[i] = malloc(SIZE * sizeof(int));

    C = (int**)malloc(SIZE * sizeof(int *));
    for(i = 0; i < SIZE; i++)
        C[i] = malloc(SIZE * sizeof(int));
	
	D = (int**)malloc(SIZE * sizeof(int *));
    for(i = 0; i < SIZE; i++)
        D[i] = malloc(SIZE * sizeof(int));

    srand(time(NULL));

    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j++) {
            A[i][j] = rand()%100;
            B[i][j] = rand()%100;
        }
    }
}

/* Matrix Multiplication non threaded*/ 
void matrixmult(){
	int c, d, k, sum;
    for (c = 0; c < SIZE; c++) {
      for (d = 0; d < SIZE; d++) {
		sum = 0;
        for (k = 0; k < SIZE; k++) {
          sum = sum + A[c][k]*B[k][d];
        }
 
        C[c][d] = sum;
      }
    }
}


/*Matrix multiplication threaded */
void* matrixmultT(void* id){

    int i, j, k, sum;
	int tid = (int)id;
    int start = tid * SIZE/NTHREADS;
    int end = (tid+1) * (SIZE/NTHREADS) - 1;

    for(i = start; i <= end; i++) {
        for(j = 0; j < SIZE; j++) {
            D[i][j] = 0;
            for(k = 0; k < SIZE; k++) {
                D[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[] ) 
{
        int i,j;         
		double tstart, tstop, ttime;
   	    pthread_t* threads;   
     

    if(argc != 3)
    {
        printf("Usage: %s <size_of_square_matrix> <number_of_threads>\n", argv[0]);
        exit(1);
    }

    SIZE = atoi(argv[1]);
	init();
	
	printf("Starting Compute\r\n");

        	
	tstart = dtime();
    matrixmult();
	tstop = dtime();

    ttime = tstop - tstart;

        if ((ttime) > 0.0)
            printf("Secs Serial = %10.3lf\n",ttime);
       
		
		
	/*threaded part of HW*/

    NTHREADS = atoi(argv[2]);
	
	threads = (pthread_t*)malloc(NTHREADS * sizeof(pthread_t));
    

    tstart = dtime();	
   	
	for(i = 0; i < NTHREADS; i++)
        pthread_create(&threads[i], NULL, matrixmultT, (void *)i);
    for(i = 0; i < NTHREADS; i++)
        pthread_join(threads[i], NULL);
	
	free(threads);
   	
	tstop=dtime();
	ttime=tstop-tstart;
	if ((ttime) > 0.0)
        {
            printf("Secs Threaded = %10.3lf\n", ttime);
        }

	/*print results just for sanity check */
	int row, col;
	for (row=0; row<SIZE; row++){
    		for(col=0; col<SIZE; col++) {
         		printf("%lf ", C[row][col]);
		}
		printf("\n");
	}
	/*check the solutions are the same in both implementations*/

	float dif, accum=0;
	for (row=0; row<SIZE; row++){
    		for(col=0; col<SIZE; col++) {
			dif=abs(C[row][col]-D[row][col]);
         		if(dif!=0) accum+=dif;
		}
	}
	if(accum < 0.1) printf("SUCESS\n");
	else printf("FAIL\n");
	
	/* Clean Up */
    for(i = 0; i < SIZE; i++)
        free((void *)A[i]);
    free((void *)A);

    for(i = 0; i < SIZE; i++)
        free((void *)B[i]);
    free((void *)B);

    for(i = 0; i < SIZE; i++)
        free((void *)C[i]);
    free((void *)C);

        return( 0 );
}
