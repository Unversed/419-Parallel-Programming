#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h> 


/* global variable accesible to all threads */
long threads_count;
int ROW, INNER, COL, NTHREADS;
int **A, **B, **C; 

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


void 
printMatrixInitialization()
{
   int i, j;
   time_t t;

   A = (int**)malloc(ROW * sizeof(int *));
   for(i = 0; i < ROW; i++)
      A[i] = malloc(INNER * sizeof(int));

   B = (int**)malloc(INNER * sizeof(int *));
   for(i = 0; i < INNER; i++)
      B[i] = malloc(COL * sizeof(int));

   C = (int**)malloc(ROW * sizeof(int *));
   for(i = 0; i < ROW; i++)
      C[i] = malloc(COL * sizeof(int));

   srand(time(&t));

   for(i = 0; i < ROW; i++)
      for(j = 0; j < INNER; j++)
         A[i][j] = rand()%100;

   for(i = 0; i < INNER; i++)
      for(j = 0; j < COL; j++)
         B[i][j] = rand()%100;

   printf("Matrices Generated\r\n");
   return;

}

/* Matrix Multiplication non threaded */ 
void 
Rect_MM_NT()
{
	for (int row = 0; row != ROW; ++row) 
	{
 		for (int col = 0; col != COL; ++col)
		{
			int sum = 0;
			for (int inner = 0; inner != INNER; ++inner)
			{
				sum += A[row][inner] * B[inner][col];
			}
			C[row][col] = sum;
		}
	}

   return;
}

/*Print non threaded matrix multiplication results to stdout */ 
void
printNonThreadedMM()
{
   double tstart, tstop, ttime;

   /* measure current system time */
   tstart = dtime();

   /* call non threaded matrix multiplication */
   Rect_MM_NT();

   /* measure new system time */
   tstop = dtime();

   /* measure system time difference */
   ttime = tstop - tstart;

   if ((ttime) > 0.0)
      printf("\nSecs Serial = %10.3lf\n",ttime);

   return;
}

/* Prints matrices clean up status to stdout */
void
printMatrixCleanUp()
{
   int i, j; 

   for(i = 0; i < ROW; i++)
      free((void *)A[i]);
   free((void *)A);

   for(i = 0; i < INNER; i++)
      free((void *)B[i]);
   free((void *)B);

   for(i = 0; i < ROW; i++)
      free((void *)C[i]);
   free((void *)C);

   printf("Matrices Free...\r\n");
   return;
}


int 
main(void) 
 {

   if(argc != 1)
   {
      printf("Usage: %s", argv[0]);
      exit(1);
   }

   SIZE = atoi(argv[1]);
   NTHREADS = atoi(argv[2]);
   int[] sizes = {512, 1024, 8192}
   int[] threads = {2, 4, 8}
   
   
   printf("Starting Computations...\r\n");
   printMatrixInitialization();

   printNonThreadedMM();



   printMatrixCleanUp();

   printf("Computations Complete... \r\n"); 
   return(0); 
	return 0;
 }