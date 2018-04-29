//
// MM.c
//
// A simple example to try 
// to get lots of Flops on 
// Intel(r) Xeon Phi(tm) co-processors.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

//
// dtime -
//
// utility routine to return 
// the current wall clock time
//
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}

#define HIGHT 1024
#define WIDTH 1024
#define FLOPS_ARRAY_SIZE (HIGHT*WIDTH) 

// declare input arrays  
//TODO

// declare the output array
//TODO

/*global variable accesible to all threads*/
long threads_count;

//
// Matrix Multiplication non threaded
// 
void matrixmult(int Hight, int Width){
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

//
//Matrix multiplication threaded
//
void* matrixmultT(void* id){

	//TODO
        return NULL;
}

int main(int argc, char *argv[] ) 
{
        int i,j;         double tstart, tstop, ttime;
   	        
        //
        // initialize the input arrays 
        //
        printf("Initializing\r\n");
        for(i=0; i<FLOPS_ARRAY_SIZE; i++)
        {
	    //TODO
        }	

        printf("Starting Compute\r\n");

        tstart = dtime();	
        // loop many times to really get lots of calculations
        for(j=0; j<1000; j++)  
        {
              	// multiply the two arrays 
               	matrixmult(HIGHT,WIDTH);
	}
        tstop = dtime();

        // elasped time
        ttime = tstop - tstart;

        //
        // Print the results
        //
        if ((ttime) > 0.0)
        {
            printf("Secs Serial = %10.3lf\n",ttime);
        }

	//threaded part of HW
	long thread;
 	pthread_t* thread_handles;
	double tstartT, tstopT, ttimeT;

  	//get number of threads for user input and allocate memory to them
  	//TODO

        tstartT = dtime();	
  	for(thread=0; thread<threads_count; thread++)
     		pthread_create(&thread_handles[thread], NULL, matrixmultT, (void*) thread);
  	for(thread=0; thread<threads_count; thread++)
     		pthread_join(thread_handles[thread], NULL);
   	free(thread_handles);
   	tstopT=dtime();
	ttimeT=tstopT-tstartT;
	if ((ttimeT) > 0.0)
        {
            printf("Secs Threaded = %10.3lf\n", ttimeT);
        }

	//print results just for sanity check
	//int row, col;
	/*for (row=0; row<HIGHT; row++){
    		for(col=0; col<WIDTH; col++) {
         		printf("%lf ", fcthread[row*WIDTH+col]);
		}
		printf("\n");
	}*/
	//check the solutions are the same in both implementations
	int row, col;
	float dif, accum=0;
	for (row=0; row<HIGHT; row++){
    		for(col=0; col<WIDTH; col++) {
			dif=abs(fc[row*WIDTH+col]-fcthread[row*WIDTH+col]);
         		if(dif!=0) accum+=dif;
		}
	}
	if(accum < 0.1) printf("SUCESS\n");
	else printf("FAIL\n");


        return( 0 );
}

