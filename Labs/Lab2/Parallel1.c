#include <stdio.h>
#include <omp.h>
#include <vector>

using namespace std;

int segment_read(char *buff, const int len, const int count) {
  return 1;  
}

void foo(char* buffer, size_t size) {
    int count_of_reads = 0;
    int count = 1;
    std::vector<int> *posa;
    int nthreads;

    #pragma omp parallel 
    {
        nthreads = omp_get_num_threads();
        const int ithread = omp_get_thread_num();
        #pragma omp single 
        {
            posa = new vector<int>[nthreads];
            posa[0].push_back(0);
        }

        //get the number of lines and end of line position
        #pragma omp for reduction(+: count)
        for(int i=0; i<size; i++) {
            if(buffer[i] == '\n') { //should add EOF as well to be safe
                count++;
                posa[ithread].push_back(i);
            }
        }

        #pragma omp for     
        for(int i=1; i<count ;i++) {    
            const int len = posa[ithread][i] - posa[ithread][i-1];
            char* buff = &buffer[posa[ithread][i-1]];
            const int sequence_counter = segment_read(buff,len,i);
            if (sequence_counter == 1) {
                #pragma omp atomic
                count_of_reads++;
                printf("\n Total No. of reads: %d \n",count_of_reads);
            }

        }
    }
    delete[] posa;
}

int main () {
  FILE * pFile;
  long lSize;
  char * buffer;
  size_t result;

  pFile = fopen ( "myfile.txt" , "rb" );
  if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

  // obtain file size:
  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  // allocate memory to contain the whole file:
  buffer = (char*) malloc (sizeof(char)*lSize);
  if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}

  // copy the file into the buffer:
  result = fread (buffer,1,lSize,pFile);
  if (result != lSize) {fputs ("Reading error",stderr); exit (3);}

  /* the whole file is now loaded in the memory buffer. */
  foo(buffer, result);
  // terminate


  fclose (pFile);
  free (buffer);
  return 0;
}