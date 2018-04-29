#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/* compile >gcc -Wall -g -o Hello Hello.c -lpthread */

/*global variable accesible to all threads*/
long threads_count;
void* Hello(void* rank);

int main(int argc, char* argv[]) {
    long thread;
    pthread_t* thread_handles;

    //get number of threads
    threads_count=strtol(argv[1], NULL, 10);
    thread_handles=malloc(threads_count*sizeof(pthread_t));
    for(thread=0; thread<threads_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, Hello, (void*) thread);
}    
    printf("Hello from the main thread\n");
    for(thread=0; thread<threads_count; thread++){
        pthread_join(thread_handles[thread], NULL);
}
    free(thread_handles);

    return 0;
}

void* 
Hello(void* rank) {
    long my_rank = (long) rank;
    printf("hello from thread %ld of %ld\n", my_rank, threads_count);

    return NULL;
}
