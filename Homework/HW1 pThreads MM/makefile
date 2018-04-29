# Makefile to build OpenMP_Matrix_Mult program
# --- macros
CC= gcc
CFLAGS= -g -pthread
OBJECTS= MM.o


# --- targets
all:	MM
MM:  $(OBJECTS)
	$(CC)  -o MM $(OBJECTS)
        
MM.o:   MM.c
	$(CC) $(CFLAGS) -c MM.c
       


# --- remove binary and executable files
clean:
	rm -f MM $(OBJECTS)