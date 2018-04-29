# Makefile to build OpenMP_Matrix_Mult program
# --- macros
CC= gcc
CFLAGS= -O3 -g -fopenmp
OBJECTS= OpenMP_Matrix_Mult.o


# --- targets
all:	OpenMP_Matrix_Mult
OpenMP_Matrix_Mult:  $(OBJECTS)
	$(CC)  -o OpenMP_Matrix_Mult  $(OBJECTS)
        
OpenMP_Matrix_Mult.o:   OpenMP_Matrix_Mult.c
	$(CC) $(CFLAGS) -c OpenMP_Matrix_Mult.c
       


# --- remove binary and executable files
clean:
	rm -f OpenMP_Matrix_Mult $(OBJECTS)