all : lab8.c
	cp lab8.c lab8.cu
	/usr/local/cuda-9.0/bin/nvcc -I//home/ljcates/opencv_build/include,//home/ljcates/opencv_build/include/opencv,//home/ljcates/opencv_build/include/opencv2 -L//home/ljcates/opencv_build/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -g -o lab8 lab8.cu 
	rm lab8.cu