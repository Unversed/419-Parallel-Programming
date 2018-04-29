all : ImagetoGray.cu
	/usr/local/cuda-9.0/bin/nvcc \
-I//home/ljcates/opencv_build/include \
-I//home/ljcates/opencv_build/include/opencv \
-L//home/ljcates/opencv_build/lib \
-lopencv_core,opencv_imgproc,opencv_highgui,opencv_imgcodecs \
-g -o ImagetoGray ImagetoGray.cu