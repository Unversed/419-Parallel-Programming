#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>


#define BLUR_SIZE 3
#define USE_2D 0

//define the storage for the blur kernel in GPU Constant Memory
__constant__ float M_d1[BLUR_SIZE];

cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat image;
uchar4 *d_rgbaImage__;
uchar4 *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();


//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, uchar4 **greyImage,
				uchar4 **d_rgbaImage, uchar4 **d_greyImage,
				const std::string &filename) {
	//make sure the context initializes ok
	cudaFree(0);
	//Read Image into an OpenCV Matrix
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageRGBA.copyTo(imageGrey);
	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = (uchar4 *)imageGrey.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();

	//TODO allocate memory on the device for both input and output
	

	//TODO copy input array to the GPU
	

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	//TODO copy the output back to the host
	
	cudaDeviceSynchronize();
	//change in color space required by OpenCV	
	cv::cvtColor(imageGrey, imageGrey, CV_BGR2RGBA);
	//output the image to a file
	cv::imwrite(output_file.c_str(), imageGrey);
	//display the output image (will only work if you are on the lab machines)
	cv::imshow ("Output Image", imageGrey);
	cv::waitKey(0);
	////cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);

}
__global__
void conv1D(const uchar4* const rgbaImage,uchar4* const greyImage,int numRows, int numCols)
{
	//TODO Fill in the kernel to blur original image
	// Original Image is an array, each element of the array has 4 components .z -> R (red); .y -> G (Green) ; .x -> B (blue); .w -> A (alpha, you can ignore this one)
	//so you can read one input pixel like this:
        //B = rgbaImage[currow * numCols + curcol].x*M_d[curcolkernel]; 
	//G = rgbaImage[currow * numCols + curcol].y*M_d[curcolkernel];
	//R = rgbaImage[currow * numCols + curcol].z*M_d[curcolkernel];
	
}


void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * d_rgbaImage,
							uchar4*  d_greyImage,
							size_t numRows,
							size_t numCols)
{
	float M_h[BLUR_SIZE]={0.279,0.441,0.279};  //change this to whatever 1D filter you are using
	cudaMemcpyToSymbol(M_d,M_h, BLUR_SIZE*BLUR_SIZE*sizeof(float)); //allocates/copy to Constant Memory on the GPU
	//temp image
	uchar4 *d_greyImageTemp;
	cudaMalloc((void **)&d_greyImageTemp, sizeof(uchar4) * numRows*numCols);
	cudaMemset(d_greyImageTemp, 0, numRows*numCols * sizeof(uchar4)); //make sure no memory is left laying around
	
	int threadSize=0; //TODO change to the right value
	int gridSizeX=0; //TODO change to right value
	int gridSizeY=0; //TODO change to right value
	const dim3 blockSize(threadSize, threadSize, 1);  //TODO
	const dim3 gridSize(gridSizeY, gridSizeX, 1);  //TODO
	for (int i=0;i<1;i++){
		//row
		conv1D<<<gridSize, blockSize>>>(d_rgbaImage,d_greyImageTemp,numRows,numCols);
		cudaDeviceSynchronize();
		//col
		//TODO call your kernel now for the columns
		//swap
		d_rgbaImage=d_greyImage;
	}

}

int main(int argc, char **argv) {
	cudaDeviceReset();

	uchar4 *h_rgbaImage, *d_rgbaImage;
	uchar4 *h_greyImage, *d_greyImage;
	std::string input_file;
	std::string output_file;

	if (argc == 3) {
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
	}
	else {
		std::cerr << "Usage: ./hw input_file output_file" << std::endl;
		exit(1);
	}

	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
	//call the students' code
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    cudaDeviceSynchronize();
	cudaGetLastError();
	printf("\n");
	postProcess(output_file); //prints gray image

     cudaThreadExit();
     return 0;

}

