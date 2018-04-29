

/*
 ============================================================================
 Name        : OpenCVCu.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>

#define BLOCK_SIZE 5


cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat image;
uchar4        *d_rgbaImage__;

unsigned char *d_greyImage__;
size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();


//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
				uchar4 **d_rgbaImage, unsigned char **d_greyImage,
				const std::string &filename) {
	//make sure the context initializes ok
	cudaFree(0);
	//cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}
	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage  = imageGrey.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
	cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around
	//copy input array to the GPU
	cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels,cudaMemcpyHostToDevice);
	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	const int numPixels = numRows() * numCols();
	//copy the output back to the host
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//output the image
	cv::imwrite(output_file.c_str(), imageGrey);
	cv::imshow ("Output Image", imageGrey);
	cv::waitKey(0);
	////cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);

}

void referenceCalculation(const uchar4* const rgbaImage,
						  unsigned char *const greyImage,
						  size_t numRows,
						  size_t numCols)
{
	for (size_t r = 0; r < numRows; ++r) {
		for (size_t c = 0; c < numCols; ++c) {
			uchar4 rgba = rgbaImage[r * numCols + c];
			float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
			greyImage[r * numCols + c] = channelSum;
		}
	}
}



__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
					   unsigned char* const greyImage,
					   int numRows, int numCols)
{
	//Fill in the kernel to convert from color to greyscale
	//the mapping from components of a uchar4 to RGBA is:
	// .x -> R ; .y -> G ; .z -> B ; .w -> A
	//
	//The output (greyImage) at each pixel should be the result of
	//applying the formula: output = .299f * R + .587f * G + .114f * B;
	//Note: We will be ignoring the alpha channel for this conversion
	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset.

	//TODO
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * const d_rgbaImage,
							unsigned char* const d_greyImage,
							size_t numRows,
							size_t numCols)
{
	int threadSize=0; //TODO change it to a number that make sense max 1024
	int gridSizeY=0 ;//TODO Change to number of blocks on Y -rows- direction
	int gridSizeX=0; //TODO change to number of vlocks on X -column- direction 
	const dim3 blockSize(threadSize, threadSize, 1);   
	const dim3 gridSize(gridSizeX, gridSizeY, 1);  
	//TODO call your GPU kernel here
	cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
	cudaDeviceReset();

	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
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
