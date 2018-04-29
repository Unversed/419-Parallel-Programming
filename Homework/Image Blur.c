#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>

#define BLUR_SIZE 9

//1D Blur filter
float M_h[BLUR_SIZE]={0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0};
//define the storage for the blur kernel in GPU Constant Memory
__constant__ float M_d[BLUR_SIZE];

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
	//at least based upon my limited understanding of OpenCV, but better tocheck
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
	//make sure no memory is left laying around
	cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); 
	//copy input array to the GPU
	cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels,cudaMemcpyHostToDevice);
	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}


void your_rgba_to_blur(const uchar4 * const h_rgbaImage,
							uchar4 * d_rgbaImage,
							uchar4*  d_greyImage,
							size_t numRows,
							size_t numCols)
{

	//Copy Mask to GPU Constant Memory
	cudaMemcpyToSymbol(M_d,M_h, BLUR_SIZE*sizeof(float));
	
	//Kernel initialization, blocks and threads per block
	int threadCount=16;
	int gridSizeX=(numCols + threadCount - 1)/threadCount; 
	int gridSizeY=(numRows + threadCount - 1)/threadCount;
	const dim3 blockSize(threadCount, threadCount, 1);
	const dim3 gridSize(gridSizeX, gridSizeY, 1);
	
	//Call Blur kernel
	for (int i=0;i<30;i++){
		blur<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
		cudaDeviceSynchronize();
	}
}

__global__
void blur(uchar4* const rgbaImage, uchar4* const greyImage, int numRows, int numCols)
{
	//2D thread position
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	//Bounds Check
	if (x >= 0 && x < numCols && y >= 0 && y < numRows) { 
	
	//calculate a 1D offset.
	int oneD = y * numCols + x;
	
	//Initialize blurred RGB values
	float blurValx = 0, blurValy = 0, blurValz = 0;
	
	//Perform blur logic by applying
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			
			//Linearize imageIndex
			int yClamped = min(max(y + i, 0), numRows-1);
			int xClamped = min(max(x + j, 0), numCols-1);
			int imageIndex = yClamped * numCols + xClamped;

            //Linearize maskIndex
			int iClamped = min(max(1 + i, 0), 3-1);
			int jClamped = min(max(1 + j, 0), 3-1);
			int maskIndex = iClamped * 3 + jClamped;
			
			//Compute blur values
            int weight = M_d[maskIndex];
			blurValx += rgbaImage[imageIndex].x * weight;
			blurValy += rgbaImage[imageIndex].y * weight;
			blurValz += rgbaImage[imageIndex].z * weight;
		}
	}
	
	//Store blurred pixel
	greyImage[oneD].x = min(max(blurValx, 0), 255);
	greyImage[oneD].y = min(max(blurValy, 0), 255);
	greyImage[oneD].z = min(max(blurValx, 0), 255);
	}
}

void postProcess(const std::string& output_file) {
	
	//copy the output back to the host
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	//output the image
	cv::imwrite(output_file.c_str(), imageGrey);
	cv::imshow ("Output Image", imageGrey);
	cv::waitKey(0);
	
	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
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
	
	//call kernels
	your_rgba_to_blur(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    cudaDeviceSynchronize();
	cudaGetLastError();

	//output image
	postProcess(output_file); 

     cudaThreadExit();
     return 0;

}