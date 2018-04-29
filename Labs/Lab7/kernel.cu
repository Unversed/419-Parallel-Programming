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
uchar4 *d_rgbaImage__;

unsigned char *d_greyImage__;
size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();

template<typename _Tp> static inline _Tp saturate_cast(int v) { return _Tp(v); }

int *hist_cpu;
int *hist_gpu;
int *cumhistogram;
int *cdf_gpu;

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
	//TODO allocate memory on the device for both input and output d_rgbaImage and d_greayImage
   cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
   cudaMalloc(d_greyImage, sizeof(char) * numPixels);
   cudaMalloc(d_bw, sizeof(char) * numPixels);
	//TODO use cudamemset to set d_greayImage to 0
   cudaMemset(greyImage, 0, numPixels * sizeof(unsigned char));

	//copy input array to the GPU inputInage to d_rgbaImage
	cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels,cudaMemcpyHostToDevice);
	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	const int numPixels = numRows() * numCols();
	//TODO copy the output from GPU back to the host d_greyImage__ to imageGrey
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
   //output the image
	cv::imwrite(output_file.c_str(), imageGrey);
   
	////cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
	//cudaFree(hist_gpu);
}

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
					   unsigned char* const greyImage,
					   int numRows, int numCols) {
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

	int row= blockIdx.y * blockDim.y + threadIdx.y; // row Number
	int col = blockIdx.x * blockDim.x + threadIdx.x; // col Number
            
	if( row < numRows && col < numCols ){
      int i = row*numCols+col;
      uchar4 rgba = rgbaImage[ i ];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[ i ] = channelSum;
	}
}



__global__
void threshold(unsigned char* const greyImage, unsigned char* const bw,
				   int numRows, int numCols, int threshVal) {

	int row= blockIdx.y * blockDim.y + threadIdx.y; // row Number
	int col = blockIdx.x * blockDim.x + threadIdx.x; // col Number
   
	if( row < numRows && col < numCols ){
      int i = row*numCols+col;
      unsigned char pixel = greyImage[i];
      if (pixel < threshVal) {
         bw[i] = 0;
      }
      else {
         bw[i] = 255;
      }
	}
}

__global__ void histogram(unsigned char* const greyImage, int *hist, int numRows, int numCols) {
	//TODO write the create histogram on GPU function
   // Pixel Location
   int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
   int t = threadIdx.x + threadIdx.y * blockDim.x;
   unsigned char val = 0;
   __shared__ unsigned int localHist[1024];
   
   localHist[t] = 0;
   __syncthreads();
   //Have every thread add to their respective value
	if( row < numRows && col < numCols ){
      int i = row*numCols+col;
      //Thread's invidual pixel value
      val = greyImage[i];
      __syncthreads();
      atomicAdd(&localHist[val], 1);
	}
   //Combine fragments into the global memory
   __syncthreads();
   atomicAdd(&hist[t], localHist[t]);
   __syncthreads();
}

__global__ void cdf(int *hist, int *cdf, int numElements, 
                    int divisions, int scanLength) {
	//TODO write the create histogram on GPU function
   //Indicies to scan and store last element in acculumator
   int i, accIdx;
   //Threads corresponding to hitogram location
   int t = threadIdx.x + blockIdx.x * blockDim.x;
   __shared__ int localHist[256];
   __shared__ int accumulator[4];
   //Copy from global to local
   localHist[t] = hist[t];
   __syncthreads();
   
   //Threads will sum their own ranges from t to scanlength
   if (t % scanLength == 0) {
      for(i = t + 1; i < t + scanLength; i++) {
         atomicAdd(&localHist[i], localHist[i -1]);
      }
      __syncthreads();
      //Store the last value into an accumulator cell
      accIdx = t/scanLength;
      accumulator[accIdx] = localHist[i - 1];
   }
   __syncthreads();
   
   //Have one thread sum the accumulator
   if (t == 0) {
      for (i = 1; i < divisions; i++) {
         atomicAdd(&accumulator[i], accumulator[i-1]);
      }   
   }
   __syncthreads();

   if (t > scanLength - 1) {
      accIdx = (t / scanLength) - 1;
      __syncthreads();
      atomicAdd(&localHist[t], accumulator[accIdx]);
   }
   __syncthreads();
   
   cdf[t] = localHist[t];
}


void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * const d_rgbaImage,
							unsigned char* const d_greyImage,
							size_t numRows,
							size_t numCols)
{
	//TODO allocate mem for hist in cpu and for gpu
   unsigned int hist_size = 256;
   const size_t hs = size_t(hist_size) * sizeof(int);
   hist_cpu = (int *)malloc(hist_size * sizeof(int));
   memset(hist_cpu, 0, hs);
   cudaMalloc(&hist_gpu, hist_size * sizeof(int));
   cudaMemcpy(hist_gpu, hist_cpu, hist_size*sizeof(int), cudaMemcpyHostToDevice);
   
	//TODO decide the number of threads per block and the number of blocks
   int threadSize=32; //TODO change it to a number that make sense max 1024
   int gridSizeY=(numRows + (threadSize - 1))/threadSize ;//TODO Change to number of blocks on Y -rows- direction
   int gridSizeX=(numCols + (threadSize - 1))/threadSize; //TODO change to number of vlocks on X -column- direction 
   const dim3 blockSize(threadSize, threadSize, 1);   
   const dim3 gridSize(gridSizeX, gridSizeY, 1);
   //call kernel to trsnafrom to gray
   rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);
   cudaDeviceSynchronize();
  
   //TODO call your kernel to get the histogram
   histogram<<<gridSize, blockSize, hist_size * sizeof(int)>>>(d_greyImage, hist_gpu, numRows, numCols);
   cudaDeviceSynchronize();
   //TODO copy hist back to cpu
   cudaMemcpy(hist_cpu, hist_gpu, sizeof(int) * 256, cudaMemcpyDeviceToHost);
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
	postProcess(output_file); 

   int i = 0;
   int result = 0;
   for (i = 0; i < 256; i++) {
      printf("hist[%d] = %d\n", i, hist_cpu[i]);
      result += hist_cpu[i];
   }
   printf("result %d\n", result);
   
   //TODO write the cummulative histogram
   int numElements = 256;
   //Number of scans to make
   int divisions = 4; 
   int scanLength = numElements / divisions;
   //Kernel initialization
   int threadsPerBlock = 256;
   int blocks = (numElements + (threadsPerBlock - 1))/threadsPerBlock;
   int cacheSize = numElements * sizeof(int);
   
   cudaMalloc(&cdf_gpu, sizeof(int) * numElements);
   printf("threads %d blocks %d\n", threadsPerBlock, blocks);
   cdf<<<blocks, threadsPerBlock, cacheSize>>>(hist_gpu, cdf_gpu, numElements, divisions, scanLength);
   cudaDeviceSynchronize();
   cudaError_t err = cudaGetLastError();
   printf("Error: %s\n", cudaGetErrorString(err));
 
   cumhistogram = (int *)malloc(sizeof(int) * 256);
   memset(cumhistogram, 0, sizeof(int) * 256);
   cudaMemcpy(cumhistogram, cdf_gpu, sizeof(int) * 256, cudaMemcpyDeviceToHost); 
   
   for (i = 0; i < 256; i++) {
      printf("cum_cpu[%d] = %d\n", i, cumhistogram[i]);
   }
   
   //TODO equalize Apply eq 1 to get the final result
   long equalized[256];
   for(int i = 1; i<256; i++){
      equalized[i] = (cumhistogram[i-1])*255/(numRows()*numCols());
      printf("equ %d %ld\n",i, equalized[i]);
   }

   //writes images equalized 
   cv::Mat new_image= imageGrey.clone();
   for(int y = 0; y < numRows(); y++)
	   for(int x = 0; x < numCols(); x++)
           	new_image.at<uchar>(y,x) = saturate_cast<uchar>(equalized[imageGrey.at<uchar>(y,x)]);

   // Display the original Image
   cv::namedWindow("Original Image");
   cv::imshow("Original Image", image);
   
   // Display equilized image
   cv::namedWindow("Equilized Image");
   cv::imshow("Equilized Image",new_image);
   cv::waitKey(0);
   cv::imwrite("equilized.jpg", new_image);

  cudaThreadExit();
  return 0;

}