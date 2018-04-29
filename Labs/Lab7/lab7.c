#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>

#define HISTOGRAM_SIZE HISTOGRAM_SIZE

cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat image;
uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

int *histoMem;
int *d_histoMem;
int *histoScan;
int *d_histoScan;

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

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
							uchar4 * const d_rgbaImage,
							unsigned char* const d_greyImage,
							size_t numRows,
							size_t numCols)
{
	//allocate CPU memory for histogram
   histoMem = (int *)malloc(HISTOGRAM_SIZE * sizeof(int));
   memset(histoMem, 0, size_t(HISTOGRAM_SIZE) * sizeof(int));
   //allocate GPU memory for histogram
   cudaMalloc(&d_histoMem, HISTOGRAM_SIZE * sizeof(int));
   //Populate GPU histogram
   cudaMemcpy(d_histoMem, histoMem, HISTOGRAM_SIZE*sizeof(int), cudaMemcpyHostToDevice);
   
	//Count of blocks and threads per block
   int threadCount=32;
   int gridSizeX=(numCols + (threadCount - 1))/threadCount;
   int gridSizeY=(numRows + (threadCount - 1))/threadCount ;
   const dim3 blockSize(threadCount, threadCount, 1);   
   const dim3 gridSize(gridSizeX, gridSizeY, 1);
   
   //greyscale kernel
   rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage,d_greyImage,numRows,numCols);
   cudaDeviceSynchronize();
  
   //histogram kernel
   histogram<<<gridSize, blockSize, HISTOGRAM_SIZE * sizeof(int)>>>(d_greyImage, d_histoMem, numRows, numCols);
   cudaDeviceSynchronize();
   
   //Copy device histogram back to host
   cudaMemcpy(histoMem, d_histoMem, sizeof(int) * HISTOGRAM_SIZE, cudaMemcpyDeviceToHost);
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

	int yPixel = blockIdx.y * blockDim.y + threadIdx.y;
	int xPixel = blockIdx.x * blockDim.x + threadIdx.x;
            
	if( yPixel < numRows && xPixel < numCols ){
      int i = yPixel * numCols + xPixel;
      uchar4 rgba = rgbaImage[i];
      float output = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[i] = output;
	}
}

__global__ void histogram(unsigned char* const greyImage, int *hist, int numRows, int numCols) {

   int yPixel = blockIdx.y * blockDim.y + threadIdx.y;
int xPixel = blockIdx.x * blockDim.x + threadIdx.x;
   int thread = threadIdx.x + threadIdx.y * blockDim.x;
   
   unsigned char Pvalue = 0;
   
   __shared__ unsigned int d_local[1024];
   
   d_local[thread] = 0;
   __syncthreads();
   //Have every thread add to their respective value
	if( yPixel < numRows && xPixel < numCols ){
      int i = yPixel*numCols+xPixel;
      //Thread's individual pixel value
      Pvalue = greyImage[i];
      __syncthreads();
      atomicAdd(&d_local[Pvalue], 1);
	}
   //Combine fragments into the global memory
   __syncthreads();
   atomicAdd(&hist[thread], d_local[thread]);
   __syncthreads();
}

__device__ unsigned char checkBound(int n) {
   return n > 255 ? 255 : (n < 0 ? 0:n);
}

void postProcess(const std::string& output_file) {

	//copy the output back to the host
	cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	int i;
   int result = 0;
   
   for (i = 0; i < HISTOGRAM_SIZE; i++)
      result += histoMem[i];
   
   printf("result %d\n", result);
   
   int tiles = 4; 
   int Width = HISTOGRAM_SIZE / tiles;


   int threadsPerBlock = HISTOGRAM_SIZE;
   int blocks = (HISTOGRAM_SIZE + (threadsPerBlock - 1))/threadsPerBlock;
   int cacheSize = HISTOGRAM_SIZE * sizeof(int);
   cudaMalloc(&d_histoScan, sizeof(int) * HISTOGRAM_SIZE);
   scan<<<blocks, threadsPerBlock, cacheSize>>>(d_histoMem, d_histoScan, HISTOGRAM_SIZE, tiles, Width);
   cudaDeviceSynchronize();
   cudaError_t err = cudaGetLastError();
   printf("Error: %s\n", cudaGetErrorString(err));
 
   histoScan = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE);
   memset(histoScan, 0, sizeof(int) * HISTOGRAM_SIZE);
   cudaMemcpy(histoScan, d_histoScan, sizeof(int) * HISTOGRAM_SIZE, cudaMemcpyDeviceToHost); 
   
   long equalized[HISTOGRAM_SIZE];
   for(int i = 1; i< HISTOGRAM_SIZE; i++){
      equalized[i] = (histoScan[i-1])*255/(numPixels);
   }

   cv::Mat blured.create(imageGrey.rows, imageGrey.cols, CV_8UC1);
   for(int y = 0; y < numRows(); y++)
	   for(int x = 0; x < numCols(); x++)
           	blured.at<int>(y,x) = 
		checkBound(equalized[imageGrey.at<uchar>(y,x)]);

   
   // Display Equalized image
   cv::namedWindow("Equalized");
   cv::imshow("Equalized",blured);
   cv::waitKey(0);
   cv::imwrite("Equalized.jpg", blured);
	
	////cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
	cudaFree(d_histoMem);
	cudaFree(d_histoScan);
}



__global__ void scan(int *hist, int *scan, int scanSize, 
                    int tiles, int Width) {
   //Indices to scan and store last element in accumulator
   int i, accIdx;
   //Threads corresponding to histogram location
   int thread = threadIdx.x + blockIdx.x * blockDim.x;
   __shared__ int d_local[scanSize];
   __shared__ int accumulator[4];
   //Copy from global to local
   d_local[thread] = hist[thread];
   __syncthreads();
   
   //Threads will sum their own ranges from thread to scanlength
   if (thread % Width == 0) {
      for(i = thread + 1; i < thread + Width; i++) {
         atomicAdd(&d_local[i], d_local[i -1]);
      }
      __syncthreads();
      //Store the last value into an accumulator cell
      accIdx = thread/Width;
      accumulator[accIdx] = d_local[i - 1];
   }
   __syncthreads();
   
   //Have one thread sum the accumulator
   if (thread == 0) {
      for (i = 1; i < tiles; i++) {
         atomicAdd(&accumulator[i], accumulator[i-1]);
      }   
   }
   __syncthreads();

   if (thread > Width - 1) {
      accIdx = (thread / Width) - 1;
      __syncthreads();
      atomicAdd(&d_local[thread], accumulator[accIdx]);
   }
   __syncthreads();
   
   scan[thread] = d_local[thread];
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



  cudaThreadExit();
  return 0;

}