#ifndef _FUNCTION_H_
#define _FUNCTION_H_
#define _USE_MATH_DEFINES


#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <chrono>


using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace std::chrono;
// define detector


//////////////////////////////////////////////////////////////////////////////////
/*
*
* DECLARE SOME SUB FUNCTION TO SUPPORT DETECTION'S FUNCTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////


/**
 * Create Gaussian Kernel
 *
 * @param ksize - The size of kernel with matrix ksize x ksize
 * @param sigma - The sigma of gaussian distribution.
 * @return kernel - The kernel matrix destination.
 */
Mat createGaussianKernel(int ksize, float sigma);

/**
* Conovle source matrix with kernel
*
* @param src - The matrix of source image
* @param dest - The matrix of destination image
* @param kernel - The kernel matrix
*/
void convolve(const Mat& src, Mat& dest, const Mat& kernel);


/**
* Apply GaussianBlur kernel to image
*
* @param src - The matrix of source image (gray scale)
* @param dest - The matrix of destination image
* @param kernel - The gaussian kernel
*
*/
void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma);



/**
* Multiply matrix with value
* @param mat - The input matrix
* @param value - the value of scaling
* @return result - matrix that multiplied
*/
Mat multiply(const Mat& mat, float value);




/**
* Find max pixel of image
* @param mat - The matrix of image
* @return max - The max pixel in the image
*/
float findMaxPixel(const Mat& mat);




/**
* Compute gradient of filtered img (applied Gaussian kernel) 
* @param filteredImg - input matrix
* @param grad - gradient with 2 axis
* @param gradX - gradient with X axis
* @param gradY - gradient with Y axis
*/
void computeGradient(const Mat& filteredImg, Mat& grad, Mat& gradX, Mat& gradY);



/**
* Compute the length of hypotenuse.
* Two matrices must be the same shape
*
* @param mat1 - The first matrix
* @param mat2 - The second matrix
* @param dest - The destination matrix having same shape with two matrix
*/
void computeHypotenuse(const Mat& mat1, const Mat& mat2, Mat& dest);


/**
* Apply window size to image to compute R)
* @param gradX - Gradient with X axis
* @param gradY - Gradient with Y axis
* @param k - parameter of R 
* @param windowSize - window size to compute R
* @return Response matrix
*/
Mat computeResponseMatrix(const Mat& gradX, const Mat& gradY, float k, int windowSize);



/*
* Find all conrner of the images
* @param responseMatrix - the response matrix
* @param thresholdRatio - threshold to find corner points
* @return Vector of pair contains list of corner.
*/
vector<pair<int, int>> findCornerPoint(const Mat& responseMatrix, float thresholdRatio);


/*
* Draw corner points aftet detecting in the image 
* @param image - The mat of original image
* @param corners - The vector of map <int,int> corresponding to (x,y) of corner.
*/
void drawCornerPoints(const Mat& image, vector<pair<int, int>>& corners);



/*
* Calculate 2D - Laplacian of Gaussian.
*
* Reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
*
* @param x - the value at position x
* @param y - the value at position y
* @param sigma - Gaussian standard deviation
* @return The value of 2D Laplacian of Gaussian
*/
float calculateLaplacianOfGaussian(int x, int y, float sigma);


/*
* Create LoG (Laplacian of Gaussian) kernel
* @param ksize - the size of kernel
* @param sigma - Gaussian standard deviation
* @return -  The LogG kernel
*/
Mat createLaplacianOfGaussian(int ksize, float sigma);


/*
* Detect corner using Harris algorithms
* @param imageSource - The matric of image source (3 channel)
* @param corners - vector with pair (x,y) with pixel is corner in image
* @param k - the ccofficent of response matix
* @param windowSize - the window size when calculate response matrix
* @param thresholdRatio - threshold to consider corner
* @return destinationMatrix - with 255 is corner, otherwise 0.
*/
Mat detectHarris(const Mat& imageSource,vector<pair<int, int>>&corners,  float k = 0.04, int windowSize = 3, float thresholdRatio = 0.01);



/*
* Detect corner using Blob by LoG method
* @param imageSource - The matrix of image source
* @param corners - vector contains row, col, radius
* @param sigma - the sigma of laplacian of gaussian kernel
* @param scaleNumber - the number of scale 
* @param threshold - threshold to find maxima
* @return destination image that is mapped key point to image source
*/
Mat detectBlob(const Mat& imageSource, vector<pair<pair<int, int>, float>>& corners, float sigma =1.0, int scaleNumber= 8, float threshold = 0.2);


/*
* Detect corner using Blob by LoG method
* @param imageSource - The matrix of image source
* @param corners - vector contains row, col, radius
* @param sigma - the sigma of  gaussian kernel
* @param scaleNumber - the number of scale
* @param threshold - threshold to find maxima
* @return destination image that is mapped key point to image source
*/
Mat detectBlobDoG(const Mat& imageSource, vector<pair<pair<int, int>, float>>& corners, float sigma = 1.0, int scaleNumber = 8, float threshold = 0.2);


/*
* Convert keypoint self-implemented to keypoint openCV
* @param keypoints - vector of keypoint
* @return vector keypoint's types openCV
*/
vector<KeyPoint> convertToKeyPointVector(vector<pair<int, int>>& keypoints);

/*
* Convert keypoint self-implemented to keypoint openCV
* @param keypoints - vector of keypoint
* @return vector keypoint's types openCV
*/
vector<KeyPoint> convertToKeyPointVector(vector<pair<pair<int, int>, float>>& keypoints);


/*
* Match features by SIFT
* @param img1 - The query image
* @param img2 - The train image
* @param detector - detector used in function {1: Harris, 2: Blob, 3: Blob by DoG, 4: SIFT}
* @param k - that used in K -NN algorithms for BF matcher
* @param observe - options to show image. Default: false
* @return sumOfDistance of matching
*/
int matchBySIFT(const Mat& img1, const Mat& img2, int detector = 4, int k = 2, bool observe = false);


#endif // !_FUNCTION_H_

