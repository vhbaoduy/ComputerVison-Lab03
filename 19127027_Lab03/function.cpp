#include "function.h"

//////////////////////////////////////////////////////////////////////////////////
/*
*
* IMPLEMENT SOME SUB FUNCTION TO SUPPORT DETECTION'S FUNCTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////
Mat createGaussianKernel(int ksize, float sigma) {


	// initial kernel matrix
	if (ksize % 2 == 0) {
		ksize +=1;
	}
	Mat dest(ksize, ksize, CV_32F);
	int range = (ksize - 1) / 2;

	// initial the needed variable
	double sum = 0.0, r;
	float s = 2.0 * sigma * sigma;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {
			r = x * x + y * y;

			// Apply gaussian distribution
			dest.at<float>(x + range, y + range) = (exp(-(r / s))) / (M_PI * s);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	// Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
		}
	}
	return dest.clone();
}

float calculateLaplacianOfGaussian(int x, int y, float sigma) {
	float value1 = -((x * x + y * y) / (2.0 * sigma * sigma));
	float value2 = -1.0 / (M_PI * sigma * sigma * sigma * sigma);
	return value2 * (1 + value1) * exp(value1);
}

Mat createLaplacianOfGaussian(int ksize, float sigma) {

	if (ksize % 2 == 0) {
		ksize += 1;
	}
	Mat dest(ksize, ksize, CV_32F);
	int range = (ksize - 1) / 2;
	// initial the needed variable
	float sum = 0.0, r;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {

			// Apply gaussian distribution
			dest.at<float>(x + range, y + range) = calculateLaplacianOfGaussian(x, y, sigma);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	//Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
			//if (dest.at<float>(i, j) < 0) temp++;
		}
	}
	return dest.clone();
}

void convolve(const Mat& src, Mat& dest, const Mat& kernel) {

	// initial destination matrix
	Mat result(src.rows, src.cols, CV_32F);

	int ksize = kernel.rows;

	// compute the center of matrix
	const int dx = ksize / 2;
	const int dy = ksize / 2;

	//loop height
	for (int i = 0; i < src.rows; ++i) {
		// loop width
		for (int j = 0; j < src.cols; ++j) {
			float temp = 0.0;
			for (int k = 0; k < ksize; ++k) {
				for (int l = 0; l < ksize; ++l) {
					int x = j - dx + l;
					int y = i - dy + k;

					// check position
					if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
						if (kernel.type() == CV_32F && src.type() == CV_8U) {
							// reduce noise
							temp += src.at<uchar>(y, x) * kernel.at<float>(k, l);
						}
						else {
							temp += src.at<float>(y, x) * kernel.at<float>(k, l);
						}
					}
				}
			}

			//mapping to [0, 1]
			result.at<float>(i, j) = temp;
		}
	}
	dest = result.clone();
}


void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma) {
	Mat kernel;

	// create gaussian kernel
	kernel = createGaussianKernel(ksize, sigma);
	convolve(src, dest, kernel);
}


Mat multiply(const Mat& mat, float value) {
	Mat result(mat.rows, mat.cols, CV_32F);
	bool flag = false;
	if (mat.type() == CV_8U) {
		flag = true;
	}
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			if (flag) {
				result.at<float>(i, j) = mat.at<uchar>(i, j) * value;
			}
			else {
				result.at<float>(i, j) = mat.at<float>(i, j) * value;
			}
		}
	}
	return result.clone();
}


float findMaxPixel(const Mat& mat) {
	float max = LLONG_MIN;
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			if (mat.at<float>(i, j) > max) {
				max = mat.at<float>(i, j);
			}
		}
	}
	return max;
}

void computeHypotenuse(const Mat& mat1, const Mat& mat2, Mat& dest) {
	// check the shape of two matrices
	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = sqrt(mat1.at<float>(i, j) * mat1.at<float>(i, j) + mat2.at<float>(i, j) * mat2.at<float>(i, j));
			}
		}
		dest = result.clone();
	}
}


void computeGradient(const Mat& filteredImg, Mat& grad, Mat& gradX, Mat& gradY) {
	float xFilters[3][3] = { {-1,0, 1}, {-2,0,2},{-1,0,1} };
	float yFilters[3][3] = { {-1, -2, -1} ,{0, 0, 0},{1, 2, 1} };
	//float xFilters[3][3] = { {1,1, 1}, {0,0,0},{-1,-1,-1} };
	//float yFilters[3][3] = { {1, 2, 1} ,{0, 0, 0},{-1, -2, -1}};

	Mat Kx(3, 3, CV_32F, xFilters);
	Mat Ky(3, 3, CV_32F, yFilters);

	convolve(filteredImg, gradX, Kx);
	//normalize(Ix, 1.0 / findMaxPixel(Ix));
	convolve(filteredImg, gradY, Ky);
	//normalize(Iy, 1.0 / findMaxPixel(Iy));
	computeHypotenuse(gradX, gradY, grad);
}



Mat computeResponseMatrix(const Mat & gradX, const Mat & gradY, float k, int windowSize) {
	Mat result(gradX.rows, gradX.cols, CV_32F);

	int haftWindowSize = windowSize / 2;
	float sumXX, sumYY, sumXY, det, trace;

	// compute response matrix
	for (int i = 0; i < gradX.rows; ++i) {
		for (int j = 0; j < gradY.cols; ++j) {
			sumXX = 0.0;
			sumYY = 0.0;
			sumXY = 0.0;

			// Compute M matrix
			for (int windowRow = -haftWindowSize; windowRow <= haftWindowSize; ++windowRow) {
				for (int windowCol = -haftWindowSize; windowCol <= haftWindowSize; ++windowCol) {
					if (i + windowRow < 0 || i + windowRow >= gradX.rows || j + windowCol < 0 || j + windowCol >= gradX.cols) {
						sumXX += 0.0;
						sumYY += 0.0;
						sumXY += 0.0;
					}
					else {
						sumXX += gradX.at<float>(i + windowRow, j + windowCol) * gradX.at<float>(i + windowRow, j + windowCol);
						sumYY += gradY.at<float>(i + windowRow, j + windowCol) * gradY.at<float>(i + windowRow, j + windowCol);
						sumXY += gradX.at<float>(i + windowRow, j + windowCol) * gradY.at<float>(i + windowRow, j + windowCol);
					}
				}
			}

			// calculate det(M)
			det = sumXX * sumYY - sumXY*sumXY;

			// calculate trace(M)
			trace = sumXX + sumYY;
			result.at<float>(i, j) = det - k * trace*trace;
		}
	}
	return result.clone();
}


vector<pair<int, int>> findCornerPoint(const Mat& responseMatrix, float thresholdRatio) {
	float threshold = thresholdRatio * findMaxPixel(responseMatrix);
	Mat result(responseMatrix.rows, responseMatrix.cols, CV_32F);
	vector < pair<int, int>> corners;
	for (int i = 1; i < responseMatrix.rows - 1; ++i) {
		for (int j = 1; j < responseMatrix.cols - 1; ++j) {
			float localMax = responseMatrix.at<float>(i, j);
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					if (responseMatrix.at<float>(i + x, j + y) > localMax) {
						localMax = responseMatrix.at<float>(i + x, j + y);
					}
				}
			}
			if (responseMatrix.at<float>(i, j) == localMax && localMax > threshold) {
				result.at<float>(i, j) = 255.0;
				corners.push_back(pair<int, int>(i, j));
			}
			else {
				result.at<float>(i, j) = 0.0;
			}
		}
	}
	return corners;
}

void drawCornerPoints(const Mat& image, vector<pair<int, int>>& corners) {
	for (int i = 0; i < corners.size(); ++i) {
		int y = corners[i].first;
		int x = corners[i].second;
		circle(image, Point(x, y), 3, Scalar(0, 255, 0), 2);
	}
}



//////////////////////////////////////////////////////////////////////////////////
/*
*
* IMPLEMENT MAIN FUNCTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////

Mat detectHarris(const Mat& imageSource, vector<pair<int, int>>& corners, float k, int windowSize, float thresholdRatio) {
	Mat imageGray, destinationImage;

	if (imageSource.channels() == 3) {
		cvtColor(imageSource, imageGray, COLOR_BGR2GRAY);
	}
	else {
		imageGray = imageSource.clone();
	}
	destinationImage = imageSource.clone();
	// init neccessary variables 
	Mat imageBlur, grad, gradX, gradY, responseMatrix;
	// Apply gaussian blur to reduce noise
	applyGaussianBlur(imageGray, imageBlur, 5, 1.0);

	// Compute gradient with Sobel filters.
	computeGradient(imageBlur, grad, gradX, gradY);

	// Compute hessian matrix (response matrix)
	responseMatrix = computeResponseMatrix(gradX, gradY, k, windowSize);

	// Find corners with threshold
	corners = findCornerPoint(responseMatrix, thresholdRatio);


	// Map corner to image
	drawCornerPoints(destinationImage, corners);

	// Show result
	//cout << "The number of corners: " << corners.size() << endl;
	//imshow("Result by Harris", destinationImage);

	return destinationImage.clone();

	// 

}



Mat detectBlob(const Mat& imageSource, vector<pair<pair<int, int>, float>>& corners, float sigma, int scaleNumber, float threshold) {
	Mat imageGray;
	Mat imageClone = imageSource.clone();
	if (imageSource.channels() == 3) {
		cvtColor(imageSource, imageGray, COLOR_BGR2GRAY);
	}
	else {
		imageGray = imageSource.clone();
	}

	// initial vector to store image and sigma
	float k = sqrt(2);
	vector<Mat> logImages;
	vector <float> sigmaArray;

	// scale image
	Mat imageScale = multiply(imageGray, 1.0 / 255.0);

	//cout << findMaxPixel(imageScale) << endl;
	//applyGaussianBlur(imageScale, imageScale, 5, 1.4);

	float maxPixel = FLT_MIN;
	// convovle image with sigma
	for (int i = 0; i < scaleNumber; i++) {
		Mat tempMat;

		// scaling sigma and add to vector
		float kPow = pow(k, i);
		float sigmaCurrent = sigma * kPow;
		sigmaArray.push_back(sigmaCurrent);

		// create kernel and apply
		Mat logKernel = multiply(createLaplacianOfGaussian(int(sigmaCurrent*6), sigmaCurrent), sigmaCurrent*sigmaCurrent);
		convolve(imageScale, tempMat, logKernel);

		// normalize LoG image
		tempMat = multiply(tempMat, 1.0 / findMaxPixel(tempMat));
		//imshow("sigma " + to_string(sigmaCurrent), tempMat);
		logImages.push_back(tempMat);

		logKernel.release();
	}	
	//float thresholdPixel = maxPixel * threshold;
	int counterMaxima = 0;
	set<pair<int, int>> cornersTemp;

	// loop scale number: sigma
	for (int n = 0; n < scaleNumber; n++) {
		float radius = int(sqrt(2) * sigmaArray[n]);

		// loop rows of image
		for (int row = 0; row < imageGray.rows; row++) {
			// loop cols of image
			for (int col = 0; col < imageGray.cols; col++) {
				bool flagMax = true;
				bool flagMin = true;

				// loop to consider maxima
				for (int l = -1; l <= 1; l++) {
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							if (row + i >= 0 && col + j >= 0 && row + i < imageGray.rows &&
								col + j < imageGray.cols && n + l >= 0 && n + l < scaleNumber) {
								if (logImages[n].at<float>(row, col) >= threshold) {

									// flag of local max
									if (logImages[n].at<float>(row, col) < logImages[n].at<float>(row + i, col + j)) {
										flagMax = false;
									}
									// flag of local max
									if (logImages[n].at<float>(row, col) < logImages[n + l].at<float>(row + i, col + j)) {
										flagMax = false;
									}
									// flag of local min
									if (logImages[n].at<float>(row, col) > logImages[n].at<float>(row + i, col + j)) {
										flagMin = false;
									}
									// flag of local min
									if (logImages[n].at<float>(row, col) > logImages[n + l].at<float>(row + i, col + j)) {
										flagMin = false;
									}
								}
								else {
									flagMax = false;
									flagMin = false;
								}
							}
						}
					}
				}

				// Check maxima and draw circle
				if (flagMax || flagMin) {
					if (row - radius > 0 && col - radius > 0 && row + radius < imageGray.rows - 1 && col + radius < imageGray.cols - 1) {
						
						counterMaxima++;
						corners.push_back(pair<pair<int, int>, float>(pair<int, int>(row, col), radius));
						circle(imageClone, Point(col, row), radius, Scalar(0, 0, 255));
					}
				}
			}
		}
		
	}
	//cout << "The number of maxia by Blob LoG: " << counterMaxima << endl;
	//imshow("Result of Blob LoG", imageClone);
	return imageClone;
	// 
}


Mat detectBlobDoG(const Mat& imageSource, vector<pair<pair<int, int>, float>>& corners, float sigma, int scaleNumber, float threshold) {
	Mat imageGray;
	Mat imageClone = imageSource.clone();
	if (imageSource.channels() == 3) {
		cvtColor(imageSource, imageGray, COLOR_BGR2GRAY);
	}
	else {
		imageGray = imageSource.clone();
	}

	// scale image
	Mat imageScale = multiply(imageGray, 1.0 / 255.0);


	// create kernel and apply kernel to compute substract 2 matrix
	Mat kernel;
	kernel = createGaussianKernel(int(sigma*6), sigma);
	Mat prevImage, curImage;
	convolve(imageScale, prevImage, kernel);
	//prevImage = multiply(prevImage, 1.0 / findMaxPixel(prevImage));

	kernel.release();

	// init vector 
	float k = sqrt(2);
	vector<Mat> logImages;
	vector<float> sigmaArray;
	for (int i = 1; i < scaleNumber; i++) {
		// scaling sigma
		float sigmaCurrent = pow(k, i) * sigma;


		// create kernel and apply
		kernel = createGaussianKernel(int(sigmaCurrent * 6), sigmaCurrent);
		sigmaArray.push_back(sigmaCurrent);
		convolve(imageScale, curImage, kernel);

		// norm by divide to max
		//curImage = multiply(curImage, 1.0 / findMaxPixel(curImage));


		// subtract 2 matrix
		Mat tempImage = curImage - prevImage;
		//imshow("sigma " + to_string(sigmaCurrent), tempImage);
		tempImage = multiply(tempImage, 1.0 / findMaxPixel(tempImage));
		logImages.push_back(tempImage);

		prevImage.release();
		prevImage = curImage;
		kernel.release();
	}

	int counterMaxima = 0;

	// loop scale number: sigma
	for (int n = 0; n < scaleNumber - 1; n++) {
		float radius = int(sqrt(2) * sigmaArray[n]);
		// loop rows of image
		for (int row = 0; row < imageGray.rows; row++) {
			// loop cols of image
			for (int col = 0; col < imageGray.cols; col++) {
				bool flagMax = true;
				bool flagMin = true;
				// loop to consider maxima
				for (int l = -1; l <= 1; l++) {
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							if (row + i >= 0 && col + j >= 0 && row + i < imageGray.rows &&
								col + j < imageGray.cols && n + l >= 0 && n + l < scaleNumber -1) {
								if (abs(logImages[n].at<float>(row, col)) >= threshold) {
									// flag of local max
									if (logImages[n].at<float>(row, col) < logImages[n].at<float>(row + i, col + j)) {
										flagMax = false;
									}
									// flag of local max
									if (logImages[n].at<float>(row, col) < logImages[n + l].at<float>(row + i, col + j)) {
										flagMax = false;
									}
									// flag of local min
									if (logImages[n].at<float>(row, col) > logImages[n].at<float>(row + i, col + j)) {
										flagMin = false;
									}
									// flag of local min
									if (logImages[n].at<float>(row, col) > logImages[n + l].at<float>(row + i, col + j)) {
										flagMin = false;
									}
								}
								else {
									flagMax = false;
									flagMin = false;
								}
							}
						}
					}
				}
				

				// Check maxima and draw circle
				if (flagMax || flagMin) {
					if (row - radius > 0 && col - radius > 0 && row + radius < imageGray.rows - 1 && col + radius < imageGray.cols - 1) {
						counterMaxima++;
						corners.push_back(pair<pair<int, int>, float>(pair<int, int>(row, col), radius));
						circle(imageClone, Point(col, row), radius, Scalar(0, 0, 255));
					}
				}
			}
		}

	}

	//cout << "The number of maxia by Blob DoG: " << counterMaxima << endl;
	//imshow("Result Blob DoG", imageClone);
	return imageClone;

}

vector<KeyPoint> convertToKeyPointVector(vector<pair<int, int>>& keypoints) {
	vector<KeyPoint> result;
	for (int i = 0; i < keypoints.size(); i++) {
		KeyPoint point(keypoints[i].second,keypoints[i].first,1);
		result.push_back(point);
	}
	return result;
}

vector<KeyPoint> convertToKeyPointVector(vector<pair<pair<int, int>, float>>& keypoints) {
	vector<KeyPoint> result;
	for (int i = 0; i < keypoints.size(); i++) {
		KeyPoint point(keypoints[i].first.second, keypoints[i].first.first, keypoints[i].second);
		result.push_back(point);
	}
	return result;
}

int matchBySIFT(const Mat& img1, const Mat& img2, int detector, int k, bool observe) {
	if (detector < 1 || detector > 4) {
		return -1.0;
	}
	Mat imgGray1, imgGray2;
	if (img1.channels() == 3) {
		cvtColor(img1, imgGray1, COLOR_BGR2GRAY);
	}
	else {
		imgGray1 = img1.clone();
	}
	if (img2.channels() == 3) {
		cvtColor(img2, imgGray2, COLOR_BGR2GRAY);
	}
	else {
		imgGray2 = img2.clone();
	}
	// init keypoint
	vector<KeyPoint> keypoints1, keypoints2;
	if (detector == 1) {
		vector<pair<int, int>> corners1, corners2;
		detectHarris(imgGray1, corners1);
		detectHarris(imgGray2, corners2);
		keypoints1 = convertToKeyPointVector(corners1);
		keypoints2 = convertToKeyPointVector(corners2);
	}

	if (detector == 2) {
		vector<pair<pair<int, int>,float>> corners1, corners2;
		
		detectBlob(imgGray1, corners1,1.0,5,0.4);
		detectBlob (imgGray2, corners2,1.0, 5, 0.4);
		keypoints1 = convertToKeyPointVector(corners1);
		keypoints2 = convertToKeyPointVector(corners2);
	}
	if (detector == 3) {
		vector<pair<pair<int, int>, float>> corners1, corners2;
		detectBlob(imgGray1, corners1, 1.0, 5, 0.4);
		detectBlob(imgGray2, corners2, 1.0, 5, 0.4);
		keypoints1 = convertToKeyPointVector(corners1);
		keypoints2 = convertToKeyPointVector(corners2);
	}
	if (detector == 4) {

		Ptr<SIFT> sift = SIFT::create();

		sift->detect(imgGray1, keypoints1);
		sift->detect(imgGray2, keypoints2);
	}
	Mat imgKeypoint1, imgKeypoint2;


	Mat desciptor1, desciptor2;
	Ptr<SIFT> extractor = SIFT::create();
	

	// extract feature form image, keypoint to desciptor
	extractor->compute(imgGray1, keypoints1, desciptor1);
	extractor->compute(imgGray2, keypoints2, desciptor2);
	BFMatcher bf;
	vector<vector<DMatch>> matches;

	// using knn with 
	bf.knnMatch(desciptor1, desciptor2, matches, k);

	const float ratio_thresh = 0.75f;
	//cout << matches.size() << " " << matches[0].size() << endl;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	double sumDistances = 0;
	if (good_matches.size() == 0) {
		sumDistances = -1.0;
	}
	else {
		for (int i = 0; i < good_matches.size(); i++) {
			sumDistances += good_matches[i].distance;
		}
	}
	if (observe) {
		Mat result;
		drawMatches(img1, keypoints1, img2, keypoints2, good_matches, result);
		imshow("Result", result);
		waitKey(0);
	}
	return sumDistances;

}
