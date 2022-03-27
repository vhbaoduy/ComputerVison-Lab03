#include "function.h"
int main(int argc, char** argv) {

	// Create command line parsers
	CommandLineParser parser(argc, argv,
		"{input ||Input image's path}"
		"{method |1|Choose method [1:Harris, 2:Blob, 3: Blob by DoG, 4: Match by SIFT] to detect key point}"
		"{k |0.04|The coefficent to compute response matrix in Harris detection}"
		"{windowSize |3|The window size to compute response matrix in Harriss detection}"
		"{sigma |1.0|The kernel size of gaussian kernel. Default: 1.0}"
		"{scaleNumber |8|The number of scaling sigma in using Blob. Default: 8}"
		"{threshold ||The threshold to find local maxima. Default: [Harris: 0.01, Blob:0.2]}"
		"{detector |4|Detector used in Match by SIFT method. detector = [1: Harris, 2: Blob, 3: Blob by DoG, 4: SIFT]. Default: 4}"
		"{knn |2| K-nearest neighbors used in method [4, 5]}"
		"{imageQuery || Image query's path used method [4, 5]}"
		"{imageTrain || Image train's path used method [4, 5]}"
		);

	// Show help's commandline 
	parser.about("\n~~This program detect key point of image~~\n[Press ESC to exit program]");
	parser.printMessage();

	// Get path to Image
	String pathToImage = parser.get<String>("input");

	// Get method
	int method = stoi(parser.get<String>("method"));
		
	try {

		Mat image = imread(pathToImage, IMREAD_COLOR);
		if (pathToImage != "") {
			resize(image, image, Size(512, 512));
			imshow("Origin", image);
		}
		Mat result;

		// detect by Harris
		if (method == 1) {
			int windowSize = stoi(parser.get<String>("windowSize"));
			string threshold = parser.get<String>("threshold");
			int k = stoi(parser.get<String>("k"));
			vector<pair<int, int>> cornersHarris;
			if (threshold != "") {
				result = detectHarris(image, cornersHarris, k, windowSize, stof(threshold));
			}
			else {
				result = detectHarris(image, cornersHarris, k, windowSize);
			}
		}
		// detect by Blob LoG
		if (method == 2) {
			int scaleNumber = stoi(parser.get<String>("scaleNumber"));
			float sigma = stof(parser.get<String>("sigma"));
			string threshold = parser.get<String>("threshold");
			vector<pair<pair<int, int>, float>> corners;
			if (threshold != "") {
				result = detectBlob(image, corners, sigma, scaleNumber, stof(threshold));
			}
			else {
				result = detectBlob(image, corners, sigma, scaleNumber);
			}
		}
		// detect by Blob DoG
		if (method == 3) {
			int scaleNumber = stoi(parser.get<String>("scaleNumber"));
			float sigma = stof(parser.get<String>("sigma"));
			string threshold = parser.get<String>("threshold");
			vector<pair<pair<int, int>, float>> corners;
			if (threshold != "") {
				result = detectBlobDoG(image, corners, sigma, scaleNumber, stof(threshold));
			}
			else {
				result = detectBlobDoG(image, corners, sigma, scaleNumber);
			}
		}
		// Matching by SIFT and KNN
		if (method == 4) {
			string path1 = parser.get<String>("imageQuery");
			string path2 = parser.get<String>("imageTrain");
			int k = stoi(parser.get<String>("knn"));
			int detector = stoi(parser.get<String>("detector"));


			// read image
			Mat img1 = imread(path1);
			Mat img2 = imread(path2);
			resize(img1, img1, Size(512, 512));
			resize(img2, img2, Size(512, 512));
			matchBySIFT(img1, img2, detector, k, true);
		}

		if (!result.empty()) {
			imshow("Result", result);
			waitKey(0);
		}
	}
	catch (Exception& ex) {
		
		cout << ex.msg << endl;
		return 0;
	}
	
	return 0;
}
