#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <vector>
#include <queue>

#define _USE_MATH_DEFINES
#include <math.h>

#define TEST_LV0

#define TIMESCALE 					1.0f

#define FOR(i,a,b)		 			for (int i=(int)(a); i<(int)(b); ++i)
#define REP(i,a) 					FOR(i,0,a)
#define ZERO(m)     				memset(m,0,sizeof(m))
#define COPY(d,s)					memcpy(d,s,sizeof(s))
#define DEBUG(str)       	 		__android_log_write(ANDROID_LOG_DEBUG, "SignRecognition::Native", str)


using namespace std;
using namespace cv;


extern "C" {

static double timer0 = 0.0;
static double timer1 = 0.0;
static double timer2 = 0.0;


/* Time slot */
static const float MAX_FRAME_TIME			= 0.25f * TIMESCALE;
static const float MAX_SIGN_TIME 			= 0.025f * TIMESCALE;

/* # of iterations before checking the time slot
 * (w.r.t the implementation of processFrame it must be a power of 2) */
static const int FILTER_SYNC				= 8;
/* Sign border hue */
static const float FILTER_MEAN 				= -3.9601f;
static const float FILTER_STD 				= 11.8458f;
static const float FILTER_DENORMALIZER		= 0.0336777f;

/* # of clusters for each frame */
static const int CLUSTER_Y_NUM 				= 16;
static const int CLUSTER_X_NUM 				= 16;
/* Abs difference allowed between a cluster's value and its neighbors */
static const int CLUSTER_MAX_DELTA 			= 128;

/* Sign recognition params */
static const int SIGN_MIN_SIDE 				= 24;
static const int SIGN_MAX_SIDE 				= 75;
static const float SIGN_MAX_RATIO 			= 1.2f;

static const float CONFIDENCE_THRESHOLD		= 0.75f;

/* Simulated annealing min/max temperature */
static const float ANNEALING_MINT			= 0.01f;
static const float ANNEALING_MAXT 			= 10.0f;

/* Index 4 or 8 connected pixels: the first 4 indexes are 4-connected to the center and the second 4 are 8-connected to it
 * .....
 * .848.
 * .404.
 * .848.
 * .....
 */
static const int CONNECT_X[] 				= {1, -1, 0, 0, 1, 1, -1, -1};
static const int CONNECT_Y[] 				= {0, 0, 1, -1, -1, 1, 1, -1};

/* Each template is composed by a 32x32 pixel binary sign (1 uint32 for each row, 1 bit for each pixel) */
static const int TEMPLATE_NUM 				= 13;
static const uint TEMPLATES[] 				= {
/* Do not pass */								98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2080374846, 2013265950, 2046574494, 2030313630, 1929797838, 1946058702, 1938715086, 1946058702, 2064015582, 2064015582, 2013265950, 2080374846, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Do not pass > 3.5t */						98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2097020990, 2046754846, 2046754846, 2046754846, 1912537102, 1912537102, 1912541454, 1912545422, 2046771102, 2046770078, 2021801886, 2111254974, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* No stopping */								98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520094200, 1065354108, 1035994940, 2095058494, 2020613150, 2016950302, 2015129630, 1880023054, 1879556110, 1879556110, 1879556110, 2014240798, 2015129630, 2016950302, 2087722046, 1021316668, 1069549436, 528483320, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* No parking */								98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 528482552, 1069547644, 1021313084, 2087714878, 2016935966, 2015100958, 2014183454, 1879506958, 1879277582, 1879162894, 1879105550, 2013294622, 2013280286, 2013273118, 2080378430, 1006634812, 1040188412, 520094200, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 30 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2088516670, 2020507166, 2017879582, 2013423134, 1879467534, 1880909326, 1881040398, 1879467534, 2013423134, 2017879582, 2020507166, 2088516670, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 40 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1007681596, 2083535934, 2016443934, 2016437790, 2020632094, 1886414350, 1892705806, 1888511502, 1905288718, 2046322206, 2046322206, 2014346782, 2081438782, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 50 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2088516670, 2029551134, 2025874974, 2025874974, 1895327246, 1895589390, 1888249358, 1879860750, 2014078494, 2026661406, 2029813278, 2088254526, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 60 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1009778748, 2096643134, 2027453982, 2022204958, 2038457886, 1907910158, 1912104462, 1909220878, 1905026574, 2039244318, 2026661406, 2029551134, 2088254526, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 70 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2088516670, 2021424670, 2014078494, 2014078494, 1880909326, 1880647182, 1880647182, 1882744334, 2016437790, 2016437790, 2016443934, 2087730238, 1006633020, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Speed limit 90 */							98304, 4193280, 16776960, 33554304, 66863040, 132122592, 260047344, 520093944, 1040187516, 1006633020, 2096643134, 2029551134, 2044487198, 2039244318, 1905026574, 1910269454, 1895589390, 1886152206, 2014078494, 2022467102, 2027453982, 2096643134, 1008730172, 1040187516, 520093944, 260047344, 132122592, 66863040, 33554304, 16776960, 4193280, 98304,
/* Zebra crossing */							98304, 245760, 245760, 245760, 516096, 1044480, 1044480, 946176, 1849344, 3947520, 3947520, 3677184, 7343616, 7441920, 15830784, 31508352, 29475712, 29483904, 63083456, 125866464, 117555424, 117522656, 234954864, 506901112, 505852536, 476355384, 946641692, 952932764, 2013265950, 4294967295u, 4294967295u, 4294967295u,
/* School crossing */							98304, 245760, 245760, 245760, 516096, 1044480, 1044480, 946176, 1849344, 3947520, 3947520, 3677184, 7343616, 7343616, 15732480, 31459200, 29361024, 29361024, 63177664, 126095840, 117577952, 117938400, 235093104, 505091192, 505076856, 470423608, 939659292, 939659292, 2013265950, 4294967295u, 4294967295u, 4294967295u,
/* Speed hump */								98304, 245760, 245760, 245760, 516096, 1044480, 1044480, 946176, 1849344, 3947520, 3947520, 3677184, 7343616, 7343616, 15732480, 31459200, 29361024, 29361024, 62915520, 125829600, 117440736, 117440736, 234913904, 503570552, 504363128, 473955896, 943717916, 939524124, 2013265950, 4294967295u, 4294967295u, 4294967295u
											};

/* Camera orientation Hidden Markov Model likelihood */
static const int HMM_CAM_PRECISION			= 2;
static const float HMM_DISORIENT_SPEED		= 60.0f;
static const float HMM_ANGLE_DEV			= 5.0f;
static const int HMM_STATES_NUM				= 11;
static float HMM_CAM_STATES[HMM_STATES_NUM];
static float HMM_DENORM;


/* Parameters for the EM clustering */
static const int EM_SAMPLES_NUM				= 4;
/* Minimum interval in seconds before updating params */
static const float EM_UPDATE_DELTA			= 5.0f;
static int EM_START_INDEX					= 0;
static Mat EM_MEAN;
static Mat EM_COVARIANCE_INV;
/* Samples should be treated as a circular list of the last EM_SAMPLES_NUM sign's coordinates */
static Mat EM_SAMPLES;


/* Frame time slot beginning */
static double START_TIME;
/* Last sign detection time */
static double SIGN_DETECTED_TIME;

/* Current frame in the HSV color space */
static Mat FRAME_HSV;
/* Current frame's blue channel in the BGR color space */
static Mat FRAME_B;

/* Look Up tables for HSV -> probability mask conversion */
static Mat LUT_H;
static Mat LUT_S;


#ifdef TEST_LV0

static Mat TEST_IMAGE;
static vector<int> debugSignID;
static vector<float> debugSignValue;
static bool isNewFrame = true;
static float testScore = 0.0;

static void print(double frameRate, double lastTime) {

	int baseline = 50;

	int HMMbaserow = TEST_IMAGE.size().width - 150;
	int HMMspacing = 10;
	int HMMheight = 40;
	int HMMyaxisX = HMMbaserow + (HMM_STATES_NUM - 1) * HMMspacing / 2;

	double fontScale = 0.3;

	int signBaserow = 10;
	int signHSpacing = 130;
	int signVSpacing = 20;

	int frameRateBaserow = TEST_IMAGE.size().width - 260;

	Scalar blue = Scalar(255, 0, 0, 255);

	REP (i, debugSignID.size()) {

		/* sign ID */
		stringstream id; id << "ID: " << debugSignID[i];
		putText(TEST_IMAGE, id.str(), Point(signBaserow + i * signHSpacing, baseline - signVSpacing), 0, fontScale, blue);

		/* sign value */
		stringstream vals; vals << "F1: " << debugSignValue[i];
		putText(TEST_IMAGE, vals.str(), Point(signBaserow + i * signHSpacing, baseline), 0, fontScale, blue);

	}

	/* Sign detected time*/
	stringstream ld; ld << "LD: " << lastTime;
	putText(TEST_IMAGE, ld.str(), Point(frameRateBaserow, baseline - signVSpacing), 0, fontScale, blue);

	/* Frame rate*/
	stringstream fr; fr << "FR: " << frameRate;
	putText(TEST_IMAGE, fr.str(), Point(frameRateBaserow, baseline), 0, fontScale, blue);

	/* HMM */
	vector<vector<Point> > polys(1);
	REP(i, HMM_STATES_NUM) polys[0].push_back(Point(HMMbaserow + i * HMMspacing, baseline - HMM_CAM_STATES[i] * HMMheight));

	const Point* pts = &polys[0][0];
	int npts[2] = { HMM_STATES_NUM , 2 };

	polylines(TEST_IMAGE, &pts, npts, 1, false, blue);
	line(TEST_IMAGE, Point(HMMbaserow, baseline), Point(HMMbaserow + (HMM_STATES_NUM - 1) * HMMspacing, baseline), blue);
	line(TEST_IMAGE, Point(HMMyaxisX, baseline - HMMheight), Point(HMMyaxisX, baseline), blue);

}

#endif


/* priority_queue support class */
class clusterComparator {
public:
	bool operator()(const pair<float, Point_<int> >& a, const pair<float, Point_<int> >& b) {
		return a.first < b.first;
	}
};


/* Return a cpu (wall-clock) time for x86 (ARM) processors when LOCAL is (not) defined */
static double getTime() {
	timeval time; gettimeofday(&time, NULL);
	return time.tv_sec + (time.tv_usec * 1e-6);
}


/* Return a pseudo-random real number in the range [0..1] */
static float randReal() {
	return (float)rand()/((float)RAND_MAX + 1);
}


/* Return the Hamming weight of an uint32 (can be replaced by an intrinsic on some x86 processors) */
static uint popcount(uint x) {

	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	return ((x + (x >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;

}


/* Return the 32x32 isOutside array flood filled where mask < 128 */
static void fillOuter(const uint startY, const uint startX, bool* isOutside, const Mat& mask) {

	isOutside[startY * 32 + startX] = true;

	REP(i, 4) {

		int y = startY + CONNECT_Y[i];
		int x = startX + CONNECT_X[i];

		if (y >= 0 && y < 32 && x >= 0 && x < 32 && !isOutside[y * 32 + x] && mask.at<uchar>(y, x) < 128)
			fillOuter(y, x, isOutside, mask);

	}
}


/* Return src resized to 32x32 elements and rotated by angle */
static void resizeAndRotate(const Mat& src, Mat& dst, const double angle) {

	Mat foo; resize(src, foo, Size(32,32), 0, 0, INTER_CUBIC);
	Mat rot = getRotationMatrix2D(Point_<float>(16, 16), angle, 1.0);
	warpAffine(foo, dst, rot, Size(32,32), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));

}


/* Return a segmented sign in the current frame localized by roi and rotated by angle */
static void segmentSign(const Mat& border, const Rect_<int>& roi, uint* segment, const double angle) {

	/* Resize and rotate the sign's border */
	Mat maskBorder; resizeAndRotate(border(roi + Point(1, 1)) * 255, maskBorder, angle);

	/* Build a mask with the pixels outside of the sign set to true */
	bool isOutside[1024]; ZERO(isOutside);
	if(maskBorder.at<uchar>(0, 0) < 128 && !isOutside[0]) fillOuter(0, 0, isOutside, maskBorder);
	if(maskBorder.at<uchar>(0, 31) < 128 && !isOutside[31]) fillOuter(0, 31, isOutside, maskBorder);
	if(maskBorder.at<uchar>(31, 0) < 128 && !isOutside[992]) fillOuter(31, 0, isOutside, maskBorder);
	if(maskBorder.at<uchar>(31, 31) < 128 && !isOutside[1023]) fillOuter(31, 31, isOutside, maskBorder);

	/* Resize and rotate the sign's blue channel */
	Mat maskInside;	resizeAndRotate(FRAME_B(roi), maskInside, angle);

	/* Noise reduction possibly preserving the edges */
	Mat foo; bilateralFilter(maskInside, foo, 0, 15, 2, BORDER_REPLICATE); maskInside = foo;

	/* Separate the background (high blue values, i.e. white or blue pixels)
	 * from the foreground (low blue values, i.e. black or red pixels) */
	adaptiveThreshold(maskInside, maskInside, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 33, 0);

	/* Merge together the sign's border and content */
	REP(i, 32) REP(j, 32)
		segment[i] = (segment[i] << 1) | (!isOutside[i * 32 + j] & ((maskBorder.at<uchar>(i, j) | maskInside.at<uchar>(i, j)) >> 7));

}


/* Return the score for the best sign and its index in TEMPLATES */
static pair<float, int> evaluate(const uint* segment) {

	float bestValue = 0.0;
	int index = 0;

	REP(t, TEMPLATE_NUM) {

		uint sumNeg = 0;
		uint sumPos = 0;

		REP(i, 32) {
			/* # of pixels that don't match */
			sumNeg += popcount(TEMPLATES[t * 32 + i] ^ segment[i]);
			/* # of pixels that match */
			sumPos += popcount(TEMPLATES[t * 32 + i] & segment[i]);
		}

		/* F1-score */
		float value = sumPos * 2.0f / (2 * sumPos + sumNeg);

		if (bestValue < value) {
			bestValue = value;
			index = t;
		}

	}

	return make_pair(bestValue, index);

}


/* Return the value of a gaussian distribution pdf with mean 0 and std sigma at x */
static float gaussianPDF(const float x, const float sigma) {
	return exp(- x * x / (2 * sigma * sigma)) / (sigma * sqrt(2 * M_PI));
}


/* Update the Hidden Markov Model representing the camera orientation accordingly
 * to the measured angle and the time interval from the last measurement */
static void updateCameraParams(const int angle, const double time) {

	float maxValue = 0.0;
	float normalizer = 0.0;
	float states[HMM_STATES_NUM]; ZERO(states);

	/* For each state calculate the new probability */
	REP(i, HMM_STATES_NUM) {
		REP(j, HMM_STATES_NUM)
			states[i] += HMM_CAM_STATES[j] *
						gaussianPDF(HMM_CAM_PRECISION * (i - j), time / HMM_DISORIENT_SPEED) *
						gaussianPDF(HMM_CAM_PRECISION * (i - angle), HMM_ANGLE_DEV);
		normalizer += states[i];
	}

	/* Normalize states */
	REP(i, HMM_STATES_NUM) {
		HMM_CAM_STATES[i] = states[i] / normalizer;
		maxValue = max(maxValue, HMM_CAM_STATES[i]);
	}

	HMM_DENORM = maxValue;

}


/* Insert the point (x, y) in the circular list EM_SAMPLES of size EM_SAMPLES_NUM and update the mean
 * and covariance matrixes */
static void updateEMparams(const Point_<float>& p) {

	EM_SAMPLES.at<float>(EM_START_INDEX, 0) = p.x; EM_SAMPLES.at<float>(EM_START_INDEX, 1) = p.y;
	EM_START_INDEX = (EM_START_INDEX + 1) % EM_SAMPLES_NUM;

	Mat cov; calcCovarMatrix(EM_SAMPLES, cov, EM_MEAN, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);
	Mat covInv; double det = invert(cov, covInv);
	if (det != 0.0) EM_COVARIANCE_INV = covInv;

}


/* Calculate the non-normalized probability that p is associated with the cluster center */
static float calcEMprob(const Point_<float>& p) {

	Mat x = (Mat_<float>(1, 2, CV_32FC1) << p.x - EM_MEAN.at<float>(0, 0), p.y - EM_MEAN.at<float>(0, 1));
	Mat exponent = x * EM_COVARIANCE_INV * x.t();

	return exp(-0.5f * exponent.at<float>(0, 0));

}


/* Return true if the cooling is over (as in Simulated Annealing) */
static bool isStable(const float delta, const float done) {
	float t = ANNEALING_MAXT * pow(ANNEALING_MINT / ANNEALING_MAXT, done);
	return done >= 1.0f || randReal() > exp(-delta / t);
}


/* Return the sign (i.e. the TEMPLATES index) which border contains the pixel at coordinates cluster.
 * Or -1 if cluster is not part of a sign. */
static int getSign(Mat& mask, const Point_<int>& cluster, Mat& visitedGlobal) {

	double startIterTime = getTime();

	/* Allocate a time slot for this sign w.r.t the camera orientation probability distribution */
	float expectedIter = 0.0f; REP(i, HMM_STATES_NUM) expectedIter += HMM_CAM_STATES[i] / HMM_DENORM;
	float expectedTime = MAX_SIGN_TIME * expectedIter;

	vector<Point_<int> > frontier; frontier.push_back(cluster);

	float bestGlobalValue = 0.0f;
	int bestGlobalSignId = 0;
	int bestGlobalAngle = 0;
	Mat initialVisitedGlobal; visitedGlobal.copyTo(initialVisitedGlobal);

	Rect_<int> bestROI;

	/* Jump to a neighbor solution in the search space by increasing the allowed difference between
	 * the cluster's value and the cluster neighbors' value */
	REP(delta, CLUSTER_MAX_DELTA + 1) {

		Mat visitedDelta; initialVisitedGlobal.copyTo(visitedDelta);
		Rect_<int> roi = Rect_<int>(cluster.x, cluster.y, 1, 1);

		/* Find the roi for this value of delta */
		if(!floodFill(mask, visitedDelta, cluster, 0, &roi, Scalar(delta), Scalar(0), 8 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY))
			continue;

		/* Check if the roi can be a sign w.r.t its dimensions */
		if (roi.width > SIGN_MAX_SIDE || roi.height > SIGN_MAX_SIDE) break;
		if (roi.width < SIGN_MIN_SIDE || roi.height < SIGN_MIN_SIDE) continue;
		if (max(roi.width, roi.height) > SIGN_MAX_RATIO * min(roi.width, roi.height)) continue;

		float bestDeltaValue = 0.0f;
		int bestDeltaSignId = 0;
		int bestDeltaAngle = 0;

		/* Find the best TEMPLATE that match the sign inside roi */
		REP(angle, HMM_STATES_NUM)
		/* Rotate the sign of angle if the camera is likely to be rotated of the same degrees */
		if (randReal() <= HMM_CAM_STATES[angle] / HMM_DENORM) {

			double st1 = getTime();
			uint segment[32]; ZERO(segment); segmentSign(visitedDelta, roi, segment, angle * HMM_CAM_PRECISION - HMM_STATES_NUM + 1);
			timer1 += getTime() - st1;

			double st2 = getTime();
			pair<float, int> sign = evaluate(segment);
			timer2 += getTime() - st2;

			if (sign.first > bestDeltaValue) {

				bestDeltaValue = sign.first;
				bestDeltaSignId = sign.second;
				bestDeltaAngle = angle;

			}

		}

		/* Always accept a solution with higher score than a previous one */
		if (bestDeltaValue > bestGlobalValue) {

			bestGlobalValue = bestDeltaValue;
			bestGlobalSignId = bestDeltaSignId;
			bestGlobalAngle = bestDeltaAngle;

			if (bestGlobalValue > CONFIDENCE_THRESHOLD)	{
				visitedGlobal = visitedDelta;
				bestROI = roi;
			}

#ifdef TEST_LV0
			testScore = max(testScore, bestGlobalValue);
			uchar value = (uchar)min(bestGlobalValue * 255.0f / CONFIDENCE_THRESHOLD, 255.0f);
			rectangle(TEST_IMAGE, roi.tl(), roi.br(), Scalar(0, value, 255 - value, 255), value == 255 ? 2 : 1);
#endif
		}
		/* Search a better solution accordingly to the cooling scheduling
		 * (as in Simulated Annealing), otherwise stop it */
		else if (isStable(bestDeltaValue - bestGlobalValue, (getTime() - startIterTime) / expectedTime)) break;

		/* Update delta for time optimization purposes */
		delta += (int)max(0.0f, 4.0f * (1.0f - bestGlobalValue * 1.0f / CONFIDENCE_THRESHOLD));

	}

	/* Update the Hidden Markov Model for the camera orientation and the EM params */
	if (bestGlobalValue > CONFIDENCE_THRESHOLD) {
		if (SIGN_DETECTED_TIME + EM_UPDATE_DELTA < getTime()) {
			updateCameraParams(bestGlobalAngle, getTime() - SIGN_DETECTED_TIME);
			Point_<float> point = bestROI.tl(); point += Point_<float>(bestROI.width * 0.5f, bestROI.height * 0.5f);
			updateEMparams(point);
		}
		SIGN_DETECTED_TIME = getTime();
	}

#ifdef TEST_LV0

	if (bestGlobalValue > CONFIDENCE_THRESHOLD) {
		if (isNewFrame) {
			debugSignID.clear();
			debugSignValue.clear();
			isNewFrame = false;
		}
		debugSignID.push_back(bestGlobalSignId);
		debugSignValue.push_back(bestGlobalValue);
	}

#endif

	return bestGlobalValue > CONFIDENCE_THRESHOLD ? bestGlobalSignId : -1;

}


/* Return the probability for each pixel to be part of the border of a sign */
static void calcProbabilityMask(Mat& mask) {

	vector<Mat> channelsHSV; split(FRAME_HSV, channelsHSV);

	/* Compute probability that a pixel is part of a sign's border for channels H */
	Mat maskH; LUT(channelsHSV[0], LUT_H, maskH);

	/* Merge together the masks and scale the result such that is in the range [0..128].
	 * It overflows, I know. This is statistically fairly unlikely, though */
	mask = maskH.mul(channelsHSV[1] / 64 + 1).mul(channelsHSV[2] / 64 + 1) / 2;

	/* Noise reduction */
	medianBlur(mask, mask, 3);

#ifdef TEST_LV0
	cvtColor(mask, TEST_IMAGE, CV_GRAY2BGRA, 4);
	TEST_IMAGE *= 2;
#endif

}


/* Process a frame in order to find road signs inside it */
static void processFrame() {

	double st0 = getTime();
	Mat mask; calcProbabilityMask(mask);
	timer0 += getTime() - st0;

	priority_queue< pair<float, Point_<int> >, vector<pair<float, Point_<int> > >, clusterComparator > clusters;

	int CLUSTER_H = FRAME_HSV.rows / CLUSTER_Y_NUM;
	int CLUSTER_W = FRAME_HSV.cols / CLUSTER_X_NUM;

	/* Divide the frame in windows and find the best cluster candidate for each window.
	 * A pixel is the best cluster candidate if it has the highest probability to be
	 * part of a sign in that window */
	REP(i, CLUSTER_Y_NUM) REP(j, CLUSTER_X_NUM) {

		Point p;
		double max;

		Point tl = Point(j * CLUSTER_W, i * CLUSTER_H);
		Point br = Point((j + 1) * CLUSTER_W, (i + 1) * CLUSTER_H);

		Rect roi = Rect(tl, br);
		minMaxLoc(mask(roi), NULL, &max, NULL, &p); p += tl;
		clusters.push(make_pair(max * calcEMprob(p), p));

	}

	Mat visitedGlobal = Mat(FRAME_HSV.size() + Size(2, 2), CV_8UC1, Scalar(false));

	double startCluster = getTime();

	/* Find if a cluster is part of a sign and retrieve which sign starting from
	 * the highest probable cluster downward */
	REP (iter, clusters.size()) {

		if(clusters.empty()) break;

		/* Interrupt the search if the frame's time slot is over */
		if (iter && !(iter & (FILTER_SYNC - 1)) &&
				(iter + FILTER_SYNC) * (getTime() - startCluster) / iter > MAX_FRAME_TIME - startCluster + START_TIME)
			break;

		pair<float, Point_<int> > cls = clusters.top(); clusters.pop();
		if (!visitedGlobal.at<bool>(cls.second.y, cls.second.x)) {

#ifdef TEST_LV0
			circle(TEST_IMAGE, cls.second, 2, Scalar(0, 0, 255));
#endif
			int signID = getSign(mask, cls.second, visitedGlobal);

		}
	}
}


/* Initialize global variables for this frame */
static void initFrame(const Mat& imageBGR) {

	START_TIME = getTime();

#ifdef TEST_LV0
	isNewFrame = true;
#endif

	cvtColor(imageBGR, FRAME_HSV, CV_BGR2HSV, 3);
	vector<Mat> channelsBGR; split(imageBGR, channelsBGR);

	FRAME_B = channelsBGR[0];

}


/* JNI - Initialize global variables for every frames */
JNIEXPORT void JNICALL Java_com_micheledamian_signrecognition_SignRecognitionView_init(JNIEnv* env, jobject object) {

	SIGN_DETECTED_TIME = getTime();

	EM_SAMPLES = (Mat_<float>(EM_SAMPLES_NUM, 2, CV_32FC1) << 360.0f, 180.0f, 240.0f, 90.0f, 240.0f, 270.0f, 120.0f, 180.0f);
	updateEMparams(Point2f(360.0f, 180.0f));

	float initCameraStates[] = { .0043821f, .0134977f, .0323794f, .0604927f, .0880163f, .0997356f, .0880163f, .0604927f, .0323794f, .0134977f, .0043821f };
	COPY(HMM_CAM_STATES, initCameraStates);
	HMM_DENORM = .0997356f;

	LUT_H = Mat(1, 256, CV_8U);

	REP (i, 180)
		LUT_H.at<uchar>(0, i) = (uchar)(1 + 15 * gaussianPDF(min(i - FILTER_MEAN, 180 - i + FILTER_MEAN), FILTER_STD) / FILTER_DENORMALIZER);

}


/* JNI - Process a single frame */
JNIEXPORT void JNICALL Java_com_micheledamian_signrecognition_SignRecognitionView_findSign(JNIEnv* env, jobject object, jint width, jint height, jbyteArray yuv, jintArray bgra) {

	double st = getTime();

	jbyte* _yuv  = env->GetByteArrayElements(yuv, 0);
	jint*  _bgra = env->GetIntArrayElements(bgra, 0);

	Mat mYUV(height + height/2, width, CV_8UC1, (uchar *)_yuv);
	Mat mBGR(height, width, CV_8UC3); cvtColor(mYUV, mBGR, CV_YUV420sp2BGR, 3);

#ifdef TEST_LV0
	TEST_IMAGE = Mat(480, 320, CV_8UC4);
#endif

	Mat BGR; resize(mBGR, BGR, Size(480, 320), 0, 0, INTER_CUBIC);

	initFrame(BGR);
	processFrame();

	double nowTime = getTime();
	double processingTime = nowTime - st;

#ifdef TEST_LV0
	print(1 / processingTime, nowTime - SIGN_DETECTED_TIME);
	Mat out = Mat(height, width, CV_8UC4, (uchar *)_bgra);
	resize(TEST_IMAGE, out, out.size(), 0, 0, INTER_CUBIC);
#endif

	stringstream t0; t0 << "TIMER0 (s) = " << timer0;
	DEBUG(t0.str().c_str());

	stringstream t1; t1 << "TIMER1 (s) = " << timer1;
	DEBUG(t1.str().c_str());

	stringstream t2; t2 << "TIMER2 (s) = " << timer2;
	DEBUG(t2.str().c_str());

	stringstream fr; fr << "FRAME RATE (1/s) = " << (1 / processingTime);
	DEBUG(fr.str().c_str());

	env->ReleaseIntArrayElements(bgra, _bgra, 0);
	env->ReleaseByteArrayElements(yuv, _yuv, 0);

}

}
