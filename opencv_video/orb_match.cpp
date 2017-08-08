// #include <opencv2/opencv.hpp>
// #include <vector>
// #include <iostream>
// #include <ctime>
// #include <map>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/gms_matcher.h"
#include "include/Header.h"

// using namespace std;
// using namespace cv;
using namespace cv::xfeatures2d;
//CvCapture *cap;
void FlannMatch(Mat &_img_object, Mat &_img_scene, vector<KeyPoint> &_keypoints_object, vector<KeyPoint> &_keypoints_scene,
                Mat &_descriptors_object, Mat &_descriptors_scene, Mat &_img_matches, vector<DMatch> &_matches, vector<DMatch> &_good_matches);
void GmsMatch(Mat &_img_object, Mat &_img_scene, vector<KeyPoint> &_keypoints_object, vector<KeyPoint> &_keypoints_scene, Mat &_descriptors_object,
                Mat &_descriptors_scene, Mat &_img_matches, vector<DMatch> _matches, vector<DMatch> _good_matches);

int main(int argc, char *argv[]) {
    if(argc != 3) {
        cout << "usage: " << argv[0] << " <source image> <method: 0, 1 or 3>\n";
        return -1;
    }

    Mat img_object = imread(argv[1]);
    resize(img_object, img_object, Size(640,1000));
    if( !img_object.data ) { cout << "Err: reading object image failed...\n";}

    VideoCapture cap(0);
    //cap = cvCaptureFromCAM(0);
    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    if(!cap.isOpened())
    //if(!cap)
        return -1;

    // char* method = argv[2];

    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    Ptr<ORB> orb = ORB::create(10000);

    int minHessian = 400;
    //SurfFeatureDetector detector(minHessian);
    Ptr<SURF> detector = SURF::create(minHessian);
    //SurfDescriptorExtractor extractor;
    Ptr<SURF> extractor = SURF::create();

    //-- object
    if(0 == strcmp(argv[2], "0")) { //-- ORB
        // orb.detect(img_object, keypoints_object);
        // //drawKeypoints(img_object, keypoints_object, img_object, Scalar(0,255,255));
        // //imshow("template", img_object);
        //
        // orb.compute(img_object, keypoints_object, descriptors_object);
        orb->detectAndCompute(img_object, Mat(), keypoints_object, descriptors_object);
    } else if(0 == strcmp(argv[2], "1")){ //-- SURF test
        // detector.detect(img_object, keypoints_object);
        // extractor.compute(img_object, keypoints_object, descriptors_object);
        detector->detect(img_object, keypoints_object);
        extractor->compute(img_object, keypoints_object, descriptors_object);
    }else if(0 == strcmp(argv[2], "2")){
        orb->setFastThreshold(0);
        orb->detectAndCompute(img_object, Mat(), keypoints_object, descriptors_object);
    }
    // http://stackoverflow.com/a/11798593
    //if(descriptors_object.type() != CV_32F)
    //    descriptors_object.convertTo(descriptors_object, CV_32F);


    while(true){
        Mat frame;

        cap >> frame;

        if(frame.empty())
          break;

        //Mat img_scene = Mat(frame.size(), CV_8UC1);
        Mat img_scene;
        //Mat tmp_img;
        frame.copyTo(img_scene);
        //cvtColor(frame, img_scene, COLOR_RGB2GRAY);

        Mat img_matches;
        vector<DMatch> matches;
        vector<DMatch> good_matches;

        if(0 == strcmp(argv[2], "0")) { //-- ORB
            orb->detectAndCompute(img_scene, Mat(), keypoints_scene, descriptors_scene);
            // orb->compute(img_scene, keypoints_scene, descriptors_scene);
            //-- matching descriptor vectors using FLANN matcher
            FlannMatch(img_object, img_scene, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, img_matches, matches, good_matches);

        }else if(0 == strcmp(argv[2], "1")){ //-- SURF
            detector->detect(img_scene, keypoints_scene);
            extractor->compute(img_scene, keypoints_scene, descriptors_scene);
            FlannMatch(img_object, img_scene, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, img_matches, matches, good_matches);
        }else if(0 == strcmp(argv[2], "2")){
            orb->setFastThreshold(0);
            orb->detectAndCompute(img_scene, Mat(), keypoints_scene, descriptors_scene);
            GmsMatch(img_object, img_scene, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, img_matches, matches, good_matches);
        }

        imshow("match result", img_matches );

        int ch = waitKey(30);
        if(ch == 27)
          break;
        // if(waitKey(30) >= 0) break;
    }
    cout << "准备释放摄像头\n";
    cap.release();
    cout << "已释放摄像头\n";

    return 0;
}

void FlannMatch(Mat &_img_object, Mat &_img_scene, vector<KeyPoint> &_keypoints_object, vector<KeyPoint> &_keypoints_scene,
                Mat &_descriptors_object, Mat &_descriptors_scene, Mat &_img_matches, vector<DMatch> &_matches, vector<DMatch> &_good_matches){
    FlannBasedMatcher matcher;
    if(_descriptors_scene.type()!=CV_32F) {
      _descriptors_scene.convertTo(_descriptors_scene, CV_32F);
    }

    if(_descriptors_object.type()!=CV_32F) {
      _descriptors_object.convertTo(_descriptors_object, CV_32F);
    }

    if(!_descriptors_object.empty() && !_descriptors_scene.empty()) {
        matcher.match(_descriptors_object, _descriptors_scene, _matches);

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min idstance between keypoints
        for( int i = 0; i < _descriptors_object.rows; i++)
        { double dist = _matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        //printf("-- Max dist : %f \n", max_dist );
        //printf("-- Min dist : %f \n", min_dist );
        //-- Draw only good matches (i.e. whose distance is less than 3*min_dist)

        for( int i = 0; i < _descriptors_object.rows; i++ )
        { if( _matches[i].distance < 2*min_dist )
            { _good_matches.push_back( _matches[i]); }
        }

        drawMatches(_img_object, _keypoints_object, _img_scene, _keypoints_scene,
                _good_matches, _img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        //-- localize the object
        // vector<Point2f> obj;
        // vector<Point2f> scene;

        // for( size_t i = 0; i < _good_matches.size(); i++) {
            //-- get the keypoints from the good matches
        //     obj.push_back( _keypoints_object[ _good_matches[i].queryIdx ].pt );
        //     scene.push_back( _keypoints_scene[ _good_matches[i].trainIdx ].pt );
        // }
        // if( !obj.empty() && !scene.empty() && _good_matches.size() >= 4) {
        //     Mat H = findHomography( obj, scene, RANSAC );

            //-- get the corners from the object to be detected
            // vector<Point2f> obj_corners(4);
            // obj_corners[0] = Point(0,0);
            // obj_corners[1] = Point(_img_object.cols,0);
            // obj_corners[2] = Point(_img_object.cols,_img_object.rows);
            // obj_corners[3] = Point(0,_img_object.rows);
            //
            // vector<Point2f> scene_corners(4);
            //
            // perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            // line( _img_matches,
            //         scene_corners[0] + Point2f(_img_object.cols, 0),
            //         scene_corners[1] + Point2f(_img_object.cols, 0),
            //         Scalar(0,255,0), 4 );
            // line( _img_matches,
            //         scene_corners[1] + Point2f(_img_object.cols, 0),
            //         scene_corners[2] + Point2f(_img_object.cols, 0),
            //         Scalar(0,255,0), 4 );
            // line( _img_matches,
            //         scene_corners[2] + Point2f(_img_object.cols, 0),
            //         scene_corners[3] + Point2f(_img_object.cols, 0),
            //         Scalar(0,255,0), 4 );
            // line( _img_matches,
            //         scene_corners[3] + Point2f(_img_object.cols, 0),
            //         scene_corners[0] + Point2f(_img_object.cols, 0),
            //         Scalar(0,255,0), 4 );

        // }
    }
}

void GmsMatch(Mat &_img_object, Mat &_img_scene, vector<KeyPoint> &_keypoints_object, vector<KeyPoint> &_keypoints_scene, Mat &_descriptors_object, Mat &_descriptors_scene,
              Mat &_img_matches, vector<DMatch> _matches, vector<DMatch> _good_matches){
  BFMatcher matcher(NORM_HAMMING);
  matcher.match(_descriptors_object, _descriptors_scene, _matches);

  // GMS filter
  int num_inliers = 0;
  std::vector<bool> vbInliers;
  gms_matcher gms(_keypoints_object, _img_object.size(), _keypoints_scene, _img_scene.size(), _matches);
  num_inliers = gms.GetInlierMask(vbInliers, false, true);

  // cout << "Get total " << num_inliers << " matches." << endl;
  // draw matches
  for (size_t i = 0; i < vbInliers.size(); ++i)
  {
    if (vbInliers[i] == true)
    {
      _good_matches.push_back(_matches[i]);
    }
  }

  // drawMatches(_img_object, _keypoints_object, _img_scene, _keypoints_scene,
  //         _good_matches, _img_matches, Scalar::all(-1), Scalar::all(-1),
  //         vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
   _img_matches = DrawInlier(_img_object, _img_scene, _keypoints_object, _keypoints_scene, _good_matches, 1);
}
