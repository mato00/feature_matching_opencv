#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
//CvCapture *cap;

int main(int argc, char *argv[]) {
    if(argc != 3) {
        cout << "usage: " << argv[0] << " <source image> <method: 0 or 1>\n";
        return -1;
    }

    Mat img_object = imread(argv[1], IMREAD_GRAYSCALE);
    if( !img_object.data ) { cout << "Err: reading object image failed...\n";}

    VideoCapture cap(0);
    //cap = cvCaptureFromCAM(0);
    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    if(!cap.isOpened())
    //if(!cap)
        return -1;

    char* method = argv[2];

    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    Ptr<ORB> orb = ORB::create(10000);

    int minHessian = 400;
    //SurfFeatureDetector detector(minHessian);
    Ptr<SURF> detector = SURF::create(minHessian);
    //SurfDescriptorExtractor extractor;
    Ptr<SURF> extractor = SURF::create();

    //-- object
    if( method == 0 ) { //-- ORB
        // orb.detect(img_object, keypoints_object);
        // //drawKeypoints(img_object, keypoints_object, img_object, Scalar(0,255,255));
        // //imshow("template", img_object);
        //
        // orb.compute(img_object, keypoints_object, descriptors_object);
        orb->detectAndCompute(img_object, Mat(), keypoints_object, descriptors_object);
    } else { //-- SURF test
        // detector.detect(img_object, keypoints_object);
        // extractor.compute(img_object, keypoints_object, descriptors_object);
        detector->detect(img_object, keypoints_object);
        extractor->compute(img_object, keypoints_object, descriptors_object);
    }
    // http://stackoverflow.com/a/11798593
    //if(descriptors_object.type() != CV_32F)
    //    descriptors_object.convertTo(descriptors_object, CV_32F);


    for(;;) {
        //IplImage* frame0;
        Mat frame;

        // cap.set(CV_CAP_PROP_FRAME_WIDTH , 640);
        // cap.set(CV_CAP_PROP_FRAME_HEIGHT , 480);
        // cap.set (CV_CAP_PROP_FOURCC, CV_FOURCC('B', 'G', 'R', '3'));//diff from mine, using as example
        cap >> frame;
        //cap.operator>>(frame);
        //frame0 = cvQueryFrame(cap);
        //Mat frame = cvarrToMat(frame0, true);
        //waitKey(1000);

        Mat img_scene = Mat(frame.size(), CV_8UC1);
        cvtColor(frame, img_scene, COLOR_RGB2GRAY);

        if( method == 0 ) { //-- ORB
            orb->detect(img_scene, keypoints_scene);
            orb->compute(img_scene, keypoints_scene, descriptors_scene);
        } else { //-- SURF
            detector->detect(img_scene, keypoints_scene);
            extractor->compute(img_scene, keypoints_scene, descriptors_scene);
        }

        //-- matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        Mat img_matches;

        if(!descriptors_object.empty() && !descriptors_scene.empty()) {
            matcher.match(descriptors_object, descriptors_scene, matches);

            double max_dist = 0; double min_dist = 100;

            //-- Quick calculation of max and min idstance between keypoints
            for( int i = 0; i < descriptors_object.rows; i++)
            { double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }
            //printf("-- Max dist : %f \n", max_dist );
            //printf("-- Min dist : %f \n", min_dist );
            //-- Draw only good matches (i.e. whose distance is less than 3*min_dist)
            vector< DMatch >good_matches;

            for( int i = 0; i < descriptors_object.rows; i++ )
            { if( matches[i].distance < 3*min_dist )
                { good_matches.push_back( matches[i]); }
            }

            drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //-- localize the object
            vector<Point2f> obj;
            vector<Point2f> scene;

            for( size_t i = 0; i < good_matches.size(); i++) {
                //-- get the keypoints from the good matches
                obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
            }
            if( !obj.empty() && !scene.empty() && good_matches.size() >= 4) {
                Mat H = findHomography( obj, scene, RANSAC );

                //-- get the corners from the object to be detected
                vector<Point2f> obj_corners(4);
                obj_corners[0] = Point(0,0);
                obj_corners[1] = Point(img_object.cols,0);
                obj_corners[2] = Point(img_object.cols,img_object.rows);
                obj_corners[3] = Point(0,img_object.rows);

                vector<Point2f> scene_corners(4);

                perspectiveTransform( obj_corners, scene_corners, H);

                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                line( img_matches,
                        scene_corners[0] + Point2f(img_object.cols, 0),
                        scene_corners[1] + Point2f(img_object.cols, 0),
                        Scalar(0,255,0), 4 );
                line( img_matches,
                        scene_corners[1] + Point2f(img_object.cols, 0),
                        scene_corners[2] + Point2f(img_object.cols, 0),
                        Scalar(0,255,0), 4 );
                line( img_matches,
                        scene_corners[2] + Point2f(img_object.cols, 0),
                        scene_corners[3] + Point2f(img_object.cols, 0),
                        Scalar(0,255,0), 4 );
                line( img_matches,
                        scene_corners[3] + Point2f(img_object.cols, 0),
                        scene_corners[0] + Point2f(img_object.cols, 0),
                        Scalar(0,255,0), 4 );

            }
        }
        imshow("match result", img_matches );

        if(waitKey(30) >= 0) break;
    }

    return 0;
}
