#include "face_detection.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "draw_utils.h"
#include "other_code/pose_detector.h"

using namespace std;
namespace fs = filesystem;

DrawUtils drawer;

void DrawDetections(cv::Mat &frame, const vector<TrackedObject> &bboxes){
    for (const auto &o : bboxes){
        drawer.DrawTrackedObj(frame, o);
    }
}

void DrawFaces(cv::Mat &frame, const vector<Face> &faces){
    for (const auto &f : faces){
        drawer.DrawFace(frame, f);
    }
}

//Pose CalcPose(const FaceLandmarks &l){
//    PoseDetector pose;
//    vector<cv::Point2f> landmarks;
//    landmarks.push_back({});
//    landmarks.push_back(l.leye);
//    landmarks.push_back(l.reye);
//    landmarks.push_back(l.nose);
//    landmarks.push_back(0.5*(l.lmouth+l.rmouth));
//    auto p  = pose.Detect(landmarks);
//    return p;
//}

int main(int argc, char *argv[]) {

    auto net = cv::dnn::readNetFromONNX("/test_storage/models/Onet1.onnx");
    auto image = cv::imread("/test_storage/image_samples/Person00_crop.jpg");
    auto blob = cv::dnn::blobFromImage(image, 1.0/128, {48,48}, {127.5, 127.5, 127.5}, true);
    net.setInput(blob);
    vector<cv::Mat> outputs;
    net.forward(outputs, vector<string>{"conv6-3_Gemm_Y"});
    cout << outputs[0] << endl;
    return 0;
  FaceDetection faceDet;
  {
    auto data = LoadJSon(argv[1]);
    faceDet.Init(data["face_detection"]);
  }
  drawer.Init({});

  cv::VideoCapture cap("/home/jetson/sample-videos-master/head-pose-face-detection-female-and-male.mp4");
  cv::Mat frame;
  vector<TrackedObject> tobjects;
  vector<Face> faces;
  while (cap.read(frame)){
    tobjects = faceDet.Process(frame);

    faces.resize(tobjects.size());
    for (size_t i=0; i<tobjects.size(); ++i){
        auto l = faceDet.GetFaceLandmarks(frame, tobjects[i], false);
        faces[i].Init(frame, tobjects[i].rect, l, tobjects[i].id);

//        auto p = CalcPose(l);
//        cout << "Roll  " << faces[i].roll_ << " -- " << p.roll << endl;
//        cout << "Pitch " << faces[i].pitch_ << " -- " << p.pitch << endl;
//        cout << "Yaw   " << faces[i].yaw_ << " -- " << p.yaw << endl;
//        cout << "\n\n";
    }
    DrawFaces(frame,faces);
    cv::imshow("Dets", frame);
    if (cv::waitKey(1)==27) break;
  }
  return 0;
}
