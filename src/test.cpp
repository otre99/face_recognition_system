#include "face_detection.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "draw_utils.h"
#include "opencv_predictor.h"
//#include "other_code/pose_detector.h"

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


int main(int argc, char *argv[]) {

    auto image = cv::imread("/test_storage/image_samples/Person00_crop.jpg");
    vector<cv::Mat> images{image, image, image};

    auto p = OpenCVPredictor::Create("/test_storage/models/OnetD.onnx", "", "ONNX");
    p->SetInputParamsNorm(1.0/128, {127.5, 127.5, 127.5}, true);
    p->setExplictInputSize({48, 48});

    vector<cv::Mat> outputs;
    p->Predict(images, outputs,{"landmarks"});
    cout << outputs[0] << endl;
    return 0;

//  FaceDetection faceDet;
//  {
//    auto data = LoadJSon(argv[1]);
//    faceDet.Init(data["face_detection"]);
//  }
//  drawer.Init({});

//  cv::VideoCapture cap("/home/jetson/sample-videos/head-pose-face-detection-female-and-male.mp4");
//  //cv::VideoCapture cap(0);
//  cv::Mat frame;
//  vector<TrackedObject> tobjects;
//  vector<Face> faces;
//  while (cap.read(frame)){
//    tobjects = faceDet.Process(frame);



//    vector<cv::Rect> boxes;
//    for (const auto &o : tobjects){
//        if (o.last_frame==0){
//            boxes.push_back(o.rect);
//        }
//    }
//    auto ls = faceDet.GetFaceLandmarksOnet(frame,boxes);
//    faces.resize(ls.size());
//    for (size_t i=0; i<boxes.size(); ++i){
//        faces[i].Init(frame, boxes[i], ls[i], -1);
//    }



//    /*
//    for (size_t i=0; i<tobjects.size(); ++i){

//        FaceLandmarks l;
//        bool use_retinaface=false;
//        if (use_retinaface){
//            l = faceDet.GetFaceLandmarksRetinaFace(frame, tobjects[i]);
//            faces[i].Init(frame, tobjects[i].rect, l, tobjects[i].id);
//        } else {
//           l = faceDet.GetFaceLandmarksOnet(frame, tobjects[i].rect);
//           faces[i].Init(frame, tobjects[i].rect, l, tobjects[i].id);
//        }

////        auto p = CalcPose(l);
////        cout << "Roll  " << faces[i].roll_ << " -- " << p.roll << endl;
////        cout << "Pitch " << faces[i].pitch_ << " -- " << p.pitch << endl;
////        cout << "Yaw   " << faces[i].yaw_ << " -- " << p.yaw << endl;
////        cout << "\n\n";
//    }
//    */
//    DrawFaces(frame,faces);
//    cv::imshow("Dets", frame);
//    if (cv::waitKey(1)==27) break;
//  }
  return 0;
}
