#include "face_detection.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "draw_utils.h"

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
  FaceDetection faceDet;
  {
    auto data = LoadJSon(argv[1]);
    faceDet.Init(data["face_detection"]);
  }
  drawer.Init({});

  cv::VideoCapture cap("/home/jetson/sample-videos-master/face-demographics-walking-and-pause.mp4");
  cv::Mat frame;
  vector<TrackedObject> tobjects;
  vector<Face> faces;
  while (cap.read(frame)){
    tobjects = faceDet.Process(frame);

    faces.resize(tobjects.size());
    for (size_t i=0; i<tobjects.size(); ++i){
        auto l = faceDet.GetFaceLandmarks(tobjects[i], true);
        faces[i].Init(frame, tobjects[i].rect, l, tobjects[i].id);
    }
    DrawFaces(frame,faces);
    cv::imshow("Dets", frame);
    if (cv::waitKey(1)==27) break;
  }

  return 0;
}
