#include "face_detection.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "face.h"


using namespace std;
namespace fs = filesystem;


void DrawDetections(cv::Mat &frame, const vector<BBox> &bboxes){
    for (const auto &box : bboxes){
        cv::rectangle(frame, box.rect, {255,0,128}, 3);
    }
}


int main(int argc, char *argv[]) {
  FaceDetection faceDet;
  {
    auto data = LoadJSon(argv[1]);
    faceDet.Init(data["face_detection"]);
  }

  cv::VideoCapture cap(0);
  cv::Mat frame;
  while (cap.read(frame)){
    faceDet.DetecFaces(frame);
    DrawDetections(frame, faceDet.GetRecentDetections());
    cv::imshow("Dets", frame);
    cv::waitKey(-1);
  }

  return 0;
}
