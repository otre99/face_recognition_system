#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "./nlohmann/json.hpp"
#include "detection_decoders.h"
#include "vision/predictor.h"

using namespace std;

class FaceDetection {
public:
  void Init();
  void ProcessFrame(const cv::Mat &frame);
  vector<BBox> detections;

private:
  shared_ptr<Predictor> face_detector_{};
  unique_ptr<DetectionDecoder> det_decoder_{};
};

#endif // FACEDETECTION_H
