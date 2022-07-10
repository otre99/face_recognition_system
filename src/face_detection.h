#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "./nlohmann/json.hpp"
#include "vision/detection_decoders.h"
#include "vision/predictor.h"
#include "vision/tracker.h"

using namespace std;

class FaceDetection {
public:
  void Init(const nlohmann::json &conf);
  void Process(const cv::Mat &frame);
  void DetecFaces(const cv::Mat &frame);


private:
  vector<BBox> recent_detections_;
  shared_ptr<Predictor> face_detector_{};
  unique_ptr<DetectionDecoder> det_decoder_{};
  Tracker faces_tracker_;
  int face_label_id_{-1};
};

#endif // FACEDETECTION_H
