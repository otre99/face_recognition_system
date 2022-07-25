#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "./nlohmann/json.hpp"
#include "detection_decoders.h"
#include "predictor.h"
#include "tracker.h"
#include "face.h"

using namespace std;

class FaceDetection {
public:
  bool Init(const nlohmann::json &conf);
  const std::vector<TrackedObject> &Process(const cv::Mat &frame);
  void DetecFaces(const cv::Mat &frame);

  const vector<BBox> &GetRecentDetections() const {
      return recent_detections_;
  }

  FaceLandmarks GetFaceLandmarks(const cv::Mat &frame, const TrackedObject &obj, bool use_retinanet=false);
  FaceLandmarks GetFaceLandmarks(const cv::Mat &frame, const vector<cv::Rect> &bboxes);


private:
  FaceLandmarks GetFaceLandmarksFromOnetOutpus(const cv::Rect &rect, const cv::Mat &lands);

  vector<BBox> recent_detections_;
  shared_ptr<Predictor> face_detector_{};
  shared_ptr<Predictor> face_landmarks_{};
  shared_ptr<DetectionDecoder> det_decoder_{};
  Tracker faces_tracker_;
  vector<int> user_ids_;

  int face_label_id_{-1};
};

#endif // FACEDETECTION_H
