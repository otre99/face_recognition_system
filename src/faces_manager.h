#ifndef FACE_MANAGER_H
#define FACE_MANAGER_H

#include "./nlohmann/json.hpp"
#include "detection_decoders.h"
#include "face.h"
#include "predictor.h"
#include "tracker.h"

using namespace std;

class FacesManager {
public:
  bool Init(const nlohmann::json &conf);
  const std::vector<TrackedObject> &Process(const cv::Mat &frame);
  void DetecFaces(const cv::Mat &frame);
  int GetAlignMethod() const { return align_method_; }
  int GetEmbeddingLen() const { return embedding_len_; }
  const vector<BBox> &GetRecentDetections() const { return recent_detections_; }

  FaceLandmarks GetFaceLandmarksRetinaFace(const cv::Mat &frame,
                                           const TrackedObject &obj);
  vector<FaceLandmarks> GetFaceLandmarksOnet(const cv::Mat &frame,
                                             const vector<cv::Rect> &bboxes);
  FaceLandmarks GetFaceLandmarksOnet(const cv::Mat &frame,
                                     const cv::Rect &bbox);
  vector<float> GetFaceEmbedding(const cv::Mat &face_img);
  bool IsFrontal(const Face &face) const;

private:
  FaceLandmarks GetFaceLandmarksFromOnetOutputs(const cv::Rect &rect,
                                                const float *landmark_data);

  pair<float, float> GetPairFromJson(const nlohmann::json &conf);


  vector<BBox> recent_detections_;
  shared_ptr<Predictor> face_detector_{};
  shared_ptr<Predictor> face_landmarks_{};
  shared_ptr<Predictor> face_recognition_{};

  shared_ptr<DetectionDecoder> det_decoder_{};
  Tracker faces_tracker_;
  vector<int> user_ids_;

  int face_label_id_{-1};
  int align_method_{0};
  int embedding_len_{-1};
  pair<float, float> roll_lims_;
  pair<float, float> yaw_lims_;
  pair<float, float> pitch_lims_;
};

#endif // FACE_MANAGER_H
