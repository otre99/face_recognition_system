#include "faces_manager.h"
#include "io_utils.h"
#include <algorithm>

bool FacesManager::Init(const nlohmann::json &conf) {
  cout << "FacesManager setting up ..." << endl;
  for (const auto &m : conf["models"]) {
    if (m.value("type", "") == "detection") {
      face_detector_ = PredictorFromJson(m);
      det_decoder_ = DetectionDecoderFromJson(m);
    }
    if (m.value("type", "") == "landmarks") {
      face_landmarks_ = PredictorFromJson(m);
    }

    if (m.value("type", "") == "recognition") {
      face_recognition_ = PredictorFromJson(m);
    }
  }
  TrackerFromJson(conf["tracker"], faces_tracker_);

  align_method_ = conf.value("align_method", 0);
  cout << "Align method  : " << align_method_ << endl;

  min_box_side_ = conf.value("min_box_side", -1);
  cout << "Min. box side : " << min_box_side_ << endl;

  face_label_id_ = conf.value("face_label_id", 1);
  cout << "Face label id : " << face_label_id_ << endl;

  embedding_len_ = conf.value("embedding_len", -1);
  cout << "Embedding len : " << embedding_len_ << endl;

  roll_lims_ = GetPairFromJson(conf["roll_lims"]);
  cout << "Roll lims     : [ " << roll_lims_.first << " , " << roll_lims_.second
       << "] " << endl;

  pitch_lims_ = GetPairFromJson(conf["pitch_lims"]);
  cout << "Pitch lims    : [ " << pitch_lims_.first << " , "
       << pitch_lims_.second << "] " << endl;

  yaw_lims_ = GetPairFromJson(conf["roll_lims"]);
  cout << "Yaw lims      : [ " << yaw_lims_.first << " , " << yaw_lims_.second
       << "] " << endl;

  cout << "FacesManager setting up ... DONE" << endl;
  return true;
}

const std::vector<TrackedObject> &FacesManager::Process(const cv::Mat &frame) {
  DetecFaces(frame);
  user_ids_.resize(recent_detections_.size());
  iota(user_ids_.begin(), user_ids_.end(), 0);
  faces_tracker_.Process(recent_detections_, face_label_id_, user_ids_);
  return faces_tracker_.GetTrackedObjects();
}

void FacesManager::DetecFaces(const cv::Mat &frame) {
  vector<cv::Mat> outputs;
  face_detector_->Predict(frame, outputs, det_decoder_->ExpectedLayerNames());
  det_decoder_->Decode(outputs, recent_detections_,
                       det_decoder_->ExpectedLayerNames(), frame.size());
}

FaceLandmarks
FacesManager::GetFaceLandmarksRetinaFace(const cv::Mat &frame,
                                         const TrackedObject &obj) {

    const auto lands = GetRecentRetinaFaceLandmarks();
    const int ii = obj.user_id;
    if (obj.last_frame != 0) {
        throw "GetFaceLandmarksRetinaFace(): RetinaNet face landmarks output is "
              "only available for recent detections ";
    }
    return lands[ii];
}

vector<FaceLandmarks> &FacesManager::GetRecentRetinaFaceLandmarks() const{
    if (det_decoder_->GetName() == "RETINAFACE") {
        auto ptr = dynamic_cast<RetinaFaceDecoder *>(det_decoder_.get());
        return ptr->landmarks_;
    }
    throw "GetRecentRetinaFaceLandmarks(): face landmarks from RetinaFace are only "
          "available when RetinaFace detector is used";
}

FaceLandmarks
FacesManager::GetFaceLandmarksFromOnetOutputs(const cv::Rect &rect,
                                              const float *landmark_data) {
  FaceLandmarks l;
  l.leye.x = rect.x + landmark_data[0 + 0] * rect.width - 1;
  l.leye.y = rect.y + landmark_data[5 + 0] * rect.height - 1;

  l.reye.x = rect.x + landmark_data[0 + 1] * rect.width - 1;
  l.reye.y = rect.y + landmark_data[5 + 1] * rect.height - 1;

  l.nose.x = rect.x + landmark_data[0 + 2] * rect.width - 1;
  l.nose.y = rect.y + landmark_data[5 + 2] * rect.height - 1;

  l.lmouth.x = rect.x + landmark_data[0 + 3] * rect.width - 1;
  l.lmouth.y = rect.y + landmark_data[5 + 3] * rect.height - 1;

  l.rmouth.x = rect.x + landmark_data[0 + 4] * rect.width - 1;
  l.rmouth.y = rect.y + landmark_data[5 + 4] * rect.height - 1;
  l.relative_coords = false;
  return l;
}

pair<float, float> FacesManager::GetPairFromJson(const nlohmann::json &conf) {
  return {conf[0].get<float>(), conf[1].get<float>()};
}

vector<FaceLandmarks>
FacesManager::GetFaceLandmarksOnet(const cv::Mat &frame,
                                   const vector<cv::Rect> &bboxes) {
  vector<cv::Mat> images(bboxes.size());
  for (size_t i = 0; i < images.size(); ++i) {
    images[i] = frame(bboxes[i]);
  }

  vector<cv::Mat> outputs;
  face_landmarks_->Predict(images, outputs);
  vector<FaceLandmarks> l(bboxes.size());

  for (size_t i = 0; i < images.size(); ++i) {
    l[i] = GetFaceLandmarksFromOnetOutputs(bboxes[i],
                                           outputs[0].ptr<float>(i * 10));
  }
  return l;
}

FaceLandmarks FacesManager::GetFaceLandmarksOnet(const cv::Mat &frame,
                                                 const cv::Rect &bbox) {
  vector<cv::Mat> outputs;
    face_landmarks_->Predict(frame(RectInsideFrame(bbox, frame)), outputs, {"landmarks"});
  return GetFaceLandmarksFromOnetOutputs(bbox, outputs[0].ptr<float>(0));
}

vector<float> FacesManager::GetFaceEmbedding(const cv::Mat &face_img) {
  vector<cv::Mat> o;
  face_recognition_->Predict(face_img, o);
  return vector<float>(o[0].ptr<float>(0), o[0].ptr<float>(0) + o[0].total());
}

bool FacesManager::IsFrontal(const Face &face) const {
  const float r = face.GetRoll();
  const float p = face.GetPitch();
  const float y = face.GetYaw();
  return (roll_lims_.first <= r) && (r <= roll_lims_.second) &&
         (pitch_lims_.first <= p) && (p <= pitch_lims_.second) &&
         (yaw_lims_.first <= y) && (y <= yaw_lims_.second);
}

bool FacesManager::IsGoodForRegcognition(const Face &face) const {
  return IsFrontal(face) && (face.det_rect_.width >= min_box_side_);
}
