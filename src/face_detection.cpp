#include "face_detection.h"
#include <algorithm>
#include "io_utils.h"

void FaceDetection::Init(const nlohmann::json &conf) {

    face_detector_ = PredictorFromJson(conf["model"]);
    TrackerFromJson(conf["tracker"], faces_tracker_);
}

void FaceDetection::Process(const cv::Mat &frame) {
    DetecFaces(frame);
    faces_tracker_.Process(recent_detections_,face_label_id_);
}

void FaceDetection::DetecFaces(const cv::Mat &frame){
    vector<cv::Mat> outputs;
    face_detector_->Predict(frame, outputs, det_decoder_->ExpectedLayerNames());
    det_decoder_->Decode(outputs, recent_detections_, det_decoder_->ExpectedLayerNames(),
                         frame.size());
}

vector<Face> DetecFacesAndAlign(const cv::Mat &frame){

}
