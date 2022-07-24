#include "face_detection.h"
#include <algorithm>
#include "io_utils.h"

bool FaceDetection::Init(const nlohmann::json &conf) {

    face_detector_ = PredictorFromJson(conf["model"]);
    det_decoder_ = DetectionDecoderFromJson(conf["model"]);
    face_label_id_ = conf.value("face_label_id", 1);
    TrackerFromJson(conf["tracker"], faces_tracker_);
    return true;
}

const std::vector<TrackedObject> &FaceDetection::Process(const cv::Mat &frame) {
    DetecFaces(frame);
    user_ids_.resize(recent_detections_.size());
    iota(user_ids_.begin(), user_ids_.end(), 0);
    faces_tracker_.Process(recent_detections_,face_label_id_, user_ids_);
    return faces_tracker_.GetTrackedObjects();
}

void FaceDetection::DetecFaces(const cv::Mat &frame){
    vector<cv::Mat> outputs;
    face_detector_->Predict(frame, outputs, det_decoder_->ExpectedLayerNames());
    det_decoder_->Decode(outputs, recent_detections_, det_decoder_->ExpectedLayerNames(),
                         frame.size());
}

FaceLandmarks FaceDetection::GetFaceLandmarks(const TrackedObject &obj, bool use_retinanet){

    if (use_retinanet){
        if (det_decoder_->GetName() == "RetinaFace"){
            auto ptr = dynamic_cast<RetinaFaceDecoder*>(det_decoder_.get());
            const int ii = obj.user_id;
            if (obj.last_frame != 0){
                cerr << "Warning: RetinaNet face landmarks output is only available for recent detections " << endl;
            } else {
                return ptr->landmarks_[ii];
            }
        } else {
            cerr << "Warning: RetinaNet face landmarks output is only when RetinaNet model is used " << endl;
        }
    }
    //TODO(otre99): implement this part
}

