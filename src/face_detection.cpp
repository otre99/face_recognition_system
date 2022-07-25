#include "face_detection.h"
#include <algorithm>
#include "io_utils.h"

bool FaceDetection::Init(const nlohmann::json &conf) {
    for (const auto &m : conf["models"]){
        if ( m.value("type", "") == "detection" ){
            face_detector_ = PredictorFromJson(m);
            det_decoder_ = DetectionDecoderFromJson(m);
        }
        if ( m.value("type", "") == "landmarks" ){
            face_landmarks_ = PredictorFromJson(m);
        }
    }
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

FaceLandmarks FaceDetection::GetFaceLandmarks(const cv::Mat &frame, const TrackedObject &obj, bool use_retinanet){

    if (use_retinanet){
        if (det_decoder_->GetName() == "RETINAFACE"){
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

    if (face_landmarks_){
        vector<cv::Mat> outputs;
        face_landmarks_->Predict(frame(obj.rect), outputs, {"conv6-3_Gemm_Y"});
        cv::Mat regressionsBlob = outputs[0];
        cout << regressionsBlob << endl;
        return GetFaceLandmarksFromOnetOutpus(obj.rect, regressionsBlob);
    }
    throw "Error: There is not way to calculate face landmarks";
}

FaceLandmarks   FaceDetection::GetFaceLandmarksFromOnetOutpus(const cv::Rect &rect, const cv::Mat &lands)
{
    const float *landmark_data = (float *)lands.data;
    FaceLandmarks l;
    l.leye.x = rect.x + landmark_data[5 + 0] * rect.width - 1;
    l.leye.y = rect.y + landmark_data[0 + 0] * rect.height - 1;

    l.reye.x = rect.x + landmark_data[5 + 1] * rect.width - 1;
    l.reye.y = rect.y + landmark_data[0 + 1] * rect.height - 1;

    l.nose.x = rect.x + landmark_data[5 + 2] * rect.width - 1;
    l.nose.y = rect.y + landmark_data[0 + 2] * rect.height - 1;

    l.lmouth.x = rect.x + landmark_data[5 + 3] * rect.width - 1;
    l.lmouth.y = rect.y + landmark_data[0 + 3] * rect.height - 1;

    l.rmouth.x = rect.x + landmark_data[5 + 4] * rect.width - 1;
    l.rmouth.y = rect.y + landmark_data[0 + 4] * rect.height - 1;
    l.relative_coords=false;
    return l;
}

FaceLandmarks  FaceDetection::GetFaceLandmarks(const cv::Mat &frame, const vector<cv::Rect> &bboxes){
    vector<cv::Mat> images(bboxes.size());
    for (size_t i=0 ; i<images.size(); ++i){
        images[i] = frame(bboxes[i]);
    }

    vector<vector<cv::Mat>> outputs;
    face_landmarks_->Predict(images,outputs);
    vector<FaceLandmarks> l(bboxes.size());
}

