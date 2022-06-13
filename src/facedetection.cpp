#include "facedetection.h"
#include "cvision/funct_utils.h"
#include <algorithm>


void FaceDetection::Init(const nlohmann::json &conf)
{
    batch_size_ = conf.value("batch_size",-1);
    face_det_ = PredictorFromJson(conf["face_det_model"]);
    det_decoder_ = DetectionDecoderFromJson(conf["face_det_model"]);
    mbackend_ = conf["face_det_model"]["backend"];
    facedet_name_ = conf["face_det_model"]["name"];
}

void FaceDetection::DetectFaces(const vector<cv::Mat> &frames,
                                vector<vector<DetectionDecoder::Object>> &faces,
                                vector<vector<FaceLandmarks>> &landmarks){
    const int bn = frames.size();
    const vector<string> request_layers=det_decoder_->ExpectedLayerNames();
    vector<cv::Mat> outputs;
    vector<vector<DetectionDecoder::Object>> sub_faces;
    vector<vector<FaceLandmarks>> sub_landmarks;

    faces.clear();
    landmarks.clear();


    vector<cv::Size> frameSizes;
    for (size_t i=0; i<frames.size(); ++i) {
         frameSizes.push_back(frames[i].size());
    }

    if (mbackend_=="TRT"){

        if (bn%batch_size_ != 0){
            cerr << "Error with bacth size " << endl;
            return;
        }

        RetinaFaceDecoder *retina_decoder = dynamic_cast<RetinaFaceDecoder *>(det_decoder_.get());
        for (size_t i=0; i<bn/batch_size_; ++i){
            vector<cv::Mat> miniBatch(batch_size_);
            vector<cv::Size> miniBatchFrameSizes(batch_size_);
            for (size_t j=0; j<batch_size_; ++j){
                miniBatch[j] = frames[i*batch_size_+j];
                miniBatchFrameSizes[j] = miniBatch[j].size();
            }
            face_det_->Predict(miniBatch, outputs, request_layers);
            if (retina_decoder){
                retina_decoder->BatchDecodeWithLandmarks(outputs, sub_faces, &sub_landmarks, request_layers, miniBatchFrameSizes);
            } else {
                det_decoder_->BatchDecode(outputs, sub_faces, request_layers, miniBatchFrameSizes);
            }

//            if (sub_faces.size()){
//                cout << 0 << endl;
//            }

            faces.insert(faces.end(),sub_faces.begin(), sub_faces.end());
            landmarks.insert(landmarks.end(), sub_landmarks.begin(), sub_landmarks.end());
        }
        return;
    }

    face_det_->Predict(frames, outputs, request_layers);
    det_decoder_->BatchDecode(outputs, faces, request_layers, frameSizes);
}
