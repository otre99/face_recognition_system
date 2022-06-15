#include "face_detection.h"
#include "vision/opencv_predictor.h"
#include <algorithm>

void FaceDetection::Init() {
  const string model_path =
      "/home/rccr/REPOS/face_recognition_system/models/RFB-320.onnx";
  face_detector_ = OpenCVPredictor::Create(model_path, "", "ONNX");
  face_detector_->SetInputParamsNorm(1.0 / 128, {127.5, 127.5, 127.5}, true);
  face_detector_->setExplictInputSize({320, 240});
  det_decoder_.reset(new ULFDDecoder());
  det_decoder_->Init(0.5, 0.25, {320, 240});
}

void FaceDetection::ProcessFrame(const cv::Mat &frame) {
  vector<cv::Mat> outputs;
  face_detector_->Predict(frame, outputs, det_decoder_->ExpectedLayerNames());
  det_decoder_->Decode(outputs, detections, det_decoder_->ExpectedLayerNames(),
                       frame.size());
}
