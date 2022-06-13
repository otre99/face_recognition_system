#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "cvision/predictor.h"
#include "cvision/detection_decoder.h"
#include "./nlohmann/json.hpp"
#include "cvision/detection_decoder.h"
#include <memory>
#include "cvision/retinafacedecoder.h"

using namespace std;

class FaceDetection
{
public:
    void Init(const nlohmann::json &conf);
    void DetectFaces(const vector<cv::Mat> &frames,
                     vector<vector<DetectionDecoder::Object> > &faces,
                     vector<vector<FaceLandmarks> > &landmarks);
private:
    int batch_size_{-1};
    string mbackend_{};
    string facedet_name_{};
    std::shared_ptr<Predictor> face_det_{};
    std::shared_ptr<DetectionDecoder> det_decoder_{};
};

#endif // FACEDETECTION_H
