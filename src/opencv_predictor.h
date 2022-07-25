#ifndef OPENCV_PREDICTOR_H_
#define OPENCV_PREDICTOR_H_

#include "predictor.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
class OpenCVPredictor : public Predictor {
public:
  static shared_ptr<Predictor> Create(const string &model_path,
                                      const string &config_path,
                                      const string &framework = "ONNX");

  void Predict(const cv::Mat &img, vector<cv::Mat> &outputs,
               const vector<string> &output_names = {}) override final;
  void Predict(const vector<cv::Mat> &images, vector<vector<cv::Mat>> &outputs,
               const vector<string> &output_names = {}) override final;

private:
  OpenCVPredictor(){};

private:
  bool Init(const string &model_path, const string &config_path,
            const string &framework);
  cv::dnn::Net net_{};
};

#endif
