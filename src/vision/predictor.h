#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <functional>
#include <opencv2/opencv.hpp>

using namespace std;

class Predictor {
public:
  virtual void Predict(const cv::Mat &img, std::vector<cv::Mat> &outputs,
                       const std::vector<std::string> &output_names = {}) = 0;
  virtual void Predict(const std::vector<cv::Mat> &img,
                       std::vector<cv::Mat> &outputs,
                       const std::vector<std::string> &output_names = {}) = 0;

  void setExplictInputSize(const cv::Size &input_size) {
    input_size_ = input_size;
  }

  void SetInputParamsNorm(float scale, const cv::Scalar &mean, bool swap) {
    scale_ = scale;
    mean_ = mean;
    swap_ch_ = swap;
  }

  virtual ~Predictor(){};

protected:
  Predictor() {}
  std::function<void(cv::Mat, float *)> normalization_fnt_{};
  cv::Size input_size_{};
  int input_ch_ = 3;
  cv::Scalar mean_{};
  double scale_ = 1.0;
  bool swap_ch_ = false;
};

#endif
