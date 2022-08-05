#ifndef OPENVINOPREDICTOR_H
#define OPENVINOPREDICTOR_H
#include "predictor.h"
#include "openvino/openvino.hpp"


class OpenVinoPredictor : public Predictor {
public:
  static shared_ptr<Predictor> Create(const string &model_path,
                                      const string &config_path,
                                      const string &framework = "IE");

  void Predict(const cv::Mat &img, vector<cv::Mat> &outputs,
               const vector<string> &output_names = {}) override final;
  void Predict(const vector<cv::Mat> &images, vector<cv::Mat> &outputs,
               const vector<string> &output_names = {}) override final;

private:
  OpenVinoPredictor(){};

private:
  bool Init(const string &model_path, const string &config_path,
            const string &framework);

  ov::CompiledModel compiled_model_;
};

#endif // OPENVINOPREDICTOR_H
