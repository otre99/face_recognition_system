#include "opencv_predictor.h"
#include <stdexcept>

shared_ptr<Predictor> OpenCVPredictor::Create(const string &model_path,
                                              const string &config_path,
                                              const string &framework) {
  auto impl = new OpenCVPredictor();
  if (impl->Init(model_path, config_path, framework)) {
    return shared_ptr<Predictor>{impl};
  }
  delete impl;
  return {};
}

void OpenCVPredictor::Predict(const cv::Mat &img, vector<cv::Mat> &outputs,
                              const vector<string> &output_names) {
  vector<cv::Mat> images = {img};
  Predict(images, outputs, output_names);
}

void OpenCVPredictor::Predict(const vector<cv::Mat> &images,
                              vector<cv::Mat> &outputs,
                              const vector<string> &output_names) {

  auto blob = this->CreateInputBlob(images);
  net_.setInput(blob);
  if (output_names.empty()) {
    net_.forward(outputs);
  } else {
    net_.forward(outputs, output_names);
  }
}

bool OpenCVPredictor::Init(const string &model_path, const string &config_path,
                           const string &framework) {
  net_ = cv::dnn::readNet(model_path, config_path, framework);
  if (net_.empty()) {
    cerr << "Error loading model: " << endl;
    cerr << "   ModelWeights : " << model_path << endl;
    cerr << "   ModelConfig  : " << config_path << endl;
    return false;
  }
  return true;
}
