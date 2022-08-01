#include "opencv_predictor.h"
#include "common_utils.h"

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
  cv::Mat blob;
  if (keep_aspect_ratio_) {
    const double s = GetScaleFactorForResize(img.size(), input_size_);
    blob.create(input_size_, img.type());
    blob.setTo(cv::Scalar::all(0));
    cv::Size dstSize = cv::Size(round(s * img.cols), round(s * img.rows));
    cv::resize(img, blob({0, 0, dstSize.width, dstSize.height}), dstSize);
    blob = cv::dnn::blobFromImage(blob, scale_, input_size_, mean_, swap_ch_,
                                  false, CV_32F);
  } else {
    blob = cv::dnn::blobFromImage(img, scale_, input_size_, mean_, swap_ch_,
                                  false, CV_32F);
  }

  net_.setInput(blob);
  if (output_names.empty()) {
    net_.forward(outputs);
  } else {
    net_.forward(outputs, output_names);
  }
}

void OpenCVPredictor::Predict(const vector<cv::Mat> &images,
                              vector<cv::Mat> &outputs,
                              const vector<string> &output_names) {
  cv::Mat blob;
  if (keep_aspect_ratio_) {
    vector<cv::Mat> tmp(images.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
      tmp[i].create(input_size_, images[i].type());
      tmp[i].setTo(cv::Scalar::all(0.0));
      const double s = GetScaleFactorForResize(images[i].size(), input_size_);
      cv::Size dstSize =
          cv::Size(round(s * images[i].cols), round(s * images[i].rows));
      cv::resize(images[i], tmp[i]({0, 0, dstSize.width, dstSize.height}),
                 dstSize);
      tmp[i] = cv::dnn::blobFromImage(tmp[i], scale_, input_size_, mean_,
                                      swap_ch_, false, CV_32F);
    }
  } else {
    blob = cv::dnn::blobFromImages(images, scale_, input_size_, mean_, swap_ch_,
                                   false, CV_32F);
  }

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
