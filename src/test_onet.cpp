#include <iostream>
#include <opencv2/opencv.hpp>
#include "common_utils.h"

const int INPUT_DATA_WIDTH = 48;
const int INPUT_DATA_HEIGHT = 48;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;



inline cv::Mat cropImage(const cv::Mat &img, cv::Rect r) {
  cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
  int dx = std::abs(std::min(0, r.x));
  if (dx > 0) {
    r.x = 0;
  }
  r.width -= dx;
  int dy = std::abs(std::min(0, r.y));
  if (dy > 0) {
    r.y = 0;
  }
  r.height -= dy;
  int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
  r.width -= dw;
  int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
  r.height -= dh;
  if (r.width > 0 && r.height > 0) {
    img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
  }
  return m;
}

int main() {

  auto _net = cv::dnn::readNetFromONNX("/test_storage/models/OnetD.onnx");
  cv::Size windowSize = cv::Size(INPUT_DATA_WIDTH, INPUT_DATA_HEIGHT);
  auto image = cv::imread("/test_storage/image_samples/Person01_crop.jpg");
  cv::Rect rect(0,0,image.cols, image.rows);
  ScaleRect(rect,0.8);
  //SquareRect(rect);
  float w = rect.width;
  float h = rect.height;
  cv::Mat roi = cropImage(image, rect);
  cv::resize(roi, roi, windowSize, 0, 0, cv::INTER_AREA);


  auto blobInput =
      cv::dnn::blobFromImage(roi, IMG_INV_STDDEV, cv::Size(),
                             cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), true);

  _net.setInput(blobInput); //, "data");

  const std::vector<cv::String> outBlobNames{"scores", "landmarks", "boxes"};
  std::vector<cv::Mat> outputBlobs;

  _net.forward(outputBlobs, outBlobNames);

  cv::Mat regressionsBlob = outputBlobs[0];
  cv::Mat landMarkBlob = outputBlobs[1];
  cv::Mat scoresBlob = outputBlobs[2];

  const float *scores_data = (float *)scoresBlob.data;
  const float *landmark_data = (float *)landMarkBlob.data;
  const float *reg_data = (float *)regressionsBlob.data;

  std::vector<cv::Point2f> pts;
  const int NUM_PTS = 5;
  for (int p = 0; p < NUM_PTS; ++p) {
    float x = rect.x + landmark_data[0 + p] * w - 1;
    float y = rect.y + landmark_data[NUM_PTS + p] * h - 1;
    pts.emplace_back(x, y);
  }

  cv::rectangle(image, rect, {0,0,0}, 2);
  for (auto p : pts){
      cv::circle(image, p, 3, {0,0,255}, -1);
  }
  cv::imshow("", image);
  cv::waitKey(-1);
  return 0;
}
