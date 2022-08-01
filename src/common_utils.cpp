#include "common_utils.h"
#include <algorithm>
#include <cmath>

void ScaleRect(cv::Rect &r, float scale) {
  const int w = static_cast<int>(r.width * scale + 0.5);
  const int h = static_cast<int>(r.height * scale + 0.5);
  r.x = r.x + (r.width - w) / 2;
  r.y = r.y + (r.height - h) / 2;
  r.width = w;
  r.height = h;
}

void ScaleRect(cv::Rect &r, float sx, float sy) {
  const int w = static_cast<int>(r.width * sx + 0.5);
  const int h = static_cast<int>(r.height * sy + 0.5);
  r.x = r.x + (r.width - w) / 2;
  r.y = r.y + (r.height - h) / 2;
  r.width = w;
  r.height = h;
}

void SquareRect(cv::Rect &r, int mode) {
  int side;
  switch (mode) {
  case 0:
    side = std::max(r.width, r.height);
    break;
  case 1:
    side = std::min(r.width, r.height);
    break;
  default:
    side = (r.width + r.height) / 2;
  }
  r.x = r.x + (r.width - side) / 2;
  r.y = r.y + (r.height - side) / 2;
  r.width = r.height = side;
}

cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame) {
  return rect & cv::Rect{0, 0, frame.cols, frame.rows};
}

double GetScaleFactorForResize(const cv::Size &srcSize,
                               const cv::Size &dstSize) {
  double w = srcSize.width;
  double h = srcSize.height;
  return std::min(dstSize.width / w, dstSize.height / h);
}
