#include "common_utils.h"

void ScaleRect(cv::Rect &r, float scale) {
  const int w = static_cast<int>(r.width * scale + 0.5);
  const int h = static_cast<int>(r.height * scale + 0.5);
  r.x = r.x + (r.width - w) / 2;
  r.y = r.y + (r.height - h) / 2;
  r.width = w;
  r.height = h;
}

cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame) {
  return rect & cv::Rect{0, 0, frame.cols, frame.rows};
}

