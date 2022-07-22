#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_
#include <opencv2/opencv.hpp>
#include <cmath>

const double PI = std::acos(-1);

struct BBox {
  cv::Rect rect;
  float score;
  int label;
};

struct FaceLandmarks {
  cv::Point2f leye, reye;
  cv::Point2f nose;
  cv::Point2f lmouth, rmouth;
};

inline double GetDist2DPoints(const cv::Point2f &p1, const cv::Point2f &p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt(dx * dx + dy * dy);
}

inline double Radians(double rad){
    return rad*PI/180.0;
}

inline double Degrees(double rad){
    return rad*180.0/PI;
}

void ScaleRect(cv::Rect &r, float scale);
cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame);
inline float Clip(float x) { return std::min(std::max(0.0F, x), 1.0F); }

#endif // COMMON_UTILS_H_
