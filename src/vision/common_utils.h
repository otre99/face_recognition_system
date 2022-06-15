#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_
#include <opencv2/opencv.hpp>

struct BBox {
    cv::Rect rect;
    float score;
    int label;
};

void ScaleRect(cv::Rect &r, float scale);
cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame);
inline float Clip(float x) { return std::min(std::max(0.0F, x), 1.0F); }

#endif  // COMMON_UTILS_H_
