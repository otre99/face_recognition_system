#ifndef DRAW_UTILS_H_
#define DRAW_UTILS_H_
#include "face.h"
#include "tracker.h"

class DrawUtils {
public:
  void DrawFace(cv::Mat &frame, const Face &face);
  void DrawTrackedObj(cv::Mat &frame, const TrackedObject &obj,
                      const string &msg = {});
  void DrawFPS(cv::Mat &frame, int fps);
  void Init(const cv::Size &frameSize);
  void SetLineColor(cv::Scalar &lcolor);

private:
  cv::Scalar line_color_{0, 0, 0};
  cv::Scalar landmarks_color_{0, 0, 255};
  int line_thickness_{2};
  cv::Size label_size_;
  cv::Size title_size_;

  int baseline_;
};

#endif // DRAW_UTILS_H_
