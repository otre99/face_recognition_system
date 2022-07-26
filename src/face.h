#ifndef FACE_H
#define FACE_H

#include "common_utils.h"

class Face {
  friend class DrawUtils;

public:
  Face() = default;
  void Init(const cv::Mat &frame, const cv::Rect &det_rect,
            const FaceLandmarks &l, int align_method, long trackingId = -1);

  cv::Mat GetAlignFace(const cv::Mat &frame);
  double GetRoll() const { return roll_; }
  double GetPitch() const { return pitch_; }
  double GetYaw() const { return yaw_; }

  cv::Point2f GetLeftEye() const { return leye_; }
  cv::Point2f GetRightEye() const { return reye_; }
  cv::Point2f GetMouth() const { return mouth_; }

  // private:
  void CalculateLandmarksAbsCoords(const cv::Rect &det_rect,
                                   const FaceLandmarks &l);
  void CalculateFaceOrientation();
  cv::Rect GetAlignRectV1(const cv::Rect &frame_rect);

  // data
  long tracker_id_;
  int align_method_{0};
  cv::Rect det_rect_, align_rect_;
  double yaw_, roll_, pitch_;
  cv::Point2f leye_, reye_;
  cv::Point2f mouth_;
  cv::Point2f nose_;
  cv::Point2f nose_base_;
  cv::Point2f mid_eyes_;

  // utils
  cv::Point2f ToAbsCoords(const cv::Point2f &ori, int w, int h,
                          const cv::Point2f &p);
  double FindAngle(const cv::Point2f &p1, const cv::Point2f &p2);
  double FindSlant(double ln, double lf, double tita);
};

#endif // FACE_H
