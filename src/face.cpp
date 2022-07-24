#include "face.h"
#include <cmath>

void Face::Init(const cv::Mat &frame, const cv::Rect &det_rect,
                const FaceLandmarks &l, long trackingId) {
  CalculateLandmarksAbsCoords(det_rect, l);
  align_rect_ = GetAlignRectV1({0, 0, frame.cols, frame.rows});
  CalculateFaceOrientation();
  det_rect_ = det_rect;
  tracker_id_ = trackingId;
}

void Face::CalculateLandmarksAbsCoords(const cv::Rect &det_rect,
                                       const FaceLandmarks &l) {
  const double Rm = 0.5;
  auto tl = det_rect.tl();
  if (l.relative_coords){
      leye_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.leye);
      reye_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.leye);
      nose_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.nose);
      mouth_ = ToAbsCoords(tl, det_rect.width, det_rect.height,
                           0.5 * (l.lmouth + l.rmouth));
  } else {
      leye_ = l.leye;
      reye_ = l.reye;
      nose_ = l.nose;
      mouth_ = 0.5 * (l.lmouth + l.rmouth);
  }

  // secondary points
  mid_eyes_ = 0.5 * (leye_ + reye_);
  nose_base_ = mouth_ + (mid_eyes_ - mouth_) * Rm;
}

void Face::CalculateFaceOrientation() {
  const float Rm = 0.5f;
  const double len = GetDist2DPoints(leye_, reye_);
  // roll
  roll_ = Degrees(cv::Point2f(0, 1).ddot(leye_ - reye_) / len);
  const double nlen = GetDist2DPoints(nose_base_, nose_);
  const double eye_mouth_dist = GetDist2DPoints(mid_eyes_, mouth_);
  const double symm = FindAngle(nose_base_, mid_eyes_);
  const double tilt = FindAngle(nose_base_, nose_);
  const double tita = abs(tilt - symm);

  const double slant = FindSlant(nlen, eye_mouth_dist, tita);
  const double nx = sin(slant) * cos(2 * PI - tilt);
  const double ny = sin(slant) * sin(2 * PI - tilt);
  const double nz = -cos(slant);

  double temp = 2;
  if (nx != 0 || nz != 0 || ny != 0) {
    temp = sqrt((nx * nx + nz * nz) / (nx * nx + ny * ny + nz * nz));
  }
  // pitch
  pitch_ = (temp > -1 && temp < 1) ? acos(temp) : 0;
  pitch_ = (nose_.y - nose_base_.y < 0) ? -pitch_ : pitch_;
  pitch_ = Degrees(pitch_);

  temp = 2;
  if (nx != 0 || nz != 0) {
    temp = fabs(nz) / sqrt(nx * nx + nz * nz);
  }
  // yaw
  yaw_ = (temp > -1.0f && temp < 1) ? acos(temp) : 0;
  yaw_ = (nose_.x - nose_base_.x < 0) ? -yaw_ : yaw_;
  yaw_ = Degrees(yaw_);
}

cv::Rect Face::GetAlignRectV1(const cv::Rect &frame_rect) {

  const double eye_dist = GetDist2DPoints(leye_, reye_);
  const double eye_mouth_dist = GetDist2DPoints(mid_eyes_, mouth_);
  const double d1 = GetDist2DPoints(nose_base_, mouth_);

  float x1 = nose_base_.x - eye_dist * 1.1;
  float x2 = nose_base_.x + eye_dist * 1.1;
  float y1 = nose_base_.y - eye_mouth_dist * 1.3;
  float y2 = nose_base_.y + d1 * 1.8;

  cv::Rect rect;
  rect.x = x1;
  rect.y = y1;
  rect.width = x2 - x1 + 1;
  rect.height = y2 - y1 + 1;
  return rect & frame_rect;
}

cv::Point2f Face::ToAbsCoords(const cv::Point2f &ori, int w, int h,
                              const cv::Point2f &p) {
  return ori + cv::Point2f{p.x * w, p.y * h};
}

double Face::FindAngle(const cv::Point2f &p1, const cv::Point2f &p2) {
  return 2 * PI - std::atan2(p2.y - p1.y, p2.x - p1.x);
}

double Face::FindSlant(double ln, double lf, double tita) {
  float dz = 0;
  double slant;
  const double m1 = (ln * ln) / (lf * lf);
  const double m2 = cos(tita) * cos(tita);
  const double Rn = 0.5;
  const double Rn2 = Rn * Rn;

  if (m2 == 1) {
    dz = sqrt(Rn2 / (m1 + Rn2));
  }
  if (m2 >= 0 && m2 < 1) {
    dz = sqrt((Rn2 - m1 - 2 * m2 * Rn2 +
               sqrt(((m1 - Rn2) * (m1 - Rn2)) + 4 * m1 * m2 * Rn2)) /
              (2 * (1 - m2) * Rn2));
  }
  return acos(dz);
}
