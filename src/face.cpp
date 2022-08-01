#include "face.h"
#include <cmath>

using namespace std;

const cv::Point2f Face::LPTS[5] = {
    {30.2946 / 96, 51.6963 / 112}, // left eye
    {65.5318 / 96, 51.5014 / 112}, // right eye
    {48.0252 / 96, 71.7366 / 112}, // nose tip
    {33.5493 / 96, 92.3655 / 112}, // left lip corner
    {62.7299 / 96, 92.2041 / 112}  // right lip corner
};

void Face::Init(const cv::Mat &frame, const cv::Rect &det_rect,
                const FaceLandmarks &l, int align_method, long trackingId) {
  CalculateLandmarksAbsCoords(det_rect, l);
  const cv::Rect frame_rect(0,0,frame.cols, frame.rows);
  align_method_ = align_method;
  switch (align_method_) {
  case 0:
    align_rect_ = det_rect&frame_rect;
    break;
  case 1:
    align_rect_ = GetAlignRectV1({0, 0, frame.cols, frame.rows});
    break;
  default:
      align_rect_ = det_rect;
      ScaleRect(align_rect_, 2);
      align_rect_ &=frame_rect;
    break;
  }
  CalculateFaceOrientation();
  det_rect_ = det_rect&frame_rect;
  tracker_id_ = trackingId;
}

cv::Mat Face::GetAlignFace(const cv::Mat &frame) {
  switch (align_method_) {
  case 0:
  case 1:
    return frame(align_rect_);
    break;
  case 2:
      return GetFaceAffineTransfor(frame);
      break;
  default:
    return frame(align_rect_);
    break;
  }
}

void Face::CalculateLandmarksAbsCoords(const cv::Rect &det_rect,
                                       const FaceLandmarks &l) {
  const double Rm = 0.5;
  auto tl = det_rect.tl();
  if (l.relative_coords) {
    leye_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.leye);
    reye_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.leye);
    nose_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.nose);
    lmouth_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.lmouth);
    rmouth_ = ToAbsCoords(tl, det_rect.width, det_rect.height, l.rmouth);
    mouth_ = 0.5 * (l.lmouth + l.rmouth);
  } else {
    leye_ = l.leye;
    reye_ = l.reye;
    nose_ = l.nose;
    lmouth_ = l.lmouth;
    rmouth_ = l.rmouth;
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

    cv::Point2f mid_eye = 0.5*(leye_+reye_);
    float wside = GetDist2DPoints(leye_, reye_)*2.7;
    cv::Rect rect;
    rect.x = mid_eye.x - wside/2;
    float x2 = mid_eye.x + wside/2;
    float hside = GetDist2DPoints(mid_eye, mouth_);
    rect.y = mid_eye.y - 1.27*hside;
    float y2 = mouth_.y + 0.48*hside;

    rect.width = x2-rect.x;
    rect.height = y2-rect.y;
    return frame_rect & rect;
}

cv::Mat Face::GetFaceAffineTransfor(const cv::Mat &frame) {
  const cv::Point2f tl = align_rect_.tl();
  const vector<cv::Point2f> srcPts = {leye_ - tl,
                                      reye_ - tl,
                                      mouth_-tl
                                      };
  const int side = (align_rect_.width + align_rect_.height) / 2;
  const vector<cv::Point2f> dstPts = {
      LPTS[0] * side,
      LPTS[1] * side,
      0.5*(LPTS[3] + LPTS[4])* side,
      };
  const cv::Mat warp_mat = cv::getAffineTransform(srcPts, dstPts);
  cv::Mat result({side, side});
  cv::warpAffine(frame(align_rect_), result, warp_mat, {side, side});
  return result;
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
