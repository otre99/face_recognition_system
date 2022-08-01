/* Copyright 2016 <Admobilize>, All rights reserved. */

#include "pose_detector.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>

#include <iostream>

#include "opencv2/opencv.hpp"

using cv::KalmanFilter;
using cv::Mat;
using cv::Point;
using cv::Scalar;


PoseDetector::PoseDetector(bool use_kalman, float r_m, float r_n, float r_e,
                           float deg_yaw_gaze_th, float deg_pitch_gaze_th) {
  Init(use_kalman, r_m, r_n, r_e, deg_yaw_gaze_th, deg_pitch_gaze_th);
}

void PoseDetector::Init(bool use_kalman, float r_m, float r_n, float r_e,
                        float deg_yaw_gaze_th, float deg_pitch_gaze_th) {
  use_kalman_ = use_kalman;
  r_m_ = r_m;
  r_n_ = r_n;
  r_e_ = r_e;
  pose_.yaw = 0.;
  pose_.pitch = 0.;
  pose_.roll = 0.;
  is_kalman_initialized_ = false;
  yaw_gaze_th_ = deg_yaw_gaze_th;
  pitch_gaze_th_ = deg_pitch_gaze_th;
}

void PoseDetector::KalmanPredict() {
  if (is_kalman_initialized_) {
    const Mat y_k = kalman_.predict();
    x_k_.at<float>(0, 0) = pose_.pitch;
    x_k_.at<float>(1, 0) = (pose_.pitch - pose_.kpitch_pre);
    x_k_.at<float>(2, 0) = pose_.yaw;
    x_k_.at<float>(3, 0) = (pose_.yaw - pose_.kyaw_pre);
    kalman_.correct(x_k_);
    randn(w_k_, 0, sqrt(kalman_.processNoiseCov.data[0]));
    cv::gemm(kalman_.transitionMatrix, x_k_, 1.0, w_k_, 1.0, x_k_, 0);
    pose_.kyaw = y_k.at<float>(2, 0);
    pose_.kpitch = y_k.at<float>(0, 0);
  } else {
    InitKalmanFilter();
  }
}

void PoseDetector::Calculate(const std::vector<cv::Point2f> &face_landmarks) {
  features_.face_center = face_landmarks[0];
  features_.left_eye = face_landmarks[1];
  features_.right_eye = face_landmarks[2];
  features_.nose = face_landmarks[3];
  features_.mouth = face_landmarks[4];
  features_.mid_eyes.x = (features_.left_eye.x + features_.right_eye.x) / 2;
  features_.mid_eyes.y = (features_.left_eye.y + features_.right_eye.y) / 2;
  features_.nose_base.x =
      features_.mouth.x + (features_.mid_eyes.x - features_.mouth.x) * (r_m_);
  features_.nose_base.y =
      features_.mouth.y - (features_.mouth.y - features_.mid_eyes.y) * (r_m_);
  geometry_.left_eye_nose_distance =
      FindDistance2D32f(features_.nose, features_.left_eye);
  geometry_.right_eye_nose_distance =
      FindDistance2D32f(features_.nose, features_.right_eye);
  geometry_.left_eye_right_eye_distance =
      FindDistance2D32f(features_.left_eye, features_.right_eye);
  geometry_.nose_mouth_distance =
      FindDistance2D32f(features_.nose, features_.mouth);

  float normal_length = FindDistance2D32f(features_.nose_base, features_.nose);
  geometry_.eye_mouth_distance =
      FindDistance2D32f(features_.mid_eyes, features_.mouth);

  double len = cv::norm(features_.left_eye - features_.right_eye);

  pose_.roll =
      cv::Point2f(0, 1).dot(features_.left_eye - features_.right_eye) / len;
  pose_.roll = pose_.roll * 180 / M_PI;

  // symm angle - angle between the symmetry axis and the 'x' axis
  // tilt angle - angle between normal in image and 'x' axis
  // tita angle - angle betweenthe symmetry axis and the image normal
  // slant angle - angle between the face normal and the image normal
  float symm = FindAngle(features_.nose_base, features_.mid_eyes);
  float tilt = FindAngle(features_.nose_base, features_.nose);
  float tita = (abs(tilt - symm)) * (M_PI / 180);
  pose_.slant = FindSlant(normal_length, geometry_.eye_mouth_distance, tita);

  cv::Point3d normal;
  normal.x = (sin(pose_.slant)) * (cos((360 - tilt) * (M_PI / 180)));
  normal.y = (sin(pose_.slant)) * (sin((360 - tilt) * (M_PI / 180)));
  normal.z = -cos(pose_.slant);

  // find pitch and yaw
  float temp = 2;
  pose_.kpitch_pre = pose_.pitch;
  if (normal.x != 0 || normal.z != 0 || normal.y != 0)
    temp =
        sqrt((normal.x * normal.x + normal.z * normal.z) /
             (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z));
  pose_.pitch = (temp > -1 && temp < 1) ? acos(temp) : 0;
  pose_.pitch = (features_.nose.y - features_.nose_base.y < 0) ? -pose_.pitch
                                                               : pose_.pitch;
  pose_.pitch = pose_.pitch * 180 / M_PI;

  // pitch[frame_number] = P.pitch*(180/pi);
  temp = 2;
  pose_.kyaw_pre = pose_.yaw;
  if (normal.x != 0 || normal.z != 0)
    temp = (std::abs(normal.z)) /
           (sqrt(normal.x * normal.x + normal.z * normal.z));
  pose_.yaw = (temp > -1.0f && temp < 1) ? acos(temp) : 0;
  pose_.yaw =
      (features_.nose.x - features_.nose_base.x < 0) ? -pose_.yaw : pose_.yaw;
  pose_.yaw = pose_.yaw * 180 / M_PI;
}

Pose PoseDetector::Detect(const std::vector<cv::Point2f> &face_landmarks) {
  Calculate(face_landmarks);
  // Kalman Filter Code
  if (use_kalman_) {
    KalmanPredict();
  }
  return pose_;
}

void PoseDetector::InitKalmanFilter() {
  kalman_ = KalmanFilter(4, 4, 0);
  x_k_ = Mat(4, 1, CV_32FC1);
  w_k_ = Mat(4, 1, CV_32FC1);
  z_k_ = Mat(4, 1, CV_32FC1);
  cv::randn(x_k_, 0, 0.1);
  cv::randn(x_k_, 0, 0.1);
  z_k_ = Scalar::all(0);
  const float F[] = {1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
  memcpy(kalman_.transitionMatrix.data, F, sizeof(F));
  cv::setIdentity(kalman_.measurementMatrix, cv::Scalar::all(1));
  cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(20 * 0.0001));
  cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(80 * 1));
  cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1));

  kalman_.statePre.at<float>(0) = pose_.pitch;
  kalman_.statePre.at<float>(1) = 0;
  kalman_.statePre.at<float>(2) = pose_.yaw;
  kalman_.statePre.at<float>(3) = 0;

  kalman_.statePost.at<float>(0) = pose_.pitch;
  kalman_.statePost.at<float>(1) = 0;
  kalman_.statePost.at<float>(2) = pose_.yaw;
  kalman_.statePost.at<float>(3) = 0;

  is_kalman_initialized_ = true;
}

void PoseDetector::Draw(Mat *img, const Scalar &colour) const {
  int x = features_.nose_base.x +
          cvRound(geometry_.eye_mouth_distance * 2 * (tan(pose_.yaw)));
  int y = features_.nose_base.y +
          cvRound(geometry_.eye_mouth_distance * 2 * (tan(pose_.pitch)));
  line(*img, Point(features_.nose_base.x, features_.nose_base.y), Point(x, y),
       colour, 2, 4, 0);
}

void PoseDetector::DrawFiltered(Mat *img, const Scalar &colour) const {
  int x = ((features_.nose_base.x +
            cvRound(geometry_.eye_mouth_distance * 2 *
                    (tan(static_cast<double>(pose_.kyaw))))));
  int y = ((features_.nose_base.y +
            cvRound(geometry_.eye_mouth_distance * 2 *
                    (tan(static_cast<double>(pose_.kpitch))))));
  line(*img, Point(features_.nose_base.x, features_.nose_base.y), Point(x, y),
       colour, 2, 4, 0);
}

float PoseDetector::FindDistance(cv::Point pt1, cv::Point pt2) {
  int x, y;
  float z;
  x = pt1.x - pt2.x;
  y = pt1.y - pt2.y;
  z = (x * x) + (y * y);
  return sqrt(z);
}

float PoseDetector::FindDistance2D32f(cv::Point2f pt1, cv::Point2f pt2) {
  float x, y, z;
  x = pt1.x - pt2.x;
  y = pt1.y - pt2.y;
  z = (x * x) + (y * y);
  return sqrt(z);
}

float PoseDetector::FindAngle(cv::Point2f pt1, cv::Point2f pt2) {
  float angle;
  angle = cv::fastAtan2(pt2.y - pt1.y, pt2.x - pt1.x);
  return 360 - angle;
}

double PoseDetector::FindSlant(float ln, float lf, float tita) {
  float dz = 0;
  double slant;
  float m1 = static_cast<float>(ln * ln) / (lf * lf);
  float m2 = (cos(tita)) * (cos(tita));
  float Rn = r_n_;

  if (m2 == 1) {
    dz = sqrt((Rn * Rn) / (m1 + (Rn * Rn)));
  }
  if (m2 >= 0 && m2 < 1) {
    dz = sqrt(((Rn * Rn) - m1 - 2 * m2 * (Rn * Rn) +
               sqrt(((m1 - (Rn * Rn)) * (m1 - (Rn * Rn))) +
                    4 * m1 * m2 * (Rn * Rn))) /
              (2 * (1 - m2) * (Rn * Rn)));
  }
  slant = acos(dz);
  return slant;
}

bool PoseDetector::GetFilteredGaze() const {
  bool xangletest = fabs(pose_.kyaw) <= yaw_gaze_th_;
  bool yangletest = fabs(pose_.kpitch) <= pitch_gaze_th_;
  return xangletest && yangletest;
}

bool PoseDetector::GetGaze() const {
  bool xangletest = fabs(pose_.yaw) <= yaw_gaze_th_;
  bool yangletest = fabs(pose_.pitch) <= pitch_gaze_th_;
  return xangletest && yangletest;
}

const FaceFeatures &PoseDetector::GetFeatures() const { return features_; }

const FaceGeometry &PoseDetector::GetGeometry() const { return geometry_; }

