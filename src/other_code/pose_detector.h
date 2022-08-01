/*
 * Copyright 2016 <Admobilize>
 * All rights reserved.
 */

#ifndef ADM_DETECTOR_POSE_DETECTOR_H_
#define ADM_DETECTOR_POSE_DETECTOR_H_

#include <vector>

#include "opencv2/opencv.hpp"


struct FaceFeatures {
  cv::Point2f face_center;
  cv::Point2f left_eye;
  cv::Point2f right_eye;
  cv::Point2f nose;
  cv::Point2f mouth;
  cv::Point2f nose_base;
  cv::Point2f mid_eyes;
};

struct FaceGeometry {
  double left_eye_nose_distance{0};
  double right_eye_nose_distance{0};
  double left_eye_right_eye_distance{0};
  double nose_mouth_distance{0};
  double eye_mouth_distance{0};
};

struct Pose {
  double pitch{0}, yaw{0}, roll{0};
  double slant{0};
  double kpitch{0}, kyaw{0};
  double kpitch_pre{0}, kyaw_pre{0};
};

// This class computes face Pose and Gaze from face landmarks
// using Gee & Cipolla method.
class PoseDetector {
 public:
  explicit PoseDetector(bool use_kalman = false, float r_m = 0.5f,
                        float r_n = 0.5f, float r_e = 0.91f,
                        float deg_yaw_gaze_th = 20.0f,
                        float deg_pitch_gaze_th = 15.0f);
  void Init(bool use_kalman = false, float r_m = 0.5, float r_n = 0.5,
            float r_e = 0.91, float deg_yaw_gaze_th = 20.0f,
            float deg_pitch_gaze_th = 15.0f);

  Pose Detect(const std::vector<cv::Point2f> &face_landmarks);
  void Draw(cv::Mat *img, const cv::Scalar &colour = cv::Scalar(255)) const;
  void DrawFiltered(cv::Mat *img,
                    const cv::Scalar &colour = cv::Scalar(255)) const;
  bool GetFilteredGaze() const;
  bool GetGaze() const;
  const FaceFeatures &GetFeatures() const;
  const FaceGeometry &GetGeometry() const;

 private:
  void InitKalmanFilter();
  float FindDistance(cv::Point pt1, cv::Point pt2);
  float FindDistance2D32f(cv::Point2f pt1, cv::Point2f pt2);
  float FindAngle(cv::Point2f pt1, cv::Point2f pt2);
  double FindSlant(float ln, float lf, float tita);
  void KalmanPredict();
  void Calculate(const std::vector<cv::Point2f> &face_landmarks);

  Pose pose_;
  FaceGeometry geometry_;
  FaceFeatures features_;

  // Variables for 3D model
  float r_m_{0.5f};
  float r_n_{0.5f};
  float r_e_{0.91f};

  // Gaze angles in degrees
  float yaw_gaze_th_{20.0};
  float pitch_gaze_th_{15.0};

  // Kalman Filter Configuration
  bool use_kalman_{true};
  bool is_kalman_initialized_{false};
  cv::RNG rng;
  cv::KalmanFilter kalman_;
  cv::Mat x_k_;  // State vector, n-dim
  cv::Mat w_k_;  // Process noise, n-dim
  cv::Mat z_k_;  // Measurement vector, m-dim
};


#endif  // ADM_DETECTOR_POSE_DETECTOR_H_
