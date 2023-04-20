#include "draw_utils.h"

void DrawUtils::Init(const cv::Size &frameSize) {
  label_size_ = cv::getTextSize(" Y: -360 ", cv::FONT_HERSHEY_SIMPLEX, 0.4, 1,
                                &baseline_);
  title_size_ = cv::getTextSize(" FPS: 9999 ", cv::FONT_HERSHEY_SIMPLEX, 0.4, 1,
                                &baseline_);
}

void DrawUtils::SetLineColor(cv::Scalar &lcolor){
  line_color_ = lcolor;
}

void DrawUtils::DrawFace(cv::Mat &frame, const Face &face) {
  const cv::Rect &box = face.det_rect_;
  cv::rectangle(frame, box, line_color_, line_thickness_, cv::LINE_8);
  int x1 = box.x;
  int y1 = box.y;
  int x2 = x1 + box.width;
  int xl = x2 + line_thickness_;
  int lh = label_size_.height + baseline_;
  int yl = y1 - label_size_.height;

  cv::rectangle(frame, {xl, yl, label_size_.width, 4 * lh}, {255, 255, 255},
                -1);
  cv::putText(frame, " ID: " + std::to_string(face.tracker_id_), {x2, y1},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);
  cv::putText(frame, " R : " + std::to_string(int(face.roll_)), {x2, y1 + lh},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);
  cv::putText(frame, " P : " + std::to_string(int(face.pitch_)),
              {x2, y1 + 2 * lh}, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);
  cv::putText(frame, " Y : " + std::to_string(int(face.yaw_)),
              {x2, y1 + 3 * lh}, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);

  cv::circle(frame, face.leye_, 3, landmarks_color_, -1, cv::FILLED);
  cv::circle(frame, face.reye_, 3, landmarks_color_, -1, cv::FILLED);
  cv::circle(frame, face.mouth_, 3, landmarks_color_, -1, cv::FILLED);
}

void DrawUtils::DrawTrackedObj(cv::Mat &frame, const TrackedObject &obj,
                               const string &msg) {
  const cv::Rect &box = obj.rect;
  cv::rectangle(frame, box, line_color_, line_thickness_, cv::LINE_8);
  int x1 = box.x;
  int y1 = box.y;
  int x2 = x1 + box.width;
  int xl = x2 + line_thickness_;
  int lh = label_size_.height + baseline_;
  int yl = y1 - label_size_.height;

  const int n = msg.empty() ? 1 : 2;
  cv::rectangle(frame, {xl, yl, label_size_.width, n * lh}, {255, 255, 255},
                -1);
  cv::putText(frame, " ID: " + std::to_string(obj.id), {x2, y1},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);
  if (n == 2) {
    cv::putText(frame, " " + msg, {x2, y1 + lh}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                1);
  }
}

void DrawUtils::DrawFPS(cv::Mat &frame, int fps) {
  cv::rectangle(frame,
                {0, 0, title_size_.width, title_size_.height + baseline_},
                {255, 255, 255}, -1);
  cv::putText(frame, "FPS: " + std::to_string(fps), {1, title_size_.height},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, 1);
}
