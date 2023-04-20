#include "image_saver.h"
#include "opencv2/imgcodecs.hpp"
#include <chrono>

ImageSaver::ImageSaver() {
  exit_ = false;
  thread_ = std::thread(&ImageSaver::Loop, this);
}

ImageSaver::~ImageSaver() {
  Stop();
  thread_.join();
}

void ImageSaver::Stop() { exit_ = true; }

void ImageSaver::EnqueueImage(const cv::Mat &image, const std::string &name) {
  Data d;
  d.img_ptr = new cv::Mat();
  image.copyTo(*(d.img_ptr));
  d.name = name;
  std::unique_lock<std::mutex> lock(mutex_);
  data_.push(d);
}

void ImageSaver::Loop() {
  using namespace std::chrono_literals;
  for (;;) {
    if (exit_)
      break;
    mutex_.lock();
    bool pending = !data_.empty();
    if (pending) {
      Data dd = data_.top();
      data_.pop();
      mutex_.unlock();
      SaveImage(dd);
      continue;
    } else {
      mutex_.unlock();
    }

    std::this_thread::sleep_for(500ms);
  }

  // saving pending images
  while (!data_.empty()) {
    Data dd = data_.top();
    data_.pop();
    mutex_.unlock();
    SaveImage(dd);
  }
}

void ImageSaver::SaveImage(const Data &d) {
  cv::imwrite(d.name, *(d.img_ptr));
  delete d.img_ptr;
}
