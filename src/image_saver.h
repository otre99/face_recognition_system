#ifndef IMAGESAVER_H_
#define IMAGESAVER_H_
#include <mutex>
#include <opencv2/core.hpp>
#include <stack>
#include <thread>

class ImageSaver {
public:
  struct Data {
    std::string name;
    cv::Mat *img_ptr;
  };
  ImageSaver();
  ~ImageSaver();
  void Stop();
  void EnqueueImage(const cv::Mat &image, const std::string &name);

private:
  void Loop();
  void SaveImage(const Data &d);
  std::thread thread_;
  std::mutex mutex_;
  std::stack<Data> data_;
  bool exit_{false};
};

#endif // IMAGESAVER_H_
