#ifndef DETECTION_DECODERS_H_
#define DETECTION_DECODERS_H_
#include "common_utils.h"
#include <opencv2/opencv.hpp>
#include <set>

using namespace std;

struct Anchor {
  float cx, cy;
  float w, h;
};

vector<Anchor> GeneratePriors(const int kW, const int kH,
                              const std::vector<float> &kStrides,
                              const std::vector<std::vector<float>> &kMinBoxes,
                              float kCenterVariance = 0.1,
                              const float kSizeVariance = 0.2);

class DetectionDecoder {
public:
  virtual void Decode(const std::vector<cv::Mat> &outRaw,
                      std::vector<BBox> &objects,
                      const vector<string> &onames = {},
                      const cv::Size &img_size = {}) = 0;

  virtual void Init(float scoreTh, float nmsTh,
                    const cv::Size &network_input_size = {}) {
    score_th_ = scoreTh;
    nms_th_ = nmsTh;
    network_input_size_ = network_input_size;
  };

  virtual string GetName() const = 0;

  virtual vector<string> ExpectedLayerNames() const { return {}; }
  void SetLabels(const std::vector<std::string> &lb) { labels_ = lb; }
  std::vector<std::string> GetLabels() const { return labels_; }
  std::string IdToLabel(const int i) const {
    if (labels_.empty())
      return {};
    return labels_[i];
  }

  virtual ~DetectionDecoder() {}

protected:
  std::vector<std::string> labels_;
  float score_th_{0.5};
  float nms_th_{0.25};
  cv::Size network_input_size_{};
};

// ULFDDecoder
class ULFDDecoder : public DetectionDecoder {
public:
  void Decode(const std::vector<cv::Mat> &outRaw, std::vector<BBox> &objects,
              const vector<string> &onames = {},
              const cv::Size &img_size = {}) override final;

  void Init(float scoreTh, float nmsTh,
            const cv::Size &network_input_size = {}) override final;
  string GetName() const override final { return "ULFD"; }

  vector<string> ExpectedLayerNames() const override {
    return {"boxes", "scores"};
  }

private:
  vector<Anchor> priors_;
};

// RetinaFaceDecoder
class RetinaFaceDecoder : public DetectionDecoder {
public:
  void Init(float scoreTh, float nmsTh,
            const cv::Size &network_input_size = {}) override;
  string GetName() const override final { return "RETINAFACE"; }

  void Decode(const vector<cv::Mat> &outRaw, vector<BBox> &objects,
              const vector<string> &onames = {},
              const cv::Size &img_size = {}) override;

  vector<string> ExpectedLayerNames() const override {
    return {"boxes", "scores", "landmarks"};
  }

  vector<FaceLandmarks> landmarks_;

private:
  vector<Anchor> priors_;
};

#endif // DETECTION_DECODERS_H_
