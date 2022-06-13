#ifndef DETECTION_DECODER_H
#define DETECTION_DECODER_H
#include <opencv2/opencv.hpp>
#include <set>

using namespace std;

class DetectionDecoder {
public:
  struct Object {
    cv::Rect2f rect;
    int label;
    float prob;
  };

  struct Anchor {
    float cx, cy;
    float w, h;
  };
  static vector<Anchor> GeneratePriors(const int kW, const int kH, const std::vector<float> &kStrides,
                                       const std::vector<std::vector<float>> &kMinBoxes,
                                       float kCenterVariance = 0.1, const float kSizeVariance = 0.2);



  virtual void Init(float scoreTh, float nmsTh, cv::Size networkSize={}) {
    score_th_ = scoreTh;
    nms_th_ = nmsTh;
    network_size_ = networkSize;
  };

  virtual void Decode(const std::vector<cv::Mat> &outRaw,
                      std::vector<Object> &objects,
                      const vector<string> &onames = {},
                      const cv::Size &img_size = {}) = 0;

  virtual void BatchDecode(const vector<cv::Mat> &outRaws,
                           vector<vector<Object>> &objects,
                           const vector<string> &onames = {},
                           const vector<cv::Size> &img_sizes = {});
  virtual vector<string> ExpectedLayerNames() const { return {}; }


  virtual void ProcessInput(cv::Mat frame, float *out){};
  virtual ~DetectionDecoder() {}
  std::function<void(cv::Mat, float *)> GetNormalizationFunct() {
    return normalization_fnt_;
  };

  void SetLabels(const std::vector<std::string> &lb) { labels_ = lb; }
  std::vector<std::string> GetLabels() const { return labels_; }
  std::string IdToLabel(const int i) const {
    if (labels_.empty()) return {};
    return labels_[i];
  }
  static void DrawObjects(cv::Mat &bgr, const std::vector<Object> &objects,
                          const std::set<int> &selected_ids = {},
                          const std::vector<std::string> *labels = nullptr);

 protected:
  std::function<void(cv::Mat, float *)> normalization_fnt_{};
  std::vector<std::string> labels_;
  float score_th_{0.5};
  float nms_th_{0.25};
  cv::Size network_size_{};
};

#endif  // DETECTION_DECODER_H
