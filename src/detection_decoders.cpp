#include "detection_decoders.h"

vector<Anchor> GeneratePriors(const int kW, const int kH,
                              const vector<float> &kStrides,
                              const vector<vector<float>> &kMinBoxes,
                              float kCenterVariance,
                              const float kSizeVariance) {
  std::vector<Anchor> priors;
  std::vector<std::vector<float>> featuremap_size;
  std::vector<std::vector<float>> shrinkage_size;
  for (auto size : {kW, kH}) {
    std::vector<float> fm_item;
    for (float stride : kStrides) {
      fm_item.push_back(std::ceil(size / stride));
    }
    featuremap_size.push_back(fm_item);
  }
  shrinkage_size.push_back(kStrides);
  shrinkage_size.push_back(kStrides);

  /* generate prior anchors */
  for (int index = 0; index < 4; index++) {
    float scale_w = kW / shrinkage_size[0][index];
    float scale_h = kH / shrinkage_size[1][index];
    for (int j = 0; j < featuremap_size[1][index]; j++) {
      for (int i = 0; i < featuremap_size[0][index]; i++) {
        float x_center = (i + 0.5) / scale_w;
        float y_center = (j + 0.5) / scale_h;

        for (float k : kMinBoxes[index]) {
          float w = k / float(kW);
          float h = k / float(kH);
          priors.push_back({Clip(x_center), Clip(y_center), Clip(w), Clip(h)});
        }
      }
    }
  }
  return priors;
}

// ULFDDecoder
void ULFDDecoder::Init(float scoreTh, float nmsTh,
                       const cv::Size &networkSize) {
  DetectionDecoder::Init(scoreTh, nmsTh, networkSize);
  const std::vector<std::vector<float>> kMinBoxes{{10.0f, 16.0f, 24.0f},
                                                  {32.0f, 48.0f},
                                                  {64.0f, 96.0f},
                                                  {128.0f, 192.0f, 256.0f}};
  const std::vector<float> kStrides{8.0, 16.0, 32.0, 64.0};
  priors_ = GeneratePriors(networkSize.width, networkSize.height, kStrides,
                           kMinBoxes, 0.1, 0.2);
}

void ULFDDecoder::Decode(const std::vector<cv::Mat> &outRaw,
                         std::vector<BBox> &objects,
                         const vector<string> &onames,
                         const cv::Size &img_size) {
  float const *scores_ptr = nullptr;
  float const *bboxes_ptr = nullptr;
  float const *landmarks_ptr = nullptr;
  for (size_t i = 0; i < onames.size(); ++i) {
    if (onames[i] == "boxes") {
      bboxes_ptr = outRaw[i].ptr<float>(0);
      continue;
    }
    if (onames[i] == "scores") {
      scores_ptr = outRaw[i].ptr<float>(0);
      continue;
    }
  }

  if (bboxes_ptr == nullptr || scores_ptr == nullptr) {
    cerr << "Missing expected layers !!!" << endl;
    return;
  }

  vector<float> scores;
  vector<cv::Rect2d> detections;
  cv::Rect2d det;
  vector<cv::Point2f> lptr;
  for (size_t i = 0; i < priors_.size(); i++) {
    if (scores_ptr[i * 2 + 1] > score_th_) {

      float cx = bboxes_ptr[i * 4] * 0.1 * priors_[i].w + priors_[i].cx;
      float cy = bboxes_ptr[i * 4 + 1] * 0.1 * priors_[i].h + priors_[i].cy;
      float w = exp(bboxes_ptr[i * 4 + 2] * 0.2) * priors_[i].w;
      float h = exp(bboxes_ptr[i * 4 + 3] * 0.2) * priors_[i].h;

      det.x = (cx - 0.5 * w) * img_size.width;
      det.y = (cy - 0.5 * h) * img_size.height;
      det.width = w * img_size.width;
      det.height = h * img_size.height;

      scores.push_back(scores_ptr[i * 2 + 1]);
      detections.push_back(det);
    }
  }
  vector<int> idxs;
  cv::dnn::NMSBoxes(detections, scores, score_th_, nms_th_, idxs);

  objects.clear();
  BBox o;
  for (int i : idxs) {
    o.label = 1;
    o.score = scores[i];
    o.rect = detections[i];
    objects.push_back(o);
  }
}

// RetinaFaceDecoder
void RetinaFaceDecoder::Init(float scoreTh, float nmsTh,
                             const cv::Size &network_input_size) {
  DetectionDecoder::Init(scoreTh, nmsTh, network_input_size);
  const std::vector<float> kStrides{8.0, 16.0, 32.0};
  const std::vector<std::vector<float>> kMinBoxes{
      {16.0f, 32.0f}, {64.0f, 128.0f}, {256.0f, 512.0f}};
  priors_ = GeneratePriors(network_input_size.width, network_input_size.height,
                           kStrides, kMinBoxes, 0.1, 0.2);
}

void RetinaFaceDecoder::Decode(const std::vector<cv::Mat> &outRaw,
                               std::vector<BBox> &objects,
                               const vector<string> &onames,
                               const cv::Size &img_size) {
  float const *scores_ptr = nullptr;
  float const *bboxes_ptr = nullptr;
  float const *landmarks_ptr = nullptr;
  for (size_t i = 0; i < onames.size(); ++i) {
    if (onames[i] == "boxes") {
      bboxes_ptr = outRaw[i].ptr<float>(0);
      continue;
    }
    if (onames[i] == "scores") {
      scores_ptr = outRaw[i].ptr<float>(0);
      continue;
    }
    if (onames[i] == "landmarks") {
      landmarks_ptr = outRaw[i].ptr<float>(0);
    }
  }

  if (bboxes_ptr == nullptr || scores_ptr == nullptr ||
      landmarks_ptr == nullptr) {
    cerr << "Missing expected layers !!!" << endl;
    return;
  }

  cv::Size dstSize;
  const double s = 1.0/GetScaleFactorForResize(img_size, network_input_size_);
  dstSize.width  = network_input_size_.width*s;
  dstSize.height = network_input_size_.height*s;

  vector<float> scores;
  vector<cv::Rect2d> detections;
  cv::Rect2d det;
  FaceLandmarks flm;
  vector<FaceLandmarks> tmp_land;
  for (size_t i = 0; i < priors_.size(); i++) {
    if (scores_ptr[i * 2 + 1] > score_th_) {

      float cx = bboxes_ptr[i * 4] * 0.1 * priors_[i].w + priors_[i].cx;
      float cy = bboxes_ptr[i * 4 + 1] * 0.1 * priors_[i].h + priors_[i].cy;
      float w = exp(bboxes_ptr[i * 4 + 2] * 0.2) * priors_[i].w;
      float h = exp(bboxes_ptr[i * 4 + 3] * 0.2) * priors_[i].h;

      det.x = (cx - 0.5 * w) * dstSize.width;
      det.y = (cy - 0.5 * h) * dstSize.height;
      det.width = w * dstSize.width;
      det.height = h * dstSize.height;

      scores.push_back(scores_ptr[i * 2 + 1]);
      detections.push_back(det);

      cx = landmarks_ptr[i * 10 + 0] * 0.1 * priors_[i].w + priors_[i].cx;
      cy = landmarks_ptr[i * 10 + 1] * 0.1 * priors_[i].h + priors_[i].cy;
      flm.leye = {cx * dstSize.width, cy * dstSize.height};

      cx = landmarks_ptr[i * 10 + 2] * 0.1 * priors_[i].w + priors_[i].cx;
      cy = landmarks_ptr[i * 10 + 3] * 0.1 * priors_[i].h + priors_[i].cy;
      flm.reye = {cx * dstSize.width, cy * dstSize.height};

      cx = landmarks_ptr[i * 10 + 4] * 0.1 * priors_[i].w + priors_[i].cx;
      cy = landmarks_ptr[i * 10 + 5] * 0.1 * priors_[i].h + priors_[i].cy;
      flm.nose = {cx * dstSize.width, cy * dstSize.height};

      cx = landmarks_ptr[i * 10 + 6] * 0.1 * priors_[i].w + priors_[i].cx;
      cy = landmarks_ptr[i * 10 + 7] * 0.1 * priors_[i].h + priors_[i].cy;
      flm.lmouth = {cx * dstSize.width, cy * dstSize.height};

      cx = landmarks_ptr[i * 10 + 8] * 0.1 * priors_[i].w + priors_[i].cx;
      cy = landmarks_ptr[i * 10 + 9] * 0.1 * priors_[i].h + priors_[i].cy;
      flm.rmouth = {cx * dstSize.width, cy * dstSize.height};
      flm.relative_coords = false;
      tmp_land.push_back(flm);
    }
  }
  vector<int> idxs;
  cv::dnn::NMSBoxes(detections, scores, score_th_, nms_th_, idxs);

  objects.resize(idxs.size());
  landmarks_.resize(idxs.size());
  BBox o;
  int i;
  for (size_t j = 0; j < idxs.size(); ++j) {
    i = idxs[j];
    o.label = 1;
    o.score = scores[i];
    o.rect = detections[i];
    objects[j] = o;
    landmarks_[j] = tmp_land[i];
  }
}
