#include "detection_decoder.h"
namespace {
inline float Clip(float x) { return std::min(std::max(0.0F, x), 1.0F); }
}
vector<DetectionDecoder::Anchor> DetectionDecoder::GeneratePriors(const int kW, const int kH,
                                                                 const vector<float> &kStrides,
                                                                 const vector<vector<float>> &kMinBoxes,
                                                                 float kCenterVariance, const float kSizeVariance)
{
    std::vector<DetectionDecoder::Anchor> priors;
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

void DetectionDecoder::BatchDecode(const vector<cv::Mat> &outRaws,
                         vector<vector<Object>> &objects, const vector<string> &onames,
                         const vector<cv::Size> &img_sizes){

    objects.clear();
    const int bn = outRaws[0].size.p[0];
    for (size_t i=0; i<bn; ++i){
        vector<cv::Mat> outputs;
        for (const auto &o : outRaws){
            vector<int> ss(o.size.p,o.size.p+o.size.dims());
            ss[0]=1;
            outputs.push_back(cv::Mat(ss, o.type(), (void*)o.ptr(i)));
        }
        vector<Object> objs;
        Decode(outputs, objs, onames, img_sizes[min(img_sizes.size(),i)]);
        objects.push_back(objs);
    }
}

void DetectionDecoder::DrawObjects(cv::Mat& image,
                                   const std::vector<Object>& objects,
                                   const std::set<int>& selected_ids,
                                   const std::vector<std::string>* labels) {
  for (size_t i = 0; i < objects.size(); i++) {
    const Object& obj = objects[i];

    if (!selected_ids.empty() &&
        selected_ids.find(obj.label) == selected_ids.end())
      continue;

    cv::Scalar color = cv::Scalar(255, 255, 0);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }
    cv::rectangle(image, obj.rect, color, 2);

    char text[256];
    if (labels) {
      sprintf(text, "%s %.1f%%", (*labels)[obj.label].c_str(), obj.prob * 100);
    } else {
      sprintf(text, "label-%d %.1f%%", obj.label, obj.prob * 100);
    }

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    // int y = obj.rect.y - label_size.height - baseLine;
    if (y > image.rows) y = image.rows;
    // if (x + label_size.width > image.cols)
    // x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
}
