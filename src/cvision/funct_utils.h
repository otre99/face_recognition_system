#ifndef FUNCT_UTILS_H
#define FUNCT_UTILS_H
#include "classification_tracker.h"
#include "classifier.h"
#include "detection_decoder.h"
#include "../nlohmann/json.hpp"
#include "object_tracker.h"
#include "predictor.h"
#include "videosources.h"

nlohmann::json LoadJSon(const std::string &fpath);
std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf);
std::shared_ptr<Classifier> ClassifierFromJson(const nlohmann::json &conf);
std::vector<std::string> StringListFromJson(const nlohmann::json &json_array);
std::shared_ptr<DetectionDecoder> DetectionDecoderFromJson(
    const nlohmann::json &conf);
void ObjectTrackerFromJson(const nlohmann::json &conf, ObjectTracker &tracker);
std::shared_ptr<VideoSources> VideoSourcesFromJson(const nlohmann::json &conf);

void SoftMax(cv::Mat &out);
std::pair<int, float> ArgMax(const cv::Mat &out);
void ScaleRect(cv::Rect &r, float scale);
cv::Rect RectInsideFrame(const cv::Rect &rect, const cv::Mat &frame);
cv::Mat Get4x3Roi(cv::Mat frame);

std::vector<std::string> Split(const std::string &s, char delimiter);
std::pair<std::string, std::string> SplitEx(const std::string &s);

#endif  // FUNCT_UTILS_H
