#ifndef IO_UTILS_H_
#define IO_UTILS_H_
#include "detection_decoders.h"
#include "nlohmann/json.hpp"
#include "predictor.h"
#include "tracker.h"
#include <memory>

nlohmann::json LoadJSon(const std::string &fpath);
std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf);
std::shared_ptr<DetectionDecoder>
DetectionDecoderFromJson(const nlohmann::json &conf);

void TrackerFromJson(const nlohmann::json &conf, Tracker &tracker);

#endif // IO_UTILS_H_
