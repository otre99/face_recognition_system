#ifndef IO_UTILS_H_
#define IO_UTILS_H_
#include <memory>
#include "nlohmann/json.hpp"
#include "vision/predictor.h"
#include "vision/tracker.h"

nlohmann::json LoadJSon(const std::string &fpath);
std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf);
void TrackerFromJson(const nlohmann::json &conf, Tracker &tracker);

#endif // IO_UTILS_H_
