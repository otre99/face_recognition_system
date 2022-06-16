#ifndef IO_UTILS_H_
#define IO_UTILS_H_
#include <memory>
#include "nlohmann/json.hpp"
#include "vision/predictor.h"

nlohmann::json LoadJSon(const std::string &fpath);
std::shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf);


#endif // IO_UTILS_H_
