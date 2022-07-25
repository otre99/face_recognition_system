#include "io_utils.h"
#include <fstream>
#include "opencv_predictor.h"
using namespace std;

nlohmann::json LoadJSon(const string &fpath) {
    ifstream ifile(fpath);
    stringstream buffer;
    buffer << ifile.rdbuf();
    return nlohmann::json::parse(buffer.str());
}

shared_ptr<Predictor> PredictorFromJson(const nlohmann::json &conf){
    const string model_name = conf.value("name", "");
    const string model_type = conf.value("type", "");
    const string backend = conf.value("backend", "TRT");
    const string target = conf.value("target", "TRT");
    const string framework = conf.value("framework", "ONNX");

    const int input_w = conf.value("input_w", -1);
    const int input_h = conf.value("input_h", -1);
    const float scale = conf.value("scale", 1.0);
    const bool swap_ch = conf.value("swap_ch", false);

    const string model_path = conf.value("weights", "");
    const string mpath_config = conf.value("config", "");

    const float mc1 = conf.value("mc1", 0.0f);
    const float mc2 = conf.value("mc2", 0.0f);
    const float mc3 = conf.value("mc3", 0.0f);

    cout << "Model name     : " << model_name << endl;
    cout << "  Type         : " << model_type << endl;
    cout << "  Input Size   : [" << input_w << 'X' << input_h << ']' << endl;
    cout << "  Input Scale  : " << scale << endl;
    cout << "  Input Mean   : [" << mc1 << ',' << mc2 << ',' << mc3 << ']' << endl;
    cout << "  Swap Channels: " << swap_ch << endl;
    cout << "  Framework    : " << framework << endl;
    cout << "  Target       : " << target << endl;
    cout << "  Backend      : " << backend << endl;
    cout << "  Config file  : " << mpath_config << endl;
    cout << "  Weights file : " << model_path << endl;

    shared_ptr<Predictor> result;
    if (backend == "OCV") {
        result = OpenCVPredictor::Create(model_path, mpath_config, framework);
    } else {
        cerr << "Backend '" << backend << "' is not supported" << endl;
        return {};
    }
    result->setExplictInputSize({input_w, input_h});
    result->SetInputParamsNorm(scale, {mc1, mc2, mc3}, swap_ch);
    return result;
}

std::shared_ptr<DetectionDecoder> DetectionDecoderFromJson(const nlohmann::json &conf)
{
    shared_ptr<DetectionDecoder> result;
    const int input_w = conf.value("input_w", -1);
    const int input_h = conf.value("input_h", -1);
    const float obj_th = conf.value("obj_th", 0.5);
    const float nms_th = conf.value("nms_th", 0.2);

    auto decoder_name = conf.value("decoder", "");
    if (decoder_name=="RETINAFACE"){
        result.reset(new RetinaFaceDecoder());
    } else if (decoder_name=="ULFD") {
        result.reset(new ULFDDecoder());
    } else {
        cerr  << "Unknow detection decoder '" << decoder_name << "'" << endl;
        return {};
    }
    result->Init(obj_th,nms_th,{input_w, input_h});
    return result;
}


void TrackerFromJson(const nlohmann::json &conf, Tracker &tracker) {

    const int frames_to_count = conf.value("frames_to_count", 0);
    const int frames_to_discart = conf.value("frames_to_discart", 0);
    const float iou = conf.value("tracker_iou", 0.0);
    cout << "Configuring tracker " << endl;
    cout << "  frames_to_count   : " << frames_to_count << endl;
    cout << "  frames_to_discart : " << frames_to_discart << endl;
    cout << "  tracker_iou       : " << iou << endl;
    tracker.Init(frames_to_count, frames_to_discart, iou);
}
