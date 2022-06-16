#include "io_utils.h"
#include <fstream>

using namespace std;

nlohmann::json LoadJSon(const std::string &fpath) {
    std::ifstream ifile(fpath);
    std::stringstream buffer;
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

    const std::string model_path = conf.value("path", "");
    const std::string mpath_config = conf.value("path_config", "");

    const float mc1 = conf.value("mc1", 0.0f);
    const float mc2 = conf.value("mc2", 0.0f);
    const float mc3 = conf.value("mc3", 0.0f);

    std::cout << "Model name: " << model_name << std::endl;
    std::cout << "  Type         : " << model_type << std::endl;
    std::cout << "  Input Size   : [" << input_w << 'X' << input_h << ']'
              << std::endl;
    std::cout << "  Input Scale  :  " << scale << std::endl;
    std::cout << "  Input Mean   : [" << mc1 << ',' << mc2 << ',' << mc3 << ']'
              << std::endl;
    std::cout << "  Swap Channels:  " << swap_ch << std::endl;
    std::cout << "  Framework    :  " << framework << std::endl;
    std::cout << "  Target       :  " << target << std::endl;
    std::cout << "  Backend      :  " << backend << std::endl;
    std::cout << "  File Path    :  " << model_path << std::endl;

    std::shared_ptr<Predictor> result;
//    if (backend == "TRT") {
//        result = TensorRTPredictor::Create(model_path, framework);
//    } else if (backend == "OCV") {
//        result = OpenCVPredictor::Create(model_path, mpath_config, framework);
//        auto p = dynamic_cast<OpenCVPredictor *>(result.get());
//        if (target == "GPU") {
//            //#ifdef HAVE_CUDA
//            p->SetBackend(cv::dnn::DNN_BACKEND_CUDA);
//            p->SetTargetDevice(cv::dnn::DNN_TARGET_CUDA);
//            //#else
//            //      std::cerr << "Your version of OpenCV does not support GPU
//            //      inference. "
//            //                   "Therefore, CPU inference will be used"
//            //                << std::endl;
//            //#endif
//        } else {
//            p->SetBackend(cv::dnn::DNN_BACKEND_OPENCV);
//            p->SetTargetDevice(cv::dnn::DNN_TARGET_CPU);
//        }
//    }
    result->setExplictInputSize({input_w, input_h});
    result->SetInputParamsNorm(scale, {mc1, mc2, mc3}, swap_ch);
    return result;
}
