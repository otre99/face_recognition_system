#include "openvino_predictor.h"


using namespace std;

int main(){
    string model_path = "/test_storage/intel/face-detection-0205/FP32/face-detection-0205";
    auto det = OpenVinoPredictor::Create(model_path+".bin", model_path+".xml", "IE");


    return 0;
}
