#include <opencv2/opencv.hpp>
#include "cvision/retinafacedecoder.h"

using namespace std;
void Draw(cv::Mat &image, const vector<DetectionDecoder::Object> &objects, const string &header_msg) {
    char text[256];
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(header_msg, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);
    cv::rectangle(image, {0, 0, label_size.width+1, label_size.height + baseLine+1}, {0,0,0}, -1);
    cv::putText(image, header_msg, cv::Point(1, label_size.height+baseLine/2), cv::FONT_HERSHEY_SIMPLEX, 1, {255,255,255}, 1);

    for (const auto &obj : objects) {


        cv::Scalar color = cv::Scalar(0, 0, 0);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        } else {
            txt_color = cv::Scalar(255, 255, 255);
        }
        cv::rectangle(image, obj.rect,
                      cv::Scalar((17 * obj.label) % 256, (7 * obj.label) % 256,
                                 (37 * obj.label) % 256),
                      2);

        sprintf(text, "id-%d", obj.label);

        label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7;
        int x = obj.rect.x;
        int y = obj.rect.y - 1 - label_size.height-baseLine;
        if (y < 0) y = 0;
        cv::rectangle(
            image,
            cv::Rect(cv::Point(x, y),
                     cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
}

int main(int argc, char *argv[]){

    RetinaFaceDecoder decoder;
    decoder.Init(0.5,0.1,{320,320});

    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/rccr/REPOS/models/RetinaNetResnet50_SIZE320_BATCH1.onnx");
    cv::Mat image = cv::imread("/home/rccr/REPOS/faces.jpg");
    cv::Mat blob = cv::dnn::blobFromImage(image,1.0,{320,320},{104,117,123},false);

    net.setInput(blob);
    vector<cv::Mat> outputs;
    net.forward(outputs, vector<string>{"boxes", "scores", "landmarks"});

    vector<DetectionDecoder::Object> objects;
    decoder.Decode(outputs,objects,vector<string>{"boxes", "scores", "landmarks"},image.size());


    Draw(image, objects,"Detections");
    cv::imshow("DET",image);
    cv::waitKey(-1);

    return 0;
}
