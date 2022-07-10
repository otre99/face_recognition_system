#include <iostream>
#include <opencv2/opencv.hpp>
#include "vision/opencv_predictor.h"
#include "face_detection.h"
#include "vision/tracker.h"

using namespace std;

void Draw(cv::Mat &image, const vector<TrackedObject> &detections) {
    for (const auto &obj : detections) {

        cv::Scalar color = cv::Scalar(0, 0, 0);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        } else {
            txt_color = cv::Scalar(255, 255, 255);
        }
        cv::rectangle(image, obj.rect,
                      cv::Scalar((17 * obj.id) % 256, (7 * obj.id) % 256,
                                 (37 * obj.id) % 256),
                      2);

        char text[256];
        sprintf(text, "id-%ld", obj.id);


        int baseLine = 0;
        cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        if (y > image.rows) y = image.rows;
        cv::rectangle(
            image,
            cv::Rect(cv::Point(x, y),
                     cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
}

int main()
{
    /*
    FaceDetection faceDet;
    faceDet.Init();

    cv::VideoCapture cap("/home/rccr/REPOS/face_recognition_system/videos/face-demographics-walking.mp4");
    Tracker tracker;
    tracker.Init(3,5,0.25);

    cv::Mat frame;
    vector<cv::Mat> outputs;


    while (cap.read(frame)){
        faceDet.Process(frame);
        //Draw(frame, tracker.GetTrackedObjects());


        cv::imshow("DET", frame);
        cv::waitKey(-1);
    }

    return 0;
    */
}
