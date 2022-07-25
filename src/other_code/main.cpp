#include <iostream>
#include "pose_detector.h"
using namespace std;

int main()
{
    PoseDetector pose;

    vector<cv::Point2f> landmarks;
    landmarks.push_back({});
    landmarks.push_back({0.32466468, 0.53110427});
    landmarks.push_back({0.6256795,0.32874084});
    landmarks.push_back({0.54569644,0.6319673 });
    landmarks.push_back({0.5696851, 0.8658758});
    landmarks.push_back({0.82588696 ,0.6921813});
    landmarks[4]+=landmarks[5];
    landmarks[4]/=2;


    auto p  = pose.Detect(landmarks);

    cout << " roll: " <<  p.roll << endl;
    cout << "pitch: " <<  p.pitch << endl;
    cout << "  yaw: " <<  p.yaw << endl;

    return 0;
}
