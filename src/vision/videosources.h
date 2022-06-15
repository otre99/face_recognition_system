#ifndef VIDEOSOURCES_H
#define VIDEOSOURCES_H
#include <opencv2/opencv.hpp>

using namespace std;

class VideoSources
{

public:
    bool Init(const vector<string> &sources);
    int CollectFrames(vector<cv::Mat> &frames);
    size_t NumberOfStreams() const {return sourcesConfig_.size();}
    int OpenStreams();
    void CloseStreams();

private:
    vector<cv::VideoCapture> caps_;
    vector<bool> valid_caps_;
    vector<string> sourcesConfig_;
};

#endif // VIDEOSOURCES_H
