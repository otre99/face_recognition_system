#include "videosources.h"


bool VideoSources::Init(const vector<string> &sources)
{
    sourcesConfig_ = sources;
    return true;
}

int VideoSources::CollectFrames(vector<cv::Mat> &frames){

    frames.resize(caps_.size(), {});
    cv::Mat currFrame;
    int counter=0;
    for (size_t i=0; i<caps_.size(); ++i){
        if (valid_caps_[i]){
            valid_caps_[i] = caps_[i].read(currFrame);
            if ( valid_caps_[i] ) currFrame.copyTo(frames[i]);
        }
        counter+=valid_caps_[i];
    }
    return counter;
}

int VideoSources::OpenStreams(){
    caps_.resize(sourcesConfig_.size());
    valid_caps_.resize(sourcesConfig_.size(),true);
    int counter=0;
    for (size_t i=0; i<sourcesConfig_.size(); ++i){
        if ( caps_[i].open(sourcesConfig_[i]) == false ){
            cerr << "Warning opening video source: " << sourcesConfig_[i] << endl;
            valid_caps_[i]=false;
        }
        counter+=valid_caps_[i];
    }
    return counter;
}

void VideoSources::CloseStreams(){
    for (size_t i=0; i<caps_.size(); ++i){
        if (valid_caps_[i]) caps_[i].release();
    }
}
