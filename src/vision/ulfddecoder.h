#ifndef ULFDDECODER_H
#define ULFDDECODER_H

using namespace std;
#include "detection_decoder.h"

class ULFDDecoder : public DetectionDecoder
{
public:
    virtual void Init(float scoreTh, float nmsTh, cv::Size networkSize={}) override;
    virtual void Decode(const std::vector<cv::Mat> &outRaw,
                        std::vector<Object> &objects,
                        const vector<string> &onames = {},
                        const cv::Size &img_size={}) override;
    vector<string> ExpectedLayerNames() const override {
        return {"boxes", "scores"};
    }
private:
    vector<Anchor> priors_;
};

#endif // ULFDDECODER_H
