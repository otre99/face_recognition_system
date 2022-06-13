#ifndef RETINAFACEDECODER_H
#define RETINAFACEDECODER_H

#include "detection_decoder.h"

using namespace std;

struct FaceLandmarks {
    cv::Point2f leye, reye;
    cv::Point2f nose;
    cv::Point2f lmouth, rmouth;
};

class RetinaFaceDecoder : public DetectionDecoder
{
public:
    virtual void Init(float scoreTh, float nmsTh, cv::Size networkSize={}) override;
    virtual void Decode(const vector<cv::Mat> &outRaw,
                        vector<Object> &objects,
                        const vector<string> &onames = {},
                        const cv::Size &img_size={}) override;
    void DecodeWithLandmarks(const vector<cv::Mat> &outRaw,
                                     vector<Object> &objects,
                                     vector<FaceLandmarks> *landmarks,
                                     const vector<string> &onames,
                                     const cv::Size &img_size);
    void BatchDecodeWithLandmarks(const vector<cv::Mat> &outRaws,
                             vector<vector<Object> > &objects,
                             vector<vector<FaceLandmarks>> *landmarks,
                             const vector<string> &onames,
                             const vector<cv::Size> &img_sizes);

    vector<string> ExpectedLayerNames() const override {
        return {"boxes", "scores", "landmarks"};
    }
private:
    vector<Anchor> priors_;
};

#endif // RETINAFACEDECODER_H
