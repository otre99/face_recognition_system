#include "ulfddecoder.h"


void ULFDDecoder::Init(float scoreTh, float nmsTh, cv::Size networkSize)  {
    DetectionDecoder::Init(scoreTh, nmsTh, networkSize);

    const std::vector<std::vector<float>> kMinBoxes{{10.0f, 16.0f, 24.0f},
                                                    {32.0f, 48.0f},
                                                    {64.0f, 96.0f},
                                                    {128.0f, 192.0f, 256.0f}};
    const std::vector<float> kStrides{8.0, 16.0, 32.0, 64.0};
    priors_ = GeneratePriors(networkSize.width,
                             networkSize.height,
                             kStrides,
                             kMinBoxes,
                             0.1,0.2);
}

void ULFDDecoder::Decode(const std::vector<cv::Mat> &outRaw, std::vector<Object> &objects, const vector<string> &onames, const cv::Size &img_size)
{
    float const *scores_ptr=nullptr;
    float const *bboxes_ptr=nullptr;
    float const *landmarks_ptr=nullptr;
    for (size_t i=0; i<onames.size(); ++i){
        if (onames[i] == "boxes") {
            bboxes_ptr = outRaw[i].ptr<float>(0);
            continue;
        }
        if (onames[i] == "scores") {
            scores_ptr = outRaw[i].ptr<float>(0);
            continue;
        }
    }

    if (bboxes_ptr==nullptr || scores_ptr == nullptr){
        cerr << "Missing expected layers !!!" << endl;
        return;
    }

    vector<float> scores;
    vector<cv::Rect2d> detections;
    cv::Rect2d det;
    vector<cv::Point2f> lptr;
    for (size_t i = 0; i < priors_.size(); i++) {
        if (scores_ptr[i * 2 + 1] > score_th_) {

            float cx =
                bboxes_ptr[i * 4] * 0.1 * priors_[i].w + priors_[i].cx;
            float cy =
                bboxes_ptr[i * 4 + 1] * 0.1 * priors_[i].h + priors_[i].cy;
            float w = exp(bboxes_ptr[i * 4 + 2] * 0.2) * priors_[i].w;
            float h = exp(bboxes_ptr[i * 4 + 3] * 0.2) * priors_[i].h;

            det.x = (cx - 0.5 * w) * img_size.width;
            det.y = (cy - 0.5 * h) * img_size.height;
            det.width = w * img_size.width;
            det.height = h * img_size.height;

            scores.push_back(scores_ptr[i * 2 + 1]);
            detections.push_back(det);
        }
    }
    vector<int> idxs;
    cv::dnn::NMSBoxes(detections,scores,score_th_,nms_th_,idxs);

    objects.clear();
    Object o;
    for (int i : idxs){
        o.label = 1;
        o.prob = scores[i];
        o.rect = detections[i];
        objects.push_back(o);
    }
}

