#include "retinafacedecoder.h"


void RetinaFaceDecoder::Init(float scoreTh, float nmsTh, cv::Size networkSize)  {
    DetectionDecoder::Init(scoreTh, nmsTh, networkSize);
    const std::vector<float> kStrides{8.0, 16.0, 32.0};
    const std::vector<std::vector<float>> kMinBoxes{{16.0f, 32.0f}, {64.0f, 128.0f}, {256.0f, 512.0f}};
    priors_ = GeneratePriors(networkSize.width,
                             networkSize.height,
                             kStrides,
                             kMinBoxes,
                             0.1,0.2);
}

void RetinaFaceDecoder::Decode(const std::vector<cv::Mat> &outRaw, std::vector<Object> &objects, const vector<string> &onames, const cv::Size &img_size)
{
    DecodeWithLandmarks(outRaw,objects,nullptr,onames,img_size);
}

void RetinaFaceDecoder::DecodeWithLandmarks(const std::vector<cv::Mat> &outRaw,
                                            std::vector<Object> &objects,
                                            vector<FaceLandmarks> *landmarks,
                                            const vector<string> &onames,
                                            const cv::Size &img_size){
    float const *scores_ptr=nullptr;
    float const *bboxes_ptr=nullptr;
    float const *landmarks_ptr=nullptr;
    for (size_t i=0; i<onames.size(); ++i){
        if (onames[i] == "boxes") {
            bboxes_ptr = outRaw[i].ptr<float>(0);
            //cout << " A " << outRaw[i].size << endl;
            continue;
        }
        if (onames[i] == "scores") {
            scores_ptr = outRaw[i].ptr<float>(0);
            //cout << " B " << outRaw[i].size << endl;
            continue;
        }
        if (onames[i] == "landmarks") {
            landmarks_ptr = outRaw[i].ptr<float>(0);
            //cout << " C " << outRaw[i].size << endl;
        }
    }

    if (bboxes_ptr==nullptr || scores_ptr == nullptr || landmarks_ptr==nullptr){
        cerr << "Missing expected layers !!!" << endl;
        return;
    }

    vector<float> scores;
    vector<cv::Rect2d> detections;
    cv::Rect2d det;
    FaceLandmarks flm;
    vector<FaceLandmarks> tmp_land;
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

            if (landmarks){

                cx = landmarks_ptr[i * 10 + 0] * 0.1 * priors_[i].w + priors_[i].cx;
                cy = landmarks_ptr[i * 10 + 1] * 0.1 * priors_[i].h + priors_[i].cy;
                flm.leye = {cx*img_size.width,cy*img_size.height};

                cx = landmarks_ptr[i * 10 + 2] * 0.1 * priors_[i].w + priors_[i].cx;
                cy = landmarks_ptr[i * 10 + 3] * 0.1 * priors_[i].h + priors_[i].cy;
                flm.reye = {cx*img_size.width,cy*img_size.height};

                cx = landmarks_ptr[i * 10 + 4] * 0.1 * priors_[i].w + priors_[i].cx;
                cy = landmarks_ptr[i * 10 + 5] * 0.1 * priors_[i].h + priors_[i].cy;
                flm.nose = {cx*img_size.width,cy*img_size.height};

                cx = landmarks_ptr[i * 10 + 6] * 0.1 * priors_[i].w + priors_[i].cx;
                cy = landmarks_ptr[i * 10 + 7] * 0.1 * priors_[i].h + priors_[i].cy;
                flm.lmouth = {cx*img_size.width,cy*img_size.height};

                cx = landmarks_ptr[i * 10 + 8] * 0.1 * priors_[i].w + priors_[i].cx;
                cy = landmarks_ptr[i * 10 + 9] * 0.1 * priors_[i].h + priors_[i].cy;
                flm.rmouth = {cx*img_size.width,cy*img_size.height};

                tmp_land.push_back(flm);
            }
        }
    }
    vector<int> idxs;
    cv::dnn::NMSBoxes(detections,scores,score_th_,nms_th_,idxs);

    objects.resize(idxs.size());
    if (landmarks) landmarks->resize(idxs.size());
    Object o;
    int i;
    for (size_t j=0; j<idxs.size(); ++j){
        i = idxs[j];
        o.label = 1;
        o.prob = scores[i];
        o.rect = detections[i];
        objects[j] = o;
        if (landmarks){
            landmarks->operator[](j) = tmp_land[i];
        }
    }
}

void RetinaFaceDecoder::BatchDecodeWithLandmarks(const vector<cv::Mat> &outRaws,
                              vector<vector<Object>> &objects,
                              vector<vector<FaceLandmarks> > *landmarks,
                              const vector<string> &onames,
                                                 const vector<cv::Size> &img_sizes)
{
    objects.clear();
    if (landmarks) landmarks->clear();
    const int bn = outRaws[0].size.p[0];
    for (size_t i=0; i<bn; ++i){
        vector<cv::Mat> outputs;
        for (const auto &o : outRaws){
            vector<int> ss(o.size.p,o.size.p+o.size.dims());
            ss[0]=1;
            outputs.push_back(cv::Mat(ss, o.type(), (void*)o.ptr(i)));
        }
        vector<Object> objs;
        vector<FaceLandmarks> flms;
        DecodeWithLandmarks(outputs, objs, &flms, onames, img_sizes[min(img_sizes.size(),i)]);
        objects.push_back(objs);
        if (landmarks) landmarks->push_back(flms);
    }
}
