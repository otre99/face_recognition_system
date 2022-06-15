#include "predictor.h"

cv::Mat Predictor::CreateInputBlob(const vector<cv::Mat> &images) const {
    return cv::dnn::blobFromImages(images, scale_, input_size_, mean_, swap_ch_,
                                   false, CV_32F);
}
