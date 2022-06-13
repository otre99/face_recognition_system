#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "cvision/detection_decoder.h"
#include "cvision/object_tracker.h"
#include "cvision/retinafacedecoder.h"
#include "cvision/predictor.h"

using namespace std;


class FaceRecognition
{

public:
    static std::atomic_int task_counter;
    void Process(const cv::Mat &frame,
                 const vector<DetectionDecoder::Object> &dets,
                 const vector<FaceLandmarks> &landmarks);
    void Start(int slot, const string &name, std::condition_variable * gcv);
    void Stop();
    string GetName() const {return name_;}
    ~FaceRecognition();
    void Draw();
    void SetModels(shared_ptr<Predictor> recog_model, shared_ptr<Predictor> alig_model);

private:
    void Run();
    bool CalculateLandmarks(const cv::Mat &frame, cv::Rect &rect, FaceLandmarks &flm, float th);
    bool DecodeOnet(const vector<cv::Mat> &outputs, FaceLandmarks &flm, float th, cv::Rect &rect);

    vector<DetectionDecoder::Object> currdets_;
    vector<FaceLandmarks> landmarks_;
    cv::Mat currframe_;
    bool exit_{false};
    string name_;
    ObjectTracker faceTracker_;
    int slot_{-1};
    shared_ptr<Predictor> recog_model_;
    shared_ptr<Predictor> alig_model_;

    //-----------------------------------
    std::condition_variable cv_;
    std::mutex mutex_;
    std::condition_variable *globalCV_{nullptr};
    std::thread thread_;
    bool waiting_{false};

};

#endif // FACERECOGNITION_H
