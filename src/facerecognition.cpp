#include "facerecognition.h"
#include "cvision/funct_utils.h"
#include "cvision/tensor_rt_predictor.h"

std::atomic_int FaceRecognition::task_counter{-1};

void FaceRecognition::Start(int slot, const string &name, condition_variable *gcv){
    name_ = name;
    slot_ = slot;
    globalCV_ = gcv;
    faceTracker_.Init(3, 2, 0.2);
    exit_=false;
    thread_ = thread(&FaceRecognition::Run, this);
}

void FaceRecognition::Stop(){
    exit_=true;
    cv_.notify_all();
}

FaceRecognition::~FaceRecognition(){
    Stop();
    thread_.join();
}


void  FaceRecognition::Process(const cv::Mat &frame, const vector<DetectionDecoder::Object> &dets, const vector<FaceLandmarks> &landmarks){

    if ( !landmarks.empty() && landmarks.size()!= dets.size()){
        throw  "Detection and landmarks must have the same size";
    }

    frame.copyTo(currframe_);
    currdets_ = dets;
    landmarks_ = landmarks;
    cv_.notify_one();
}

void FaceRecognition::Draw(){
    if (currframe_.empty()) return;

    faceTracker_.Draw(currframe_);

    for (auto &obj : faceTracker_.GetTrackedObjects()){
        if (obj.InCurrentFrame()){
            FaceLandmarks l;

            cv::Rect faceRect(obj.rect);
            bool is_face = CalculateLandmarks(currframe_, faceRect, l, 0.25);
            cv::putText(currframe_,is_face ? "FACE": "NO FACE", obj.rect.tl(),1,1,{255,0,0});

            cv::circle(currframe_,l.leye,1,{0,0,0},2,-1);
            cv::circle(currframe_,l.reye,1,{0,0,0},2,-1);

            cv::circle(currframe_,l.nose,1,{0,0,0},2,-1);

            cv::circle(currframe_,l.lmouth,1,{0,0,0},2,-1);
            cv::circle(currframe_,l.rmouth,1,{0,0,0},2,-1);

            cv::imshow("face", currframe_(faceRect));
            std::cout << faceRect.size();
            cv::rectangle(currframe_,faceRect,{255,255,255},3);

        }
    }

    for (const auto l : landmarks_){
        cv::circle(currframe_,l.leye,1,{0,0,255},2,-1);
        cv::circle(currframe_,l.reye,1,{0,0,255},2,-1);

        cv::circle(currframe_,l.nose,1,{0,0,255},2,-1);

        cv::circle(currframe_,l.lmouth,1,{0,0,255},2,-1);
        cv::circle(currframe_,l.rmouth,1,{0,0,255},2,-1);
    }

    cv::imshow(name_, currframe_);
}

void FaceRecognition::SetModels(shared_ptr<Predictor> recog_model, shared_ptr<Predictor> alig_model)
{
    recog_model_ = recog_model;
    alig_model_ = alig_model;
}

void FaceRecognition::Run() {

    for(;;){
        waiting_=true;
        unique_lock<mutex> lk(mutex_);
        cv_.wait(lk);
        waiting_ = false;


        vector<long> userUniqueIds;



        faceTracker_.Update(currdets_, 1);

        FaceRecognition::task_counter.fetch_add(1);
        globalCV_->notify_one();
        if (exit_) break;
    }
}

bool FaceRecognition::CalculateLandmarks(const cv::Mat &frame, cv::Rect &rect, FaceLandmarks &flm, float th){

    cv::Mat croped_face = frame(RectInsideFrame(rect, frame));
    vector<cv::Mat> outputs;
    TensorRTPredictor *tp=dynamic_cast<TensorRTPredictor*>(alig_model_.get());
    if (tp) {
        tp->PredictOnSlot(slot_, croped_face, outputs, {"boxes", "landmarks", "scores"});
    } else {
        alig_model_->Predict(croped_face, outputs, {"boxes", "landmarks", "scores"});
    }

    return DecodeOnet(outputs, flm, th, rect);
}

bool  FaceRecognition::DecodeOnet(const vector<cv::Mat> &outputs, FaceLandmarks &flm, float th, cv::Rect &rect){
    const float *boxes_ptr = outputs[0].ptr<float>(0);
    const float *land_ptr = outputs[1].ptr<float>(0);
    const float *scores_ptr = outputs[2].ptr<float>(0);

    flm.leye.x = rect.x + rect.width*land_ptr[0];
    flm.leye.y = rect.y + rect.height*land_ptr[5];

    flm.reye.x = rect.x + rect.width*land_ptr[1];
    flm.reye.y = rect.y + rect.height*land_ptr[6];

    flm.nose.x = rect.x + rect.width*land_ptr[2];
    flm.nose.y = rect.y + rect.height*land_ptr[7];

    flm.lmouth.x = rect.x + rect.width*land_ptr[3];
    flm.lmouth.y = rect.y + rect.height*land_ptr[8];

    flm.rmouth.x = rect.x + rect.width*land_ptr[4];
    flm.rmouth.y = rect.y + rect.height*land_ptr[9];


    float x1 =  rect.x + rect.width*boxes_ptr[0];
    float y1 =  rect.y + rect.height*boxes_ptr[1];
    float x2 =  (rect.x+rect.width-1)  + rect.width*boxes_ptr[2];
    float y2 =  (rect.y+rect.height-1) + rect.height*boxes_ptr[3];

    float cx = 0.5*(x1+x2);
    float cy = 0.5*(y1+y2);

    float side = max(x2-x1, y2-y1);
    rect.x = cx-0.5*side;
    rect.y = cy-0.5*side;
    rect.width = rect.height = side;
    return true;
}
