#include "dbmanager.h"
#include "draw_utils.h"
#include "faces_manager.h"
#include "image_saver.h"
#include "io_utils.h"
#include "recognition_status_tracker.h"
#include "tracker.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
namespace fs = filesystem;
using namespace std::chrono;

DrawUtils drawer;

string GetLineColorAndMsgToDisplay(const RecognitionStatusTracker::Data &d, cv::Scalar &lcolor) {
  switch (d.status) {
  case RecognitionStatusTracker::DOUBTFUL:
    lcolor = {0,255,255};
    return "DOUBTFUL";
  case RecognitionStatusTracker::UNKNOWN:
    lcolor = {0,0,255};
    return "UNKNOWN";
  case RecognitionStatusTracker::PENDING:
    lcolor = {0,0,0};
    return "PENDING";
  default: // RecognitionStatusTracker::KNOWN:
    lcolor = {0,255,0};
    return d.faceId;
  }
}

void Draw(cv::Mat &image, const vector<TrackedObject> &detections,
          RecognitionStatusTracker &recogTracker, int fps = -1) {

  if (fps > 0) {
    drawer.DrawFPS(image, fps);
  }
  for (const auto &obj : detections) {
    if (obj.last_frame == 0) {

      if (recogTracker.Exists(obj.id)) {
        const auto d = recogTracker.Get(obj.id);
        cv::Scalar lcolor;

        const string msg = GetLineColorAndMsgToDisplay(d, lcolor);
        drawer.SetLineColor(lcolor);
        drawer.DrawTrackedObj(image, obj, msg);
      }
    }
  }
}

using Status = RecognitionStatusTracker::Status;
void AddLogLine(ofstream &ofile, const string &faceId, Status st, float recog_th, const string &img_file){

  string faceIdStr="";
  switch (st) {
  case RecognitionStatusTracker::UNKNOWN:
    faceIdStr="UNKNOWN";
    break;
  case RecognitionStatusTracker::KNOWN:
    faceIdStr=faceId;
    break;
  default:
    break;
  }
  time_t currTime = system_clock::to_time_t(system_clock::now());
  ofile << faceIdStr << ','
        << recog_th << ','
        << put_time(localtime(&currTime), "%FT%T%z") << ','
        << img_file
        << endl;
}




int main(int argc, char *argv[]) {
  FacesManager facesManager;
  DBManager dbmanager;
  string video_input = "/dev/video0";
  RecognitionStatusTracker recogTracker;
  filesystem::path captured_frames_folder;
  {
    const auto json_data = LoadJSon(argv[1]);

    // faces manager
    facesManager.Init(json_data["faces_manager"]);

    // db
    const auto mpath = json_data["db"].get<string>();
    if (!dbmanager.Open(mpath, false, facesManager.GetEmbeddingLen())) {
      return -1;
    }
    int recog_metric;
    float recog_low_th, recog_hi_th;
    recog_metric = json_data.value("recog_metric", 0);
    recog_low_th = json_data.value("recog_low_th", 0.8);
    recog_hi_th = json_data.value("recog_hi_th", 1.0);
    dbmanager.Init(recog_metric, recog_low_th, recog_hi_th);

    // tracker
    const auto m = json_data["faces_manager"]["tracker"];
    int life = m.value("frames_to_discart", 0) + 1;
    recogTracker.Init(life);

    // input video
    video_input = json_data["video_input"].get<string>();

    //captured_frames_folder
    captured_frames_folder = fs::path(json_data["captured_frames_folder"].get<string>());
  }
  drawer.Init({});

  cv::VideoCapture cap(video_input);
  if (!cap.isOpened()) {
    cerr << "Error opening input video: " << video_input << endl;
    return -1;
  }

  double fw = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  double fh = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  bool okw = cap.set(cv::CAP_PROP_FRAME_WIDTH, 1080.0);
  bool okh = cap.set(cv::CAP_PROP_FRAME_HEIGHT, fh * 1080 / fw);
  if (!(okw && okh)) {
    cerr << "Failed to resize input video" << endl;
  }
  cv::Mat frame;
  Face face;
  FaceLandmarks lands;
  vector<float> embedding;
  long nframe = 0;
  cv::TickMeter tictac;
  ImageSaver imgSaver;
  auto facesTracker = facesManager.GetFacesTracker();

  ofstream log_file(captured_frames_folder/"logs.csv");
  log_file << "FaceId,RecogTh,Time,Image\n";

//  cv::VideoWriter writer;
//  auto codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
//  double fps = 25.0;                          // framerate of the created video stream
//  string filename = "./live.avi";             // name of the output video file
//  cv::Mat src;
//  cap.read(src);
//  writer.open(filename, codec, fps, src.size(), true);

  while (cap.read(frame)) {

    tictac.start();
    const auto tracked_faces = facesManager.Process(frame);
    for (const auto &obj : tracked_faces) {
      if (!facesTracker.InCurrentFrame(obj) ||
          !facesTracker.IsAcceptableDetection(obj)) {
        continue;
      }

      Status st = RecognitionStatusTracker::PENDING;
      bool need_recog;
      if (!recogTracker.Exists(obj.id)) {
        need_recog = true;
      } else {
        st = recogTracker.Get(obj.id).status;
        need_recog = (st == RecognitionStatusTracker::DOUBTFUL) ||
                     (st == RecognitionStatusTracker::PENDING);
      }

      if (need_recog) {
        string faceId = {};
        lands = facesManager.GetFaceLandmarksOnet(frame, obj.rect);
        face.Init(frame, obj.rect, lands, facesManager.GetAlignMethod(),
                  obj.id);
        if (facesManager.IsGoodForRegcognition(face)) {

        const string image_name = to_string(nframe) + ".jpg";
          auto embedding =
              facesManager.GetFaceEmbedding(face.GetAlignFace(frame));
          auto recog = dbmanager.Find(embedding);

          if (recog.second < dbmanager.LowTh()) {
            st = RecognitionStatusTracker::KNOWN;
            faceId = recog.first;
            AddLogLine(log_file,faceId, st,recog.second, image_name);
            imgSaver.EnqueueImage(frame(obj.rect),
                                  captured_frames_folder/image_name);

          } else if (recog.second <= dbmanager.HiTh()) {
            st = RecognitionStatusTracker::DOUBTFUL;
            faceId = recog.first;
            cerr << "DOUBTFUL_FACE FaceId: '" << faceId
                 << "' RecogTh: " << recog.second << endl;
          } else {
            st = RecognitionStatusTracker::UNKNOWN;
            AddLogLine(log_file,faceId, st,recog.second, image_name);
            imgSaver.EnqueueImage(frame(obj.rect),
                                  captured_frames_folder/image_name);
          }
        }
        recogTracker.Update(obj.id, st, faceId);
      } else {
        recogTracker.UpdateLife(obj.id);
      }
    }
    for (const auto &obj : facesTracker.GetRemovedObjects()) {
      if (recogTracker.Exists(obj.id)) {
        auto st = recogTracker.Get(obj.id);
        if (st.status == RecognitionStatusTracker::PENDING ||
            st.status == RecognitionStatusTracker::DOUBTFUL) {
          cout << "NOT RECOGNITION APPLIED" << endl;
        }
        recogTracker.Remove(obj.id);
        //TODO(otre99): register
      }
    }
    tictac.stop();

    Draw(frame, tracked_faces, recogTracker, tictac.getFPS());
    cv::imshow("Faces", frame);
    if (cv::waitKeyEx(1) == 27){
      break;
    }
    ++nframe;

//    writer.write(frame);
  }
}
