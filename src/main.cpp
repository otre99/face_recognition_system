#include "dbmanager.h"
#include "draw_utils.h"
#include "faces_manager.h"
#include "io_utils.h"
#include "recognition_status_tracker.h"
#include "tracker.h"
#include <chrono>
#include <iostream>
#include "image_saver.h"
#include <opencv2/opencv.hpp>
using namespace std;

DrawUtils drawer;

string GetMsgToDisply(const RecognitionStatusTracker::Data &d) {
  switch (d.status) {
  case RecognitionStatusTracker::DOUBTFUL:
    return "DOUBTFUL";
  case RecognitionStatusTracker::UNKNOWN:
    return "UNKNOWN";
  case RecognitionStatusTracker::PENDING:
    return "PENDING";
  default: // RecognitionStatusTracker::KNOWN:
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
      const auto d = recogTracker.Get(obj.id);
      drawer.DrawTrackedObj(image, obj, GetMsgToDisply(d));
    }
  }
}

using Status = RecognitionStatusTracker::Status;

int main(int argc, char *argv[]) {
  FacesManager facesManager;
  DBManager dbmanager;
  string video_input = "/dev/video0";
  RecognitionStatusTracker recogTracker;
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
  while (cap.read(frame)) {

    tictac.start();
    const auto tracked_faces = facesManager.Process(frame);
    for (const auto &obj : tracked_faces) {
      if (obj.last_frame != 0)
        continue;

      Status st = RecognitionStatusTracker::PENDING;
      bool need_recog;
      if (!recogTracker.Exists(obj.id)) {
        need_recog = true;
      } else {
        st = recogTracker.Get(obj.id).status;
        need_recog = (st == RecognitionStatusTracker::DOUBTFUL) ||
                     (st == RecognitionStatusTracker::PENDING);
      }

      string faceId = {};
      if (need_recog) {
        lands = facesManager.GetFaceLandmarksOnet(frame, obj.rect);
        face.Init(frame, obj.rect, lands, facesManager.GetAlignMethod(),
                  obj.id);
        if (facesManager.IsGoodForRegcognition(face)) {
          auto embedding =
              facesManager.GetFaceEmbedding(face.GetAlignFace(frame));
          auto recog = dbmanager.Find(embedding);
          cout << "ID " << recog.first << " " << recog.second << endl;
          if (recog.second < dbmanager.LowTh()) {
            st = RecognitionStatusTracker::KNOWN;
            faceId = recog.first;
            std::time_t end_time = std::chrono::system_clock::to_time_t(
                chrono::system_clock::now());
            cout << "Face recognized " << faceId
                 << " Time: " << std::ctime(&end_time) << endl;
            imgSaver.EnqueueImage(face.GetAlignFace(frame), std::to_string(nframe)+".jpg");
          } else if (recog.second <= dbmanager.HiTh()) {
            st = RecognitionStatusTracker::DOUBTFUL;
          } else {
            st = RecognitionStatusTracker::UNKNOWN;
              cout << facesManager.GetAlignMethod() << endl;
            imgSaver.EnqueueImage(face.GetAlignFace(frame), std::to_string(nframe)+".jpg");
          }
        }
      }
      recogTracker.Update(obj.id, st, faceId);
    }
    recogTracker.RemoveOldObjects();
    tictac.stop();

    Draw(frame, tracked_faces, recogTracker, tictac.getFPS());
    cv::imshow("faces", frame);
    cv::waitKeyEx(1);
    ++nframe;
  }
}
