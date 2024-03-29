#include "dbmanager.h"
#include "face.h"
#include "faces_manager.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = filesystem;

void DrawLandmarks(cv::Mat &image, const Face &face) {
  cv::circle(image, face.GetLeftEye(), 3, {0, 0, 255}, -1, cv::FILLED);
  cv::circle(image, face.GetRightEye(), 3, {0, 0, 255}, -1, cv::FILLED);
  cv::circle(image, face.GetMouth(), 3, {0, 0, 255}, -1, cv::FILLED);
}

int main(int argc, char *argv[]) {
  DBManager dbmanager;
  FacesManager faceDet;
  filesystem::path folder_path;
  string images_file_path, mpath;
  {
    auto data = LoadJSon(argv[1]);
    mpath = data["db"].get<string>();
    faceDet.Init(data["faces_manager"]);
    if (!dbmanager.Open(mpath, true, faceDet.GetEmbeddingLen())) {
      return -1;
    }

    images_file_path = data["txt_file"].get<string>();
    if (!fs::exists(images_file_path)) {
      cerr << "ERROR: File '" << images_file_path << "' doesn't exists!";
      return -1;
    }
    folder_path = fs::path(data["txt_file"]).parent_path();
  }

  string face_id, img_name;
  ifstream ifile(images_file_path);
  if (!ifile) {
    cerr << "ERROR: Error opening file '" << images_file_path
         << "' for reading!" << endl;
    return 0;
  }

  cv::Mat dp_img(cv::Size(2 * 640, 2 * 320), CV_8UC3, cv::Scalar::all(0.0));

  bool debug = false;
  if (argc == 3) {
    if (string(argv[2]) == "1") {
      debug = true;
    }
  }
  while (ifile >> face_id >> img_name) {
    const string img_path = folder_path / img_name;
    cv::Mat image = cv::imread(img_path);
    if (image.empty()) {
      cerr << "Error opening image: " << img_path << endl;
      cerr << "Skiping this image" << endl;
    }
    faceDet.DetecFaces(image);
    auto dets = faceDet.GetRecentDetections();

    if (dets.empty()) {
      cerr << "Not faces detected in image image: " << img_path << endl;
      cerr << "Skiping this image" << endl;
      continue;
    }

    if (dets.size() > 1) {
      cerr << "More than one face detected in image image: " << img_path
           << endl;
      cerr << "Skiping this image" << endl;

      for (const auto &box : dets) {
        cv::rectangle(image, box.rect, {0, 0, 0}, 3);
      }
      if (debug) {
        double s = 1080.0 / image.cols;
        cv::resize(image, image, {}, s, s);
        cv::imshow("MoreThanOneFace", image);
        cv::waitKey(-1);
      }
      continue;
    }

    cout << "Processing image  '" << img_name << "' FaceID='" << face_id << "'"
         << endl;

    FaceLandmarks land = faceDet.GetFaceLandmarksOnet(image, dets[0].rect);
    //    FaceLandmarks land = faceDet.GetRecentRetinaFaceLandmarks()[0];

    Face face;
    cv::Rect face_rect = RectInsideFrame(dets[0].rect, image);
    face.Init(image, face_rect, land, faceDet.GetAlignMethod(), -1);

    if (!faceDet.IsFrontal(face)) {
      cerr << "Warning: Face in image '" << img_path << "' is not frontal"
           << endl;
      cerr << "  Roll  = " << face.GetRoll() << endl;
      cerr << "  Pitch = " << face.GetPitch() << endl;
      cerr << "  Yaw   = " << face.GetYaw() << endl;
    }

    cv::Mat align_face;
    face.GetAlignFace(image).copyTo(align_face);
    auto embeddings = faceDet.GetFaceEmbedding(align_face);
    if (!dbmanager.AddData(face_id.c_str(), embeddings)) {
      cerr << "Failed adding embedding data to file " << endl;
      return -1;
    }

    if (debug) {
      cv::resize(align_face, dp_img({640 + 160, 160, 320, 320}), {320, 320});
      DrawLandmarks(image, face);
      cv::resize(image(face_rect), dp_img({160, 160, 320, 320}), {320, 320});
      cv::imshow("Faces", dp_img);
      cv::waitKey(-1);
    }
  }
  dbmanager.Close();
  return 0;
}
