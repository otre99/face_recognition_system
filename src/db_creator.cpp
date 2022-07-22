#include "dbmanager.h"
#include "face_detection.h"
#include "io_utils.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
namespace fs = filesystem;




int main(int argc, char *argv[]) {
  DBManager dbmanager;
  FaceDetection faceDet;
  filesystem::path folder_path;
  string images_file_path;
  {
    auto data = LoadJSon(argv[1]);
    const string mpath = data["db"].get<string>();
    const int embedding_len = data["embedding_len"].get<int>();
    if (!dbmanager.Open(mpath, true, embedding_len)) {
      return -1;
    }
    faceDet.Init(data["face_detection"]);
    images_file_path = data["txt_file"].get<string>();
    if (!fs::exists(images_file_path)) {
      cerr << "ERROR: File '" << images_file_path << "' doesn't exists!";
      return -1;
    }
    folder_path = fs::path(data["txt_file"]).parent_path();
  }

  string face_id, img_path;
  ifstream ifile(images_file_path);
  if (!ifile) {
    cerr << "ERROR: Error opening file '" << images_file_path
         << "' for reading!" << endl;
    return 0;
  }
  while (ifile >> face_id >> img_path) {
    const string img_path =  folder_path / img_path;
    const cv::Mat image = cv::imread(img_path);
    if (image.empty()){
        cerr << "Error opening image: " << img_path << endl;
        cerr << "Skiping this image" << endl;
    }
    faceDet.DetecFaces(image);
    auto dets = faceDet.GetRecentDetections();

    if ( dets.empty() ){
        cerr << "Not faces detected in image image: " << img_path << endl;
        cerr << "Skiping this image" << endl;
    }

    if ( dets.size() > 1 ) {
        cerr << "More than one face detected in image image: " << img_path << endl;
        cerr << "Skiping this image" << endl;
    }


  }

  return 0;
}
