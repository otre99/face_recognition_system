#include "dbmanager.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

DBManager::~DBManager() { Close(); }

void DBManager::Init(int metric, float lowTh, float hiTh) {
  metric_ = metric;
  low_th_ = lowTh;
  hi_th_ = hiTh;
}

bool DBManager::Open(const string &mpath, bool write, int32_t embedding_len) {
  if (write == false) {
    iofile_.open(mpath, ios::binary | ios::in);
    if (!iofile_) {
      cerr << "Error opening file " << mpath << " for read " << endl;
      return false;
    }
    iofile_.read(reinterpret_cast<char *>(&embedding_len_), sizeof(int32_t));
    if (embedding_len_ <= 0) {
      cerr << "Wrong 'embedding_len' in " << mpath
           << " file. embedding_len = " << embedding_len << endl;
      return false;
    }
    const size_t RECORD_SIZE = 32 + sizeof(float) * embedding_len_;
    iofile_.seekg(0, ios::end);

    size_t data_size = iofile_.tellg();
    data_size -= sizeof(int32_t);

    if (data_size % RECORD_SIZE != 0) {
      cerr << "Wrong size in file " << mpath << endl;
      return false;
    }

    size_t n = data_size / RECORD_SIZE;
    embedding_data_.clear();
    Data dd;
    dd.embedding.resize(embedding_len_);
    cout << "Found " << n << " entries in file " << mpath << endl;
    iofile_.seekg(sizeof(int32_t), ios::beg);
    char face_id[KEY_SIZE + 1] = {0};
    for (int i = 0; i < n; ++i) {
      iofile_.read(reinterpret_cast<char *>(face_id), KEY_SIZE);
      iofile_.read(reinterpret_cast<char *>(dd.embedding.data()),
                   sizeof(float) * embedding_len_);

      dd.face_id = string(reinterpret_cast<const char *>(face_id));
      embedding_data_.push_back(dd);
    }
    return true;
  }

  if (embedding_len <= 0) {
    cerr << "Wrong value of 'embedding_len' " << endl;
    return false;
  }

  if (filesystem::exists(mpath)) {
    cout << " File '" << mpath
         << "' already exists, so the new data will be added to the end"
         << endl;
    iofile_.open(mpath);
    int32_t n;
    iofile_.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    iofile_.close();
    if (n != embedding_len) {
      cerr << "ERROR: Impossible to add new data to file of '" << mpath
           << "' because the length of the new face embedding differs from "
              "those in the file"
           << endl;
      iofile_.close();
      return false;
    }
    iofile_.open(mpath, ios::binary | ios::app | ios::out);
    if (!iofile_.is_open()) {
      cerr << "Failing opening file: '" << mpath << "'" << endl;
      return false;
    }
  } else {
    cout << "Creating dataset '" << mpath << "'" << endl;
    iofile_.open(mpath, ios::binary | ios::out);
    if (!iofile_) {
      cerr << "Error creating file " << mpath << " for write " << endl;
      return false;
    }
    iofile_.write(reinterpret_cast<char *>(&embedding_len), sizeof(int32_t));
  }
  embedding_len_ = embedding_len;
  return true;
}

void DBManager::Close() {
  if (iofile_.is_open())
    iofile_.close();
}

bool DBManager::AddData(const char *faceId, const vector<float> &data) {
  if (!iofile_.is_open()) {
    cerr << " Not output file!" << endl;
    return false;
  }

  if (strlen(faceId) > KEY_SIZE - 1) {
    cerr << " FaceId lenght must be less than " << KEY_SIZE << endl;
    return false;
  }

  if (data.size() != embedding_len_) {
    cerr << " Wrong size of embedding data " << endl;
    return false;
  }

  char key[KEY_SIZE + 1];
  strncpy(key, faceId, KEY_SIZE);
  iofile_.write(key, KEY_SIZE);
  auto cpy = data;
  Normalize(cpy);
  iofile_.write(reinterpret_cast<const char *>(cpy.data()),
                sizeof(float) * embedding_len_);

  return true;
}

pair<string, float> DBManager::Find(const vector<float> &x) const {
  pair<string, float> l;
  auto funct =
      metric_ == 0 ? DBManager::CalcEuclideanDist : DBManager::CalcCosineDist;
  float score = -1.0;
  string faceId = {};

  float lastValue = numeric_limits<float>::max();
  for (const auto &e : embedding_data_) {
    const float rr = funct(e.embedding, x);
    if (rr < lastValue) {
      score = rr;
      faceId = e.face_id;
      lastValue = rr;
    }
  }
  return {faceId, score};
}

double DBManager::CalcL2Norm(const vector<float> &x) {
  double r = 0;
  for (auto xi : x) {
    r += xi * xi;
  }
  return sqrt(r);
}

void DBManager::Normalize(vector<float> &x) {
  auto r = CalcL2Norm(x);
  for (auto &xi : x) {
    xi /= r;
  }
}

float DBManager::CalcEuclideanDist(const vector<float> &x,
                                   const vector<float> &y) {
  double result = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double delta = x[i] - y[i];
    result += delta * delta;
  }
  return sqrt(result);
}

float DBManager::CalcCosineDist(const vector<float> &x,
                                const vector<float> &y) {
  double result = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    result += x[i] * y[i];
  }
  return 1.0 - result;
}
