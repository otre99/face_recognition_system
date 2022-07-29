#ifndef DBMANAGER_H_
#define DBMANAGER_H_
#include <fstream>
#include <string>
#include <vector>

using namespace std;

double CalcL2Norm(const vector<float> &x);
void Normalize(vector<float> &x);

class DBManager {
  static const int KEY_SIZE = 32;

public:
  struct Data {
    string face_id;
    vector<float> embedding;
  };

  DBManager() = default;
  void Init(int metric, float lowTh, float hiTh);
  bool Open(const string &mpath, bool write = false,
            int32_t embedding_len = -1);
  void Close();
  bool AddData(const char *faceId, const vector<float> &data);
  pair<string, float> Find(const vector<float> &x) const;
  float LowTh() const { return low_th_; }
  float HiTh() const { return hi_th_; }

  ~DBManager();

  const vector<Data> &GetEmbeddingData() const { return embedding_data_; }
  static float CalcEuclideanDist(const vector<float> &x,
                                 const vector<float> &y);
  static float CalcCosineDist(const vector<float> &x, const vector<float> &y);

private:
  fstream iofile_;
  vector<Data> embedding_data_;
  int32_t embedding_len_{-1};

  double CalcL2Norm(const vector<float> &x);
  void Normalize(vector<float> &x);
  int metric_{0};
  float low_th_{0.8};
  float hi_th_{1.0};
};

#endif // DBMANAGER_H_
