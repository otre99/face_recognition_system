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
  struct Data {
    char *face_id[KEY_SIZE];
    vector<float> embedding;
  };

public:
  DBManager() = default;
  bool Open(const string &mpath, bool write = false,
            int32_t embedding_len = -1);
  void Close();

  bool AddData(const char *faceId, const vector<float> &data);
  ~DBManager();

private:
  fstream iofile_;
  vector<Data> embedding_data_;
  int32_t embedding_len_{-1};

  double CalcL2Norm(const vector<float> &x);
  void Normalize(vector<float> &x);
  float CalcEuclideanDist(const vector<float> &x, const vector<float> &y);
  float CalcCosineDist(const vector<float> &x, const vector<float> &y);

};

#endif // DBMANAGER_H_
