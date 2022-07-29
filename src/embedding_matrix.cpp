#include "dbmanager.h"
#include <fstream>

using namespace std;

inline size_t Index(size_t i, size_t j, size_t cols) { return i * cols + j; }

void SaveCSVFile(const vector<DBManager::Data> &dd, const vector<float> &matrix,
                 const string &fname) {

  ofstream ofile(fname);
  const size_t n = dd.size();
  //ofile << ',';
  for (size_t i = 0; i < n - 1; ++i) {
    ofile << dd[i].face_id << ',';
  }
  ofile << dd[n - 1].face_id << '\n';

  for (size_t i = 0; i < n; ++i) {
    //ofile << dd[i].face_id << ',';
    for (size_t j = 0; j < n - 1; ++j) {
      ofile << matrix[Index(i, j, n)] << ',';
    }
    ofile << matrix[Index(i, n - 1, n)] << '\n';
  }
}

int main(int argc, char *argv[]) {
  DBManager dbmanager;
  dbmanager.Open(argv[1], false, -1);

  const auto data = dbmanager.GetEmbeddingData();
  const size_t n = data.size();
  vector<float> arr(data.size() * data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    arr[Index(i, i, n)] = 0.0;
    for (size_t j = i + 1; j < data.size(); ++j) {
      arr[Index(j, i, n)] = arr[Index(i, j, n)] =
          DBManager::CalcCosineDist(data[i].embedding, data[j].embedding);
    }
  }
  SaveCSVFile(data, arr, "cosine_matrix.csv");

  for (size_t i = 0; i < data.size(); ++i) {
    arr[Index(i, i, n)] = 0.0;
    for (size_t j = i + 1; j < data.size(); ++j) {
      arr[Index(j, i, n)] = arr[Index(i, j, n)] =
          DBManager::CalcEuclideanDist(data[i].embedding, data[j].embedding);
    }
  }
  SaveCSVFile(data, arr, "euclidean_matrix.csv");
}
