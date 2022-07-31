#ifndef RECOGNITION_STATUS_TRACKER_H_
#define RECOGNITION_STATUS_TRACKER_H_
#include <unordered_map>

using namespace std;

class RecognitionStatusTracker {
public:
  enum Status { PENDING = 0, UNKNOWN = 1, KNOWN = 2, DOUBTFUL = 3 };
  struct Data {
    Status status = PENDING;
    int life = 10;
    string faceId = {};
  };
  RecognitionStatusTracker() = default;

  void Init(int life);
  void Update(long tracker_id, RecognitionStatusTracker::Status status,
              const string &faceId);
  void UpdateLife(long tracker_id);
  bool Exists(long tracker_id) const;
  Data Get(long id) const;
  void RemoveOldObjects();

private:
  std::unordered_map<long, Data> data_;
  int life_;
};

#endif // RECOGNITION_STATUS_TRACKER_H_
