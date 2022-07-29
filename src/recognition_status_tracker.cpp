#include "recognition_status_tracker.h"

void RecognitionStatusTracker::Init(int life) { life_ = life; }

void RecognitionStatusTracker::Update(long tracker_id,
                                      RecognitionStatusTracker::Status status,
                                      const string &faceId) {
  if (data_.find(tracker_id) != data_.end()) {
    data_[tracker_id].life = life_;
    data_[tracker_id].status = status;
    data_[tracker_id].faceId = faceId;
  } else {
    data_.insert({tracker_id, {status, life_, faceId}});
  }
}

bool RecognitionStatusTracker::Exists(long tracker_id) const {
  return data_.find(tracker_id) != data_.end();
}

RecognitionStatusTracker::Data RecognitionStatusTracker::Get(long id) const {
  return data_.at(id);
}

void RecognitionStatusTracker::RemoveOldObjects() {
  for (auto iter = data_.begin(); iter != data_.end();) {
    iter->second.life--;
    if (iter->second.life <= 0) {
      iter = data_.erase(iter);
    } else {
      ++iter;
    }
  }
}
