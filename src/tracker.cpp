#include "tracker.h"
#include "kuhnmunkres.h"
#include <opencv2/opencv.hpp>

namespace {
constexpr int NEW_OBJ_PENDING = -1;
constexpr int NEW_OBJ_ADDED = -2;
constexpr int DEFAULT_USER_ID = std::numeric_limits<int>::min();
} // namespace

void Tracker::Init(int frames_to_count, int frames_to_discard, float iou_th) {

  tracked_objects_.clear();
  frames_to_count_ = frames_to_count;
  frames_to_discard_ = frames_to_discard;
  iou_th_ = iou_th;
  ncount_ = 0;
  removed_count_ = 0;
}

void Tracker::Reset() {
  tracked_objects_.clear();
  ncount_ = 0;
  removed_count_ = 0;
}

void Tracker::Process(const std::vector<BBox> &objects, int obj_label) {
  vector<TrackedObject> new_objects;
  TrackedObject obj;
  for (size_t i = 0; i < objects.size(); ++i) {
    const auto &o = objects[i];
    if (o.label != obj_label)
      continue;
    obj.id = NEW_OBJ_PENDING;
    obj.last_frame = 0;
    obj.frames_count = 1;
    obj.rect = o.rect;
    obj.score = o.score;
    obj.user_id = DEFAULT_USER_ID;
    new_objects.push_back(obj);
  }
  AddNewDetections(new_objects);
}

void Tracker::Process(const vector<BBox> &objects, int obj_label, vector<int> user_ids){
    assert(objects.size()==user_ids.size());

    vector<TrackedObject> new_objects;
    TrackedObject obj;
    for (size_t i = 0; i < objects.size(); ++i) {
      const auto &o = objects[i];
      if (o.label != obj_label)
        continue;
      obj.id = NEW_OBJ_PENDING;
      obj.last_frame = 0;
      obj.frames_count = 1;
      obj.rect = o.rect;
      obj.score = o.score;
      obj.user_id = user_ids[i];
      new_objects.push_back(obj);
    }
    AddNewDetections(new_objects);
}

ulong Tracker::GenNewUniqueId() { return std::max(ncount_++, ulong(0)); }

void Tracker::AddNewDetections(vector<TrackedObject> &new_objects) {

  if (tracked_objects_.empty()) {
    for (auto &o : new_objects) {
      o.id = GenNewUniqueId();
      tracked_objects_.push_back(o);
    }
    return;
  }

  for (auto &o : tracked_objects_){
    o.last_frame += 1;
  }

  removed_objects_.clear();
  if (!new_objects.empty()) {
    cv::Mat cost_matrix(tracked_objects_.size(), new_objects.size(), CV_32F);
    for (size_t i = 0; i < tracked_objects_.size(); ++i) {
      auto ptr = cost_matrix.ptr<float>(i);
      for (size_t j = 0; j < new_objects.size(); j++) {
        ptr[j] = Distance(tracked_objects_[i], new_objects[j]);
      }
    }

    auto m = KuhnMunkres().Solve(cost_matrix);
    const float TH = 1.0 - iou_th_;

    for (size_t i = 0; i < tracked_objects_.size(); ++i) {
      const int j = m[i];
      const float th = cost_matrix.ptr<float>(i)[j];
      if (j < new_objects.size() && th < TH) {
        tracked_objects_[i].rect = new_objects[j].rect;
        tracked_objects_[i].user_id = new_objects[j].user_id;
        tracked_objects_[i].score = new_objects[j].score;
        tracked_objects_[i].last_frame = 0;
        new_objects[j].id = NEW_OBJ_ADDED;
      }
    }
  }

  vector<TrackedObject> new_tracked_objs;
  for (auto &obj : tracked_objects_) {
    obj.frames_count += 1;
    if (obj.last_frame < frames_to_discard_) {
      new_tracked_objs.push_back(obj);
    } else {
      if (IsAcceptableDetection(obj)) {
        removed_objects_.push_back(obj);
      }
    }
  }

  for (auto &obj : new_objects) {
    if (obj.id == NEW_OBJ_PENDING) {
      obj.id = GenNewUniqueId();
      new_tracked_objs.push_back(obj);
    }
  }
  tracked_objects_ = new_tracked_objs;
}

float Tracker::Distance(const TrackedObject &obj1, const TrackedObject &obj2) {
  const float iou = static_cast<float>((obj1.rect & obj2.rect).area()) /
                    (obj1.rect | obj2.rect).area();
  return 1.0 - iou;
}
