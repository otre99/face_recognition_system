#ifndef TRACKER_H_
#define TRACKER_H_
#include "common_utils.h"
#include <opencv2/core.hpp>

using namespace std;

struct TrackedObject {
  long id;
  float score;
  int last_frame;
  int frames_count;
  cv::Rect rect;
  int user_id;
};

class Tracker {
public:
  Tracker() = default;
  void Init(int frames_to_count = 3, int frames_to_discard = 5,
            float iou_th = 0.25);
  void Reset();
  void Process(const vector<BBox> &objects, int obj_label);
  void Process(const vector<BBox> &objects, int obj_label, vector<int> user_ids);

  const std::vector<TrackedObject> &GetTrackedObjects() const {
    return tracked_objects_;
  }
  const std::vector<TrackedObject> &GetRemovedObjects() const {
    return removed_objects_;
  }

  inline bool InCurrentFrame(const TrackedObject &o) const {
    return o.last_frame == 0;
  }

  inline bool IsAcceptableDetection(const TrackedObject &o) const {
    return o.frames_count >= frames_to_count_;
  }

private:
  // metric
  float Distance(const TrackedObject &obj1, const TrackedObject &obj2);

  // methods
  ulong GenNewUniqueId();
  void AddNewDetections(vector<TrackedObject> &dets);

  ulong ncount_{0};
  ulong removed_count_{0};
  int frames_to_count_{3};
  int frames_to_discard_{5};
  float iou_th_{0.25};

  std::vector<TrackedObject> tracked_objects_;
  std::vector<TrackedObject> removed_objects_;
};

#endif // TRACKER_H_
