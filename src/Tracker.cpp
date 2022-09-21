#include <depth_image_extractor/Tracker.h>

Object::Object() : KF_(nullptr) {}

Object::Object(const unsigned int& id, const float& dt, const bool& use_dim,
               const bool& use_vel, const std::vector<float>& Q,
               const std::vector<float>& R) : KF_(nullptr) {
  KF_ = new BaseKalmanFilter(dt, use_dim, use_vel, Q, R);
  id_ = id;
  nb_frames_ = 0;
  nb_skipped_frames_ = 0;
}

Object::~Object() {
  delete KF_;
}

void Object::setState(const std::vector<float>& state){
  KF_->resetFilter(state);
}

void Object::predict() {
  KF_->predict();
}

void Object::predict(const float& dt) {
  KF_->predict(dt);
}

void Object::correct(const std::vector<float>&  Z) {
  KF_->correct(Z);
  nb_skipped_frames_ = 0;
}

void Object::newFrame() {
  nb_frames_ ++;
  nb_skipped_frames_ ++;
}

void Object::getState(std::vector<float>& state) {
  KF_->getState(state);
}

void Object::getUncertainty(std::vector<float>& uncertainty) {
  KF_->getUncertainty(uncertainty);
}

int Object::getSkippedFrames() {
  return nb_skipped_frames_;
}

Object2D::Object2D() {}

Object2D::Object2D(const unsigned int& id, const float& dt, const bool& use_dim,
                   const bool& use_vel, const std::vector<float>& Q,
                   const std::vector<float>& R) {
  KF_ = new KalmanFilter2D(dt, use_dim, use_vel, Q, R);
  std::vector<float> vec(6);
  KF_->resetFilter(vec);
  setState(vec);
  id_ = id;
  nb_frames_ = 0;
  nb_skipped_frames_ = 0;
}

Object3D::Object3D() : Object::Object() {}

Object3D::Object3D(const unsigned int& id, const float& dt, const bool& use_dim,
                   const bool& use_vel, const std::vector<float>& Q,
                   const std::vector<float>& R) {
  KF_ = new KalmanFilter3D(dt, use_dim, use_vel, Q, R);
  id_ = id;
  nb_frames_ = 0;
  nb_skipped_frames_ = 0;
}

BaseTracker::BaseTracker() : HA_() {}


BaseTracker::BaseTracker(const int& max_frames_to_skip, const float& dist_treshold,
                         const float& center_threshold, const float& area_threshold,
                         const float& body_ratio, const float& dt, const bool& use_dim,
                         const bool& use_vel, const std::vector<float>& Q,
                         const std::vector<float>& R) : HA_(nullptr) {

  max_frames_to_skip_ = max_frames_to_skip;
  distance_threshold_ = dist_treshold;
  center_threshold_ = center_threshold;
  area_threshold_ = area_threshold;
  body_ratio_ = body_ratio;

  Q_ = Q;
  R_ = R;
  use_dim_ = use_dim;
  use_vel_ = use_vel;
  dt_ = dt;

  track_id_count_ = 0;
}

Tracker2D::Tracker2D(const int& max_frames_to_skip, const float& dist_treshold,
                     const float& center_threshold, const float& area_threshold,
                     const float& body_ratio, const float& dt, const bool& use_dim,
                     const bool& use_vel, const std::vector<float>& Q,
                     const std::vector<float>& R) : BaseTracker::BaseTracker(max_frames_to_skip,
                     dist_treshold, center_threshold, area_threshold, body_ratio,
                     dt, use_dim, use_vel, Q, R) {}

Tracker3D::Tracker3D(const int& max_frames_to_skip, const float& dist_treshold,
                     const float& center_threshold, const float& area_threshold,
                     const float& body_ratio, const float& dt, const bool& use_dim,
                     const bool& use_vel, const std::vector<float>& Q,
                     const std::vector<float>& R) : BaseTracker::BaseTracker(max_frames_to_skip,
                     dist_treshold, center_threshold, area_threshold, body_ratio,
                     dt, use_dim, use_vel, Q, R) {}


void BaseTracker::getStates(std::map<unsigned int, std::vector<float>>& tracks){
  std::vector<float> state; 
  for (auto & element : Objects_) {
    element.second->getState(state);
    tracks[element.first] = state;
  }
}

void BaseTracker::incrementFrame(){
  //for (const std::pair<unsigned int, *Object>& element : Drones_) {
  for (auto & element : Objects_) {
    element.second->newFrame();
  }
}


bool BaseTracker::isMatch(const std::vector<float>& s1, const std::vector<float>& s2) const {
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d centroidError: %.3f, %.3f\n",__func__,__LINE__,centroidsError(s1,s2), center_threshold_);
#endif
  if (centroidsError(s1,s2) < center_threshold_) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d areaRatio: %.3f, %.3f\n",__func__,__LINE__,areaRatio(s1,s2), area_threshold_);
#endif
    if ((1/area_threshold_ < areaRatio(s1,s2)) && (areaRatio(s1,s2) < area_threshold_)) {
      return true;
    }
  }
  return false;
}

void BaseTracker::computeCost(std::vector<std::vector<double>>& cost, const std::vector<std::vector<float>>& states, std::map<int, int>& tracker_mapping) {
  cost.resize(Objects_.size(), std::vector<double>(states.size()));
  unsigned int i = 0;
  std::vector<float> state(states[0].size());
  
  for (auto & element : Objects_) {
    tracker_mapping[i] = element.first;
    for (unsigned int j=0; j<states.size(); j++) {
      element.second->getState(state);
      if (centroidsError(state,states[j]) < distance_threshold_) {
        cost[i][j] = (double) centroidsError(state,states[j]);
      } else {
        cost[i][j] = 1e6;
      }
    }
    i ++;
  }
}

void BaseTracker::hungarianMatching(std::vector<std::vector<double>>& cost, std::vector<int>& assignments) {
  HA_->Solve(cost, assignments);
}

void BaseTracker::update(const float& dt, const std::vector<std::vector<float>>& states){

  // Update the Kalman filters
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Updating Kalman filters.\n", __func__, __LINE__);
#endif
  for (auto & element : Objects_) {
    element.second->predict(dt);
  }
  
  // Increment the number of frames.
  incrementFrame();
  
  if (states.empty()) {
    return;
  }
  std::vector<std::vector<double>> cost;
  std::map<int, int> tracks_mapping;
  std::vector<int> assignments;
  std::vector<float> state(states[0].size());
  std::vector<int> unassigned_tracks;
  std::vector<int> unassigned_detections;

  // If tracks are empty, then create some new tracks.
  if (Objects_.empty()) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Creating new tracks.\n", __func__, __LINE__);
#endif
    for (unsigned int i=0; i < states.size(); i++) {
      addNewObject();
      Objects_[track_id_count_]->setState(states[i]);
      track_id_count_ ++;
    }
  }

  // Match the new detections with the current tracks.
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Computing cost\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i < states.size(); i++) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d state %d:\n", __func__, __LINE__, i);
#endif
    for (unsigned int j=0; j < states[i].size(); j++) {
#ifdef DEBUG_TRACKER
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d  + state %d-%d: %.3f\n", __func__, __LINE__,i,j,states[i][j]);
#endif
    }
  }
  unsigned int i = 0;
  for (auto & element : Objects_) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d kalman states %d:\n", __func__, __LINE__, element.first);
#endif
    element.second->getState(state);
    for (unsigned int j=0; j < state.size(); j++) {
#ifdef DEBUG_TRACKER
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d  + kalman state %d-%d: %.3f\n", __func__, __LINE__,i,j,state[j]);
#endif
    } 
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d kalman states %d:\n", __func__, __LINE__, element.first);
#endif
    i ++;
  }
  computeCost(cost, states, tracks_mapping); 
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Performing matching\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i < cost.size(); i++) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d cost %d:\n", __func__, __LINE__, i);
#endif
    for (unsigned int j=0; j < cost[i].size(); j++) {
#ifdef DEBUG_TRACKER
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d  + cost %d-%d: %.3f\n", __func__, __LINE__,i,j,cost[i][j]);
#endif
    }
  }
  hungarianMatching(cost, assignments);
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d assignments are:\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i<assignments.size();i++) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d  + matching track raw %d with detection %d.\n", __func__, __LINE__,i,assignments[i]);
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d  + matching track %d with detection %d.\n", __func__, __LINE__,tracks_mapping[i],assignments[i]);
#endif
  }

  // Check for unassigned tracks
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Cheking for unassigned tracks.\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i < assignments.size(); i++) {
    if (assignments[i] == -1) {
      unassigned_tracks.push_back(i);
    } else {
      Objects_[tracks_mapping[i]]->getState(state);
      if (!isMatch(state, states[assignments[i]])) {
        assignments[i] = -1;
        unassigned_tracks.push_back(i);
      }
    }

  }

  // Check for unassigned detections
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Cheking for unassigned detections.\n", __func__, __LINE__);
#endif
  bool assigned;
  for (unsigned int i=0; i < states.size(); i++) {
    assigned = false;
    for (unsigned int j=0; j < assignments.size(); j++) {
      if (assignments[j] == i) {
        assigned = true;
      }
    }
    if (!assigned) {
      unassigned_detections.push_back(i);
    }
  }


  // Start new tracks
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Starting new tracks.\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i < unassigned_detections.size(); i++) {
    addNewObject();
    Objects_[track_id_count_]->setState(states[unassigned_detections[i]]);
    track_id_count_ ++;
  }

  // Correct the Kalman filters
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Correcting filters.\n", __func__, __LINE__);
#endif
  for (unsigned int i=0; i < assignments.size(); i++) {
#ifdef DEBUG_TRACKER
    printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d running for 1\n", __func__, __LINE__);
#endif
    if (assignments[i] != -1) {
#ifdef DEBUG_TRACKER
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d assignment is ok\n", __func__, __LINE__);
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d %d\n", __func__, __LINE__, assignments[i]);
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d %d\n", __func__, __LINE__, tracks_mapping[i]);
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d %d\n", __func__, __LINE__, Objects_[tracks_mapping[i]]->getSkippedFrames());
      printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d %ld\n", __func__, __LINE__, Objects_.size());
#endif
      Objects_[tracks_mapping[i]]->correct(states[assignments[i]]);
    }
  }

  // Remove old tracks
#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Removing old tracks.\n", __func__, __LINE__);
#endif
  for (auto it = Objects_.cbegin(); it != Objects_.cend();)
  {
    if (it->second->getSkippedFrames() > max_frames_to_skip_)
    {
      it = Objects_.erase(it);
    }
    else
    {
      ++it;
    }
  }

#ifdef DEBUG_TRACKER
  printf("\e[1;33m[DEBUG  ]\e[0m Tracker::%s::l%d Num tracks %d\n", __func__, __LINE__, track_id_count_);
#endif
}

float BaseTracker::centroidsError(const std::vector<float>& s1, const std::vector<float>& s2) const {
   return 0.0;
}

float BaseTracker::areaRatio(const std::vector<float>& s1, const std::vector<float>& s2) const {
  return 0.0;
}

void BaseTracker::addNewObject() {
  Objects_.insert(std::make_pair(track_id_count_, new Object(track_id_count_, dt_, use_dim_, use_vel_, Q_, R_)));
}

float Tracker2D::centroidsError(const std::vector<float>& s1, const std::vector<float>& s2) const {
   return sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]));
}

float Tracker2D::areaRatio(const std::vector<float>& s1, const std::vector<float>& s2) const {
  return (s1[4]*s1[5]) / (s2[4]*s2[5]);
}

void Tracker2D::addNewObject() {
  Objects_.insert(std::make_pair(track_id_count_, new Object2D(track_id_count_, dt_, use_dim_, use_vel_, Q_, R_)));
}

float Tracker3D::centroidsError(const std::vector<float>& s1, const std::vector<float>& s2) const {
   return sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]) + (s1[2] - s2[2])*(s1[2] - s2[2]));
}

float Tracker3D::areaRatio(const std::vector<float>& s1, const std::vector<float>& s2) const {
  return (s1[6]*s1[7]) / (s2[6]*s2[7]);
}

void Tracker3D::addNewObject() {
  Objects_.insert(std::make_pair(track_id_count_, new Object3D(track_id_count_, dt_, use_dim_, use_vel_, Q_, R_)));
}