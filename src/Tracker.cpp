#include <depth_image_extractor/Tracker.h>

Object::Object() : KF_() {  
}

Object::Object(const unsigned int& id, const float& dt, const bool& use_dim, const bool& use_z, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) : KF_() {
  KF_ = new KalmanFilter(dt, use_dim, use_z, use_vel, Q, R);
  id_ = id;
  nb_frames_ = 0;
  nb_skipped_frames_ = 0;
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



Tracker::Tracker() : HA_() {

}

Tracker::Tracker(const int& max_frames_to_skip, const float& dist_treshold, const float& center_threshold,
                 const float& area_threshold, const float& body_ratio, const float& dt, const bool& use_dim,
                 const bool& use_z, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) : HA_() {

  max_frames_to_skip_ = max_frames_to_skip;
  dist_threshold_ = dist_treshold;
  center_threshold_ = center_threshold;
  area_threshold_ = area_threshold;
  body_ratio_ = body_ratio;

  Q_ = Q;
  R_ = R;
  use_dim_ = use_dim;
  use_z_ = use_z;
  use_vel_ = use_vel;
  dt_ = dt;

  track_id_count_ = 0;
}

void Tracker::getStates(std::map<int, std::vector<float>>& tracks){
  std::vector<float> state(8); 
  for (auto & element : Objects_) {
    element.second->getState(state);
    tracks[element.first] = state;
  }
}

void Tracker::incrementFrame(){
  //for (const std::pair<unsigned int, *Object>& element : Drones_) {
  for (auto & element : Objects_) {
    element.second->newFrame();
  }
}

float Tracker::centroidsError(const std::vector<float>& s1, const std::vector<float>& s2) const {
   return sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]) + (s1[2] - s2[2])*(s1[2] - s2[2]));
}

float Tracker::areaRatio(const std::vector<float>& s1, const std::vector<float>& s2) const {
  return (s1[6]*s1[7]) / (s2[6]*s2[7]);
}

float Tracker::bodyShapeError(const std::vector<float>& s1, const std::vector<float>& s2) const {
  return (s1[6]/s1[7]) / (s2[6]/s2[7]);
}

bool Tracker::isMatch(const std::vector<float>& s1, const std::vector<float>& s2) const {
  ROS_INFO("centroidError: %.3f, %.3f",centroidsError(s1,s2), center_threshold_);
  if (centroidsError(s1,s2) < center_threshold_) {
    ROS_INFO("areaRatio: %.3f, %.3f",areaRatio(s1,s2), area_threshold_);
    if ((1/area_threshold_ < areaRatio(s1,s2)) && (areaRatio(s1,s2) < area_threshold_)) {
      return true;
    }
  }
  return false;
}

void Tracker::computeCost(std::vector<std::vector<double>>& cost, const std::vector<std::vector<float>>& states, std::map<int, int>& tracker_mapping) {
  cost.resize(Objects_.size(), std::vector<double>(states.size()));
  unsigned int i = 0;
  std::vector<float> state(8);
  
  for (auto & element : Objects_) {
    tracker_mapping[i] = element.first;
    for (unsigned int j=0; j<states.size(); j++) {
      element.second->getState(state);
      if (centroidsError(state,states[j]) < 200.f) {
        cost[i][j] = (double) centroidsError(state,states[j]);
      } else {
        cost[i][j] = 1e6;
      }
    }
    i ++;
  }
}

void Tracker::hungarianMatching(std::vector<std::vector<double>>& cost, std::vector<int>& assignments) {
  HA_->Solve(cost, assignments);
}

void Tracker::update(const float& dt, const std::vector<std::vector<float>>& states){
  std::vector<std::vector<double>> cost;
  std::map<int, int> tracks_mapping;
  std::vector<int> assignments;
  std::vector<float> state(8);
  std::vector<int> unassigned_tracks;
  std::vector<int> unassigned_detections;

  // Update the Kalman filters
  ROS_INFO("Updating Kalman filters.");
  for (auto & element : Objects_) {
    element.second->predict(dt);
  }

  // Increment the number of frames.
  incrementFrame();

  if (states.empty()) {
    return;
  }

  // If tracks are empty, then create some new tracks.
  if (Objects_.empty()) {
    ROS_INFO("Creating new tracks.");
    for (unsigned int i=0; i < states.size(); i++) {
      Objects_.insert(std::make_pair(track_id_count_, new Object(track_id_count_, dt_, use_dim_, use_z_, use_vel_, Q_, R_)));
      Objects_[track_id_count_]->setState(states[i]);
      track_id_count_ ++;
    }
  }

  // Match the new detections with the current tracks.
  ROS_INFO("Computing cost");
  for (unsigned int i=0; i < states.size(); i++) {
    ROS_INFO("state %d:", i);
    for (unsigned int j=0; j < states[i].size(); j++) {
      ROS_INFO(" + state %d-%d: %.3f",i,j,states[i][j]);
    }
  }
  unsigned int i = 0;
  for (auto & element : Objects_) {
    ROS_INFO("kalman states %d:", element.first);
    element.second->getState(state);
    for (unsigned int j=0; j < state.size(); j++) {
      ROS_INFO(" + kalman state %d-%d: %.3f",i,j,state[j]);
    } 
    i ++;
  }
  computeCost(cost, states, tracks_mapping); 
  ROS_INFO("Performing matching");
  for (unsigned int i=0; i < cost.size(); i++) {
    ROS_INFO("cost %d:", i);
    for (unsigned int j=0; j < cost[i].size(); j++) {
      ROS_INFO(" + cost %d-%d: %.3f",i,j,cost[i][j]);
    }
  }
  hungarianMatching(cost, assignments);
  ROS_INFO("assignments are:");
  for (unsigned int i=0; i<assignments.size();i++) {
    ROS_INFO(" + matching track raw %d with detection %d.",i,assignments[i]);
    ROS_INFO(" + matching track %d with detection %d.",tracks_mapping[i],assignments[i]);
  }

  // Check for unassigned tracks
  ROS_INFO("Cheking for unassigned tracks.");
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
  ROS_INFO("Cheking for unassigned detections.");
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

  // Remove old tracks
  ROS_INFO("Removing old tracks.");
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

  // Start new tracks
  ROS_INFO("Starting new tracks.");
  for (unsigned int i=0; i < unassigned_detections.size(); i++) {
    Objects_.insert(std::make_pair(track_id_count_, new Object(track_id_count_, dt_, use_dim_, use_z_, use_vel_, Q_, R_)));
    Objects_[track_id_count_]->setState(states[unassigned_detections[i]]);
    track_id_count_ ++;
  }

  // Correct the Kalman filters
  ROS_INFO("Correcting filters.");
  for (unsigned int i=0; i < assignments.size(); i++) {
    if (assignments[i] != -1) {
      Objects_[tracks_mapping[i]]->correct(states[assignments[i]]);
    }
  }
  ROS_INFO("Num tracks %d", track_id_count_); 
}