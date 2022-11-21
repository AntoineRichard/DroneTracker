#include <detect_and_track/DetectionUtils.h>

Track2D::Track2D(){}

Track2D::Track2D(DetectionParameters& det_p, KalmanParameters& kal_p, TrackingParameters& tra_p, BBoxRejectionParameters& bbo_p){
  Q_ = kal_p.Q;
  R_ = kal_p.R;
  dist_threshold_ = tra_p.distance_thresh;
  center_threshold_ = tra_p.center_thresh;
  area_threshold_ = tra_p.area_thresh;
  body_ratio_ = tra_p.body_ratio;
  use_dim_ = kal_p.use_dim;
  use_vel_ = kal_p.use_vel;
  dt_ = tra_p.dt;
  max_frames_to_skip_ = tra_p.max_frames_to_skip;
  min_bbox_width_ = bbo_p.min_bbox_width;
  max_bbox_width_ = bbo_p.max_bbox_width;
  min_bbox_height_ = bbo_p.min_bbox_height;
  max_bbox_height_ = bbo_p.max_bbox_width;
  class_map_ = det_p.class_map;

  for (unsigned int i=0; i<det_p.num_classes; i++){ // Create as many trackers as their are classes
    Trackers_.push_back(new Tracker2D(max_frames_to_skip_, dist_threshold_, center_threshold_,
                      area_threshold_, body_ratio_, dt_, use_dim_,
                      use_vel_, Q_, R_)); 
  }
}

void Track2D::buildTrack2D(DetectionParameters& det_p, KalmanParameters& kal_p, TrackingParameters& tra_p, BBoxRejectionParameters& bbo_p){
  Q_ = kal_p.Q;
  R_ = kal_p.R;
  dist_threshold_ = tra_p.distance_thresh;
  center_threshold_ = tra_p.center_thresh;
  area_threshold_ = tra_p.area_thresh;
  body_ratio_ = tra_p.body_ratio;
  use_dim_ = kal_p.use_dim;
  use_vel_ = kal_p.use_vel;
  dt_ = tra_p.dt;
  max_frames_to_skip_ = tra_p.max_frames_to_skip;
  min_bbox_width_ = bbo_p.min_bbox_width;
  max_bbox_width_ = bbo_p.max_bbox_width;
  min_bbox_height_ = bbo_p.min_bbox_height;
  max_bbox_height_ = bbo_p.max_bbox_width;
  class_map_ = det_p.class_map;

  for (unsigned int i=0; i<det_p.num_classes; i++){ // Create as many trackers as their are classes
    Trackers_.push_back(new Tracker2D(max_frames_to_skip_, dist_threshold_, center_threshold_,
                      area_threshold_, body_ratio_, dt_, use_dim_,
                      use_vel_, Q_, R_)); 
  }
}

Track2D::~Track2D() {
  Trackers_.clear();
}
  

void Track2D::cast2states(std::vector<std::vector<std::vector<float>>>& states, const std::vector<std::vector<BoundingBox>>& bboxes) {
  states.clear();
  std::vector<std::vector<float>> state_vec;
  std::vector<float> state(6);

  for (unsigned int i; i < bboxes.size(); i++) {
    state_vec.clear();
    for (unsigned int j; j < bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      if (bboxes[i][j].h_ > max_bbox_height_) {
        continue;
      }
      if (bboxes[i][j].w_ > max_bbox_width_) {
        continue;
      }
      if (bboxes[i][j].h_ < min_bbox_height_) {
        continue;
      }
      if (bboxes[i][j].w_ < min_bbox_width_) {
        continue;
      }
      state[0] = bboxes[i][j].x_;
      state[1] = bboxes[i][j].y_;
      state[2] = 0;
      state[3] = 0;
      state[4] = bboxes[i][j].w_;
      state[5] = bboxes[i][j].h_;
      state_vec.push_back(state);
      printf("state %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", j, state[0], state[1], state[2], state[3], state[4], state[5]);
    }
    states.push_back(state_vec);
  }
}

void Track2D::track(const std::vector<std::vector<BoundingBox>>& bboxes,
               std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states) {
#ifdef PROFILE
  start_tracking_ = std::chrono::system_clock::now();
#endif 
  std::vector<std::vector<std::vector<float>>> states;
  cast2states(states, bboxes);
  for (unsigned int i=0; i < tracker_states.size(); i++){
    std::vector<std::vector<float>> states_to_track;
    states_to_track = states[i];
    Trackers_[i]->update(dt_, states_to_track);
    Trackers_[i]->getStates(tracker_states[i]);
  }
#ifdef PROFILE
  end_tracking_ = std::chrono::system_clock::now();
#endif
}

void Track2D::track(const std::vector<std::vector<BoundingBox>>& bboxes,
               std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states, const float& dt) {
#ifdef PROFILE
  start_tracking_ = std::chrono::system_clock::now();
#endif 
  std::vector<std::vector<std::vector<float>>> states;
  cast2states(states, bboxes);
  for (unsigned int i=0; i < tracker_states.size(); i++){
    std::vector<std::vector<float>> states_to_track;
    states_to_track = states[i];
    Trackers_[i]->update(dt, states_to_track);
    Trackers_[i]->getStates(tracker_states[i]);
  }
#ifdef PROFILE
  end_tracking_ = std::chrono::system_clock::now();
#endif
}

void Track2D::generateTrackingImage(cv::Mat& image, const std::vector<std::map<unsigned int, std::vector<float>>> tracker_states) {
  for (unsigned int i=0; i < tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      cv::Rect rect(element.second[0] - element.second[4]/2, element.second[1]-element.second[5]/2, element.second[4], element.second[5]);
      cv::rectangle(image, rect, ColorPalette[element.first % 24], 3);
      cv::putText(image, class_map_[i]+" "+std::to_string(element.first), cv::Point(element.second[0]-element.second[4]/2,element.second[1]-element.second[5]/2-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[element.first % 24], 2);
    }
  }
}

void Track2D::printProfilingTracking(){
#ifdef PROFILE
  printf(" - Tracking done in %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_tracking_ - start_tracking_).count());
#endif
}
