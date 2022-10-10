#include <detect_and_track/ObjectDetector.h>

Detect::Detect() : OD_() {}

Detect::Detect(std::string path_to_engine, int image_rows, int image_cols, int num_buffers,
                         int num_classes, float nms_thresh, float conf_thresh, int max_output_bbox_count,
                         std::vector<std::string> class_map) : OD_() {
  // Object detector parameters
  image_rows_ = image_rows;
  image_cols_ = image_cols;
  num_classes_ = num_classes;
  class_map_ = class_map;

  image_size_ = std::max(image_cols_, image_rows_);
  padded_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC3);

  // Object instantiation
  OD_ = new ObjectDetector(path_to_engine, nms_tresh, conf_tresh, max_output_bbox_count, num_buffers, image_size_, num_classes_);
}


Detect::~Detect() {
}

void Detect::padImage(const cv::Mat& image) {
  float r;
  r = (float) image_size_ / std::max(image.rows, image.cols);
  ROS_INFO("%f",r);
  if (r != 1) {
    ROS_ERROR("Not implemented");
  } else {
    padding_rows_ = (image_size_ - image.rows)/2;
    padding_cols_ = (image_size_ - image.cols)/2;
    image.copyTo(padded_image_(cv::Range(padding_rows_,padding_rows_+image.rows),cv::Range(padding_cols_,padding_cols_+image.cols)));
  }
}

void Detect::adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>& bboxes) {
  for (unsigned int i=0; i < bboxes.size(); i++) {
    for (unsigned int j=0; j < bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      bboxes[i][j].x_ -= padding_cols_;
      bboxes[i][j].y_ -= padding_rows_;
      bboxes[i][j].x_min_ = std::max(bboxes[i][j].x_ - bboxes[i][j].w_/2, (float) 0.0);
      bboxes[i][j].x_max_ = std::min(bboxes[i][j].x_ + bboxes[i][j].w_/2, (float) image_cols_);
      bboxes[i][j].y_min_ = std::max(bboxes[i][j].y_ - bboxes[i][j].h_/2, (float) 0.0);
      bboxes[i][j].y_max_ = std::min(bboxes[i][j].y_ + bboxes[i][j].h_/2, (float) image_rows_);
    }
  }
}

void Detect::generateObjectDetectionImage(cv::Mat& image, const std::vector<std::vector<BoundingBox>>& bboxes) {
  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size()+1; j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      if (i == 0) {
        num_bboxes ++;
      }
      const cv::Rect rect(bboxes[i][j].x_min_, bboxes[i][j].y_min_, bboxes[i][j].w_, bboxes[i][j].h_);
      cv::rectangle(image, rect, ColorPalette[0], 3);
      cv::putText(image, class_map_[i], cv::Point(bboxes[i][j].x_min_,bboxes[i][j].y_min_-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[i], 2);
    }
  }
}

void Detect::detectObjects(const cv::Mat& image, std::vector<std::vector<BoundingBox>> bboxes) {
  bboxes.clear();
  bboxes.resize(num_classes_);
#ifdef PROFILE
  start_image_ = std::chrono::system_clock::now();
#endif
  padImage(image);
#ifdef PROFILE
  end_image_ = std::chrono::system_clock::now();
  start_detection_ = std::chrono::system_clock::now();
#endif
  OD_->detectObjects(padded_image_, bboxes);
  adjustBoundingBoxes(bboxes);
#ifdef PROFILE
  end_detection_ = std::chrono::system_clock::now();
#endif
}

void Detect::printProfiling(){
  printf("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection_ - start_image_).count());
  printf(" - Image processing done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_image_ - start_image_).count());
  printf(" - Object detection done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection_ - start_detection_).count());
}

void Detect::apply(std::string source, std::string destination, bool is_video) {}

DetectAndLocate::DetectAndLocate() : Detect(), PE_() {}

DetectAndLocate::DetectAndLocate(std::string path_to_engine, int image_rows, int image_cols, int num_buffers,
                         int num_classes, float nms_thresh, float conf_thresh, int max_output_bbox_count,
                         std::vector<std::string> class_map, float reject_thresh, float keep_thresh,
                         std::vector<float> camera_parameters, std::vector<float> lens_parameters,
                         std::string distortion_model, std::string position_mode) : Detect(path_to_engine,
                         image_rows, image_cols, num_buffers, num_classes, nms_thresh, conf_thresh,
                         max_output_bbox_count, class_map), PE_() {
  // Object instantiation
  PE_ = new PoseEstimator(reject_thresh, keep_thresh, image_rows, image_cols, camera_parameters, lens_parameters, distortion_model, position_mode);
}

void DetectAndLocate::updateCameraInfo(const std::vector<float>& camera_parameters, const std::vector<float>& lens_parameters){
  PE_->updateCameraParameters(camera_parameters, lens_parameters);
}

void DetectAndLocate::locate(const cv::Mat& depth_image, const std::vector<std::vector<BoundingBox>> bboxes,
                             std::vector<std::vector<float>>& distances, std::vector<std::vector<std::vector<float>>>& points){
#ifdef PROFILE
  auto end_detection = std::chrono::system_clock::now();
  auto start_distance = std::chrono::system_clock::now();
#endif
  distances.clear();
  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);
#ifdef PROFILE
  auto end_distance = std::chrono::system_clock::now();
  auto start_position = std::chrono::system_clock::now();
#endif
  points.clear();
  std::vector<std::vector<std::vector<float>>> points;
  points = PE_->estimatePosition(distances, bboxes);
}

void DetectAndLocate::printProfiling(){
  printf("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_position_ - start_image_).count());
  printf(" - Image processing done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_image_ - start_image_).count());
  printf(" - Object detection done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection_ - start_detection_).count());
  printf(" - Distance estimation done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_distance_ - start_distance_).count());
  printf(" - Position estimation done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_position_ - start_position_).count());
}

void DetectAndLocate::apply(std::string source, std::string destination, bool is_video){};

DetectAndTrack2D::DetectAndTrack2D(): Detect() {}

DetectAndTrack2D::DetectAndTrack2D(std::string path_to_engine, int image_rows, int image_cols, int num_buffers,
                         int num_classes, float nms_thresh, float conf_thresh, int max_output_bbox_count,
                         std::vector<std::string> class_map, std::vector<float> Q, std::vector<float> R,
                         float dist_thresh, float center_thresh, float area_thresh, float body_ratio,
                         bool use_dim, bool use_vel, int max_frames_to_skip, float min_bbox_width,
                         float max_bbox_width, float min_bbox_height, float max_bbox_width) : Detect(path_to_engine,
                         image_rows, image_cols, num_buffers, num_classes, nms_thresh, conf_thresh,
                         max_output_bbox_count, class_map){
  Q_ = Q;
  R_ = R;
  dist_threshold_ = dist_thresh;
  center_threshold_ = center_thresh;
  area_threshold_ = area_thresh;
  body_ratio_ = body_ratio_;
  use_dim_ = use_dim;
  use_vel_ = use_vel;
  max_frames_to_skip = max_frames_to_skip_;
  min_bbox_width_ = min_bbox_width;
  max_bbox_width_ = max_bbox_width;
  min_bbox_height_ = min_bbox_height;
  max_bbox_height_ = max_bbox_width;

  for (unsigned int i=0; i<num_classes_; i++){ // Create as many trackers as their are classes
    Trackers_.push_back(new Tracker2D(max_frames_to_skip_, dist_threshold_, center_threshold_,
                      area_threshold_, body_ratio_, dt_, use_dim_,
                      use_vel_, Q_, R_)); 
}
  
void Detect::generateObjectDetectionImage(cv::Mat& image, const std::vector<std::map<unsigned int, std::vector<float>>> tracker_states) {
  for (unsigned int i=0; i < tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      cv::Rect rect(element.second[0] - element.second[4]/2, element.second[1]-element.second[5]/2, element.second[4], element.second[5]);
      cv::rectangle(image, rect, ColorPalette[element.first % 24], 3);
      cv::putText(image, class_map_[i]+" "+std::to_string(element.first), cv::Point(element.second[0]-element.second[4]/2,element.second[1]-element.second[5]/2-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[element.first % 24], 2);
    }
  }

void DetectAndTrack2D::track(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&,
               std::vector<std::map<unsigned int, std::vector<float>>>&);

void DetectAndTrack2D::printProfiling(){
  printf("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection_ - start_image_).count());
  printf(" - Image processing done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_image_ - start_image_).count());
  printf(" - Object detection done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection_ - start_detection_).count());
}

void DetectAndTrack2D::apply(std::string source, std::string destination, bool is_video){};




void ROSDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  t2_ = t1_;
  t1_ = ros::Time::now();
  ros::Duration dt = t1_ - t2_; 
  dt_ = (float) (dt.toSec());
#ifdef PROFILE
  auto start_image = std::chrono::system_clock::now();
#endif
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::Mat image_tracker = image.clone();
  padImage(image);
#ifdef PROFILE
  auto end_image = std::chrono::system_clock::now();
  auto start_detection = std::chrono::system_clock::now();
#endif
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  OD_->detectObjects(padded_image_, bboxes);
  adjustBoundingBoxes(bboxes);
#ifdef PROFILE
  auto end_detection = std::chrono::system_clock::now();
  auto start_tracking = std::chrono::system_clock::now();
#endif 
  std::vector<std::vector<std::vector<float>>> states;
  std::vector<std::map<unsigned int, std::vector<float>>> tracker_states;
  tracker_states.resize(num_classes_);
  cast2states(states, bboxes);
  for (unsigned int i=0; i < num_classes_; i++){
    std::vector<std::vector<float>> states_to_track;
    states_to_track = states[i];
    Trackers_[i]->update(dt_, states_to_track);
    Trackers_[i]->getStates(tracker_states[i]);
  }
#ifdef PROFILE
  auto end_tracking = std::chrono::system_clock::now();
  auto start_distance = std::chrono::system_clock::now();
#endif
  std::vector<std::map<unsigned int, float>> distances;
  distances = PE_->extractDistanceFromDepth(depth_image_, tracker_states);
#ifdef PROFILE
  auto end_distance = std::chrono::system_clock::now();
  auto start_position = std::chrono::system_clock::now();
#endif
  std::vector<std::map<unsigned int, std::vector<float>>> points;
  points = PE_->estimatePosition(distances, tracker_states);
#ifdef PROFILE
  auto end_position = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_tracking - start_image).count());
  ROS_INFO(" - Image processing done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_image - start_image).count());
  ROS_INFO(" - Object detection done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection - start_detection).count());
  ROS_INFO(" - Distance estimation done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
  ROS_INFO(" - Position estimation done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_position - start_position).count());
  ROS_INFO(" - Tracking done in %ld us", std::chrono::duration_cast<std::chrono::microseconds>(end_tracking - start_tracking).count());
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  float u,v;
  PE_->projectPixel2PointPinHole(uav_pose_cam_.pose.position.x, uav_pose_cam_.pose.position.y, uav_pose_cam_.pose.position.z, u, v);
  ROS_INFO("x: %.3f,y: %.3f, z: %.3f", uav_pose_cam_.pose.position.x, uav_pose_cam_.pose.position.y, uav_pose_cam_.pose.position.z);
  ROS_INFO("u: %.3f,v: %.3f", u, v);
  cv:circle(image, cv::Point(u,v), 10, cv::Scalar(255,0,0),cv::FILLED, cv::LINE_8);
  unsigned int num_bboxes = 0;
  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size()+1; j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      if (i == 0) {
        num_bboxes ++;
      }
      const cv::Rect rect(bboxes[i][j].x_min_, bboxes[i][j].y_min_, bboxes[i][j].w_, bboxes[i][j].h_);
      cv::rectangle(image, rect, ColorPalette[0], 3);
      cv::putText(image, class_map_[i], cv::Point(bboxes[i][j].x_min_,bboxes[i][j].y_min_-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[i], 2);
    }
  }

  for (unsigned int i=0; i < tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      cv::Rect rect(element.second[0] - element.second[4]/2, element.second[1]-element.second[5]/2, element.second[4], element.second[5]);
      cv::rectangle(image_tracker, rect, ColorPalette[element.first % 24], 3);
      cv::putText(image_tracker, class_map_[i]+" "+std::to_string(element.first), cv::Point(element.second[0]-element.second[4]/2,element.second[1]-element.second[5]/2-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[element.first % 24], 2);
    }
  }
  ROS_INFO("num detected bboxes: %d", num_bboxes);
  
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  detection_pub_.publish(image_ptr_out_);
    
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  detect_and_track::PositionBoundingBox2DArray ros_bboxes;
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<geometry_msgs::Pose> poses;
  detect_and_track::PositionBoundingBox2D ros_bbox;
  std::vector<detect_and_track::PositionBoundingBox2D> vec_ros_bboxes;
  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.bbox.height = element.second[5];
      ros_bbox.bbox.width = element.second[4];
      ros_bbox.bbox.class_id = i;
      ros_bbox.bbox.detection_id = element.first;
      ros_bbox.position.x = points[i].at(element.first)[0];
      ros_bbox.position.y = points[i].at(element.first)[1];
      ros_bbox.position.z = points[i].at(element.first)[2];
      pose.position.x = points[i].at(element.first)[0];
      pose.position.y = points[i].at(element.first)[1];
      pose.position.z = points[i].at(element.first)[2];
      pose.orientation.w = 0;
      vec_ros_bboxes.push_back(ros_bbox);
      poses.push_back(pose);
    }
  }
  ros_bboxes.header.stamp = cv_ptr->header.stamp;
  ros_bboxes.header.frame_id = cv_ptr->header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;
  pose_array.header = ros_bboxes.header;
  pose_array.poses = poses;
  
  positions_bboxes_pub_.publish(ros_bboxes);
  pose_array_pub_.publish(pose_array);

#else
  unsigned int counter = 0;
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  detect_and_track::PositionIDArray id_positions;
  detect_and_track::PositionID id_position;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;
  std::vector<detect_and_track::PositionID> vec_id_positions;

  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.height = element.second[5];
      ros_bbox.width = element.second[4];
      ros_bbox.class_id = i;
      ros_bbox.detection_id = element.first;
      id_position.position.x = points[i].at(element.first)[0];
      id_position.position.y = points[i].at(element.first)[1];
      id_position.position.z = points[i].at(element.first)[2];
      id_position.detection_id = element.first;
      vec_ros_bboxes.push_back(ros_bbox);
      vec_id_positions.push_back(id_position);
    }
  }
  ros_bboxes.header.stamp = cv_ptr->header.stamp;
  ros_bboxes.header.frame_id = cv_ptr->header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;
  id_positions.header.stamp = cv_ptr->header.stamp;
  id_positions.header.frame_id = cv_ptr->header.frame_id;
  id_positions.positions = vec_id_positions;

  bboxes_pub_.publish(ros_bboxes);
  positions_pub_.publish(id_positions);
#endif
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetector rd;
  ros::spin();
  return 0;
}