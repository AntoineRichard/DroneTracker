#include <string>
#include <vector>
#include <map>

#include <depth_image_extractor/PoseEstimator.h>
#include <depth_image_extractor/ObjectDetection.h>
#include <depth_image_extractor/Tracker.h>

#include <depth_image_extractor/BoundingBox2D.h>
#include <depth_image_extractor/BoundingBoxes2D.h>
#include <depth_image_extractor/PositionBoundingBox2D.h>
#include <depth_image_extractor/PositionBoundingBox2DArray.h>
#include <depth_image_extractor/PositionID.h>
#include <depth_image_extractor/PositionIDArray.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class ROSDetector {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Publisher pose_array_pub_;
    ros::Subscriber uav_pose_sub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher detection_pub_;
    image_transport::Publisher tracker_pub_;
#endif
    ros::Subscriber depth_info_sub_;
#ifdef PUBLISH_DETECTION_WITH_POSITION
    ros::Publisher positions_bboxes_pub_;
#else
    ros::Publisher bboxes_pub_;
    ros::Publisher positions_pub_;
#endif

    // Image parameters
    int image_size_;
    int image_rows_;
    int image_cols_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;
    cv::Mat depth_image_;
    sensor_msgs::ImagePtr image_ptr_out_;

    // Object detector parameters
    int num_classes_;
    std::vector<std::string> class_map_;

    // Transform parameters
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener listener_;
    std::string global_frame_;

    // Tracker parameters
    std::vector<float> Q_;
    std::vector<float> R_;
    float dist_threshold_;
    float center_threshold_;
    float area_threshold_;
    float body_ratio_;
    bool use_dim_;
    bool use_z_;
    bool use_vel_;
    int max_frames_to_skip_;
    float min_bbox_width_;
    float min_bbox_height_;
    float max_bbox_width_;
    float max_bbox_height_;

    // dt update for Kalman 
    float dt_;
    ros::Time t1_;
    ros::Time t2_;   

    // Position estimation
    geometry_msgs::PoseStamped uav_pose_;
    geometry_msgs::PoseStamped uav_pose_cam_;

    ObjectDetector* OD_;
    PoseEstimator* PE_;
    std::vector<Tracker2D*> Trackers_;

    void imageCallback(const sensor_msgs::ImageConstPtr&);
    void depthCallback(const sensor_msgs::ImageConstPtr&);
    void depthInfoCallback(const sensor_msgs::CameraInfoConstPtr&);
    void uavPoseCallback(const geometry_msgs::PoseStampedConstPtr&);
    void cast2states(std::vector<std::vector<std::vector<float>>>&,
                     const std::vector<std::vector<BoundingBox>>&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);

  public:
    ROSDetector();
    ~ROSDetector();
};

ROSDetector::ROSDetector() : nh_("~"), listener_(tf_buffer_), it_(nh_), OD_(), PE_() {
  // Object detector parameters
  float nms_tresh, conf_tresh;
  int max_output_bbox_count; 
  std::string default_path_to_engine("None");
  std::string path_to_engine;
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", path_to_engine, default_path_to_engine);
  nh_.param("num_classes", num_classes_, 1);
  nh_.param("class_map", class_map_, default_class_map);
  nh_.param("image_rows", image_rows_, 480);
  nh_.param("image_cols", image_cols_, 640);
  nh_.param("nms_tresh",nms_tresh,0.45f);
  nh_.param("conf_tresh",conf_tresh,0.25f);
  nh_.param("max_output_bbox_count", max_output_bbox_count, 1000);

  // Tracker parameters
  std::vector<float> default_Q {9.0, 9.0, 200.0, 200.0, 5.0, 5.0};
  std::vector<float> default_R {2.0, 2.0, 200.0, 200.0, 2.0, 2.0};
  Q_.resize(6);
  R_.resize(6);
  nh_.param("max_frames_to_skip", max_frames_to_skip_, 15);
  nh_.param("dist_threshold", dist_threshold_, 150.0f);
  nh_.param("center_threshold", center_threshold_, 80.0f);
  nh_.param("area_threshold", area_threshold_, 3.0f);
  nh_.param("body_ratio", body_ratio_, 0.5f);
  nh_.param("dt", dt_, 0.02f);
  nh_.param("use_dim", use_dim_, true);
  nh_.param("use_vel", use_vel_, false);
  nh_.param("Q", Q_, default_Q);
  nh_.param("R", R_, default_R);
  nh_.param("max_bbox_width", max_bbox_width_, 400.0f);
  nh_.param("max_bbox_height", max_bbox_height_, 300.0f);
  nh_.param("min_bbox_width", min_bbox_width_, 60.0f);
  nh_.param("min_bbox_height", min_bbox_height_, 60.0f);
  
  // Global position rejection
  std::vector<std::string> default_tracked_transforms;

  image_size_ = std::max(image_cols_, image_rows_);
  padded_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC3);

  // Object instantiation
  OD_ = new ObjectDetector(path_to_engine, nms_tresh, conf_tresh, max_output_bbox_count, 2, image_size_, num_classes_);
  PE_ = new PoseEstimator(0.02, 0.15, 58.0, 87.0, image_rows_, image_cols_); 
  for (unsigned int i=0; i<num_classes_; i++){ // Create as many trackers as their are classes
    Trackers_.push_back(new Tracker2D(max_frames_to_skip_, dist_threshold_, center_threshold_,
                      area_threshold_, body_ratio_, dt_, use_dim_,
                      use_vel_, Q_, R_)); 
  }

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetector::imageCallback, this);
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetector::depthCallback, this);
  depth_info_sub_ = nh_.subscribe("/camera/aligned_depth_to_color/camera_info", 1, &ROSDetector::depthInfoCallback, this);
  uav_pose_sub_ = nh_.subscribe("/vrpn_client_node/targetUAV/pose", 1, &ROSDetector::uavPoseCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  detection_pub_ = it_.advertise("/detection/raw_detection", 1);
  tracker_pub_ = it_.advertise("/detection/tracking", 1);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  positions_bboxes_pub_ = nh_.advertise<depth_image_extractor::PositionBoundingBox2DArray>("/detection/positions_bboxes",1);
#else
  positions_pub_ = nh_.advertise<depth_image_extractor::PositionIDArray>("/detection/positions",1);
  bboxes_pub_ = nh_.advertise<depth_image_extractor::BoundingBox2D>("/detection/bounding_boxes", 1);
#endif
  pose_array_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection/pose_array", 1);
  t1_ = ros::Time::now();
}

ROSDetector::~ROSDetector() {
}

void ROSDetector::padImage(const cv::Mat& image) {
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

void ROSDetector::depthInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
  PE_->updateCameraParameters(msg->K[2], msg->K[5], msg->K[0], msg->K[4], msg->D);
}

void ROSDetector::depthCallback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv_ptr->image.convertTo(depth_image_, CV_32F, 0.001);
}
  
void ROSDetector::uavPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  uav_pose_ = *msg;
  try {
    //tf_buffer_.lookupTransform("world", "camera_color_optical_frame", ros::Time(0));
    tf_buffer_.transform(uav_pose_, uav_pose_cam_, "camera_color_optical_frame", ros::Duration(1.0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s",ex.what());
  }
}

void ROSDetector::adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>& bboxes) {
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

void ROSDetector::cast2states(std::vector<std::vector<std::vector<float>>>& states, const std::vector<std::vector<BoundingBox>>& bboxes) {
  states.clear();
  std::vector<std::vector<float>> state_vec;
  std::vector<float> state(6);

  for (unsigned int i; i < bboxes.size(); i++) {
    state_vec.clear();
    for (unsigned int j; j < bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      ROS_INFO("w:%d, h:%d", int(bboxes[i][j].w_), int(bboxes[i][j].h_));
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
      ROS_INFO("state %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", j, state[0], state[1], state[2], state[3], state[4], state[5]);
    }
    states.push_back(state_vec);
  }
}

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
  depth_image_extractor::PositionBoundingBox2DArray ros_bboxes;
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<geometry_msgs::Pose> poses;
  depth_image_extractor::PositionBoundingBox2D ros_bbox;
  std::vector<depth_image_extractor::PositionBoundingBox2D> vec_ros_bboxes;
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
  depth_image_extractor::BoundingBoxes2D ros_bboxes;
  depth_image_extractor::BoundingBox2D ros_bbox;
  depth_image_extractor::PositionIDArray id_positions;
  depth_image_extractor::PositionID id_position;
  std::vector<depth_image_extractor::BoundingBox2D> vec_ros_bboxes;
  std::vector<depth_image_extractor::PositionID> vec_id_positions;

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