#include <string>
#include <vector>
#include <map>

#include <depth_image_extractor/PoseEstimator.h>
#include <depth_image_extractor/ObjectDetection.h>
#include <depth_image_extractor/Tracker.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

# define M_PI           3.14159265358979323846  /* pi */

class ROSDetector {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Publisher detection_pub_;
    image_transport::Publisher tracker_pub_;
    ros::Subscriber depth_info_sub_;
    ros::Subscriber camera_pose_sub_;
    ros::Subscriber drone_pose_sub_;
    ros::Publisher drone_distance_opt_pub_;
    ros::Publisher drone_distance_rs2_pub_;
    ros::Publisher drone_relative_pose_pub_;

    // Image parameters
    int image_size_;
    int image_rows_;
    int image_cols_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;
    cv::Mat depth_image_;
    sensor_msgs::ImagePtr image_ptr_out_;

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
    geometry_msgs::PoseStamped camera_pos_;
    geometry_msgs::PoseStamped drone_pos_;
    float cam2drone_;

    ObjectDetector* OD_;
    PoseEstimator* PE_;
    Tracker* TR_;

    void imageCallback(const sensor_msgs::ImageConstPtr&);
    void depthCallback(const sensor_msgs::ImageConstPtr&);
    void cameraPoseCallback(const geometry_msgs::PoseStampedConstPtr&);
    void dronePoseCallback(const geometry_msgs::PoseStampedConstPtr&);
    void depthInfoCallback(const sensor_msgs::CameraInfoConstPtr&);
    void cast2states(std::vector<std::vector<float>>&,
                     const std::vector<std::vector<BoundingBox>>&,
                     const std::vector<std::vector<float>>&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);

  public:
    ROSDetector();
    ~ROSDetector();
};

ROSDetector::ROSDetector() : nh_("~"), it_(nh_), OD_(), PE_(), TR_() {
  float nms_tresh, conf_tresh;
  int max_output_bbox_count;
  
  std::string default_path_to_engine("None");
  std::string path_to_engine;

  Q_.resize(8);
  R_.resize(8);

  nh_.param("nms_tresh",nms_tresh,0.45f);
  nh_.param("image_size",image_size_,640);
  nh_.param("conf_tresh",conf_tresh,0.25f);
  nh_.param("max_output_bbox_count", max_output_bbox_count, 1000);
  nh_.param("path_to_engine", path_to_engine, default_path_to_engine);
  nh_.param("image_rows", image_rows_, 480);
  nh_.param("image_cols", image_cols_, 640);
  nh_.param("max_frames_to_skip", max_frames_to_skip_, 15);
  nh_.param("dist_threshold", dist_threshold_, 54.0f);
  nh_.param("center_threshold", center_threshold_, 80.0f);
  nh_.param("area_threshold", area_threshold_, 3.0f);
  nh_.param("body_ration", body_ratio_, 0.5f);
  nh_.param("dt", dt_, 0.02f);
  nh_.param("use_dim", use_dim_, true);
  nh_.param("use_z", use_z_, false);
  nh_.param("Q0", Q_[0], 3.0f);
  nh_.param("Q1", Q_[1], 3.0f);
  nh_.param("Q2", Q_[2], 0.3f);
  nh_.param("Q3", Q_[3], 3.0f);
  nh_.param("Q4", Q_[4], 3.0f);
  nh_.param("Q5", Q_[5], 0.3f);
  nh_.param("Q6", Q_[6], 5.0f);
  nh_.param("Q7", Q_[7], 5.0f);
  nh_.param("R0", R_[0], 5.0f);
  nh_.param("R1", R_[1], 5.0f);
  nh_.param("R2", R_[2], 0.3f);
  nh_.param("R3", R_[3], 5.0f);
  nh_.param("R4", R_[4], 5.0f);
  nh_.param("R5", R_[5], 0.3f);
  nh_.param("R6", R_[6], 3.0f);
  nh_.param("R7", R_[7], 5.0f);
  nh_.param("max_bbox_width", max_bbox_width_, 400.0f);
  nh_.param("max_bbox_height", max_bbox_height_, 300.0f);
  nh_.param("min_bbox_width", min_bbox_width_, 60.0f);
  nh_.param("min_bbox_height", min_bbox_height_, 60.0f);

  OD_ = new ObjectDetector(path_to_engine, nms_tresh, conf_tresh,max_output_bbox_count, 2, image_size_);
  PE_ = new PoseEstimator(0.02, 0.15, 58.0, 87.0, image_rows_, image_cols_);
  TR_ = new Tracker(max_frames_to_skip_, dist_threshold_, center_threshold_,
                    area_threshold_, body_ratio_, dt_, use_dim_,
                    use_z_, use_vel_, Q_, R_);
  
  padded_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC3);
  cam2drone_ = 0;

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetector::imageCallback, this);
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetector::depthCallback, this);
  depth_info_sub_ = nh_.subscribe("/camera/aligned_depth_to_color/camera_info", 1, &ROSDetector::depthInfoCallback, this);
  camera_pose_sub_ = nh_.subscribe("/vrpn_client_node/CAM/pose", 1, &ROSDetector::cameraPoseCallback, this);
  drone_pose_sub_ = nh_.subscribe("/vrpn_client_node/UAV/pose", 1, &ROSDetector::dronePoseCallback, this);
  detection_pub_ = it_.advertise("/detection/raw_detection", 1);
  tracker_pub_ = it_.advertise("/detection/tracking", 1);
  drone_relative_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/detection/pose", 1);
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

void ROSDetector::cameraPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  camera_pos_ = *msg;
}

void ROSDetector::dronePoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  drone_pos_ = *msg;
  cam2drone_ = std::sqrt((drone_pos_.pose.position.x - camera_pos_.pose.position.x) * (drone_pos_.pose.position.x - camera_pos_.pose.position.x) +
                        (drone_pos_.pose.position.y - camera_pos_.pose.position.y) * (drone_pos_.pose.position.y - camera_pos_.pose.position.y) +
                        (drone_pos_.pose.position.z - camera_pos_.pose.position.z) * (drone_pos_.pose.position.z - camera_pos_.pose.position.z));
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

void ROSDetector::adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>& bboxes) {
  for (unsigned int j=0; j < bboxes[0].size(); j++) {
      if (!bboxes[0][j].valid_) {
        continue;
      }
    bboxes[0][j].x_ -= padding_cols_;
    bboxes[0][j].y_ -= padding_rows_;
    bboxes[0][j].x_min_ = std::max(bboxes[0][j].x_ - bboxes[0][j].w_/2, (float) 0.0);
    bboxes[0][j].x_max_ = std::min(bboxes[0][j].x_ + bboxes[0][j].w_/2, (float) image_cols_);
    bboxes[0][j].y_min_ = std::max(bboxes[0][j].y_ - bboxes[0][j].h_/2, (float) 0.0);
    bboxes[0][j].y_max_ = std::min(bboxes[0][j].y_ + bboxes[0][j].h_/2, (float) image_rows_);
  }
}

void ROSDetector::cast2states(std::vector<std::vector<float>>& states, const std::vector<std::vector<BoundingBox>>& bboxes, const std::vector<std::vector<float>>& points) {
  states.clear();
  std::vector<float> state(8);

  for (unsigned int i; i < points.size(); i++) {
    if (!bboxes[0][i].valid_) {
      continue;
    }
    if (bboxes[0][i].h_ > max_bbox_height_) {
      continue;
    }
    if (bboxes[0][i].w_ > max_bbox_width_) {
      continue;
    }
    if (bboxes[0][i].h_ < min_bbox_height_) {
      continue;
    }
    if (bboxes[0][i].w_ < min_bbox_width_) {
      continue;
    }
    state[0] = bboxes[0][i].x_;
    state[1] = bboxes[0][i].y_;
    state[2] = points[0][2];
    state[3] = 0;
    state[4] = 0;
    state[5] = 0;
    state[6] = bboxes[0][i].h_;
    state[7] = bboxes[0][i].w_;
    states.push_back(state);
    ROS_INFO("state %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", i, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);
  }
}

void ROSDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  t2_ = t1_;
  t1_ = ros::Time::now();
  ros::Duration dt = t1_ - t2_; 
  dt_ = (float) (dt.toSec());
  ROS_INFO("Delta time: %.3f, %.3f, %.3f, %.3f", t1_.toSec(), t2_.toSec(), dt_, dt.toSec());
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
  std::vector<std::vector<BoundingBox>> bboxes(ObjectClass::NUM_CLASS);
  OD_->detectObjects(padded_image_, bboxes);
  adjustBoundingBoxes(bboxes);
#ifdef PROFILE
  auto end_detection = std::chrono::system_clock::now();
  auto start_distance = std::chrono::system_clock::now();
#endif
  std::vector<float> distances;
  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);
#ifdef PROFILE
  auto end_distance = std::chrono::system_clock::now();
  auto start_position = std::chrono::system_clock::now();
#endif
  std::vector<std::vector<float>> points;
  points = PE_->estimatePosition(distances, bboxes);
#ifdef PROFILE
  auto end_position = std::chrono::system_clock::now();
  auto start_tracking = std::chrono::system_clock::now();
#endif 
  std::vector<std::vector<float>> states;
  cast2states(states, bboxes, points);
  TR_->update(dt_, states);
  std::map<int, std::vector<float>> tracker_states;
  TR_->getStates(tracker_states);
#ifdef PROFILE
  auto end_tracking = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_tracking - start_image).count());
  ROS_INFO(" - Image processing done in %d us", std::chrono::duration_cast<std::chrono::microseconds>(end_image - start_image).count());
  ROS_INFO(" - Object detection done in %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_detection - start_detection).count());
  ROS_INFO(" - Distance estimation done in %d us", std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
  ROS_INFO(" - Position estimation done in %d us", std::chrono::duration_cast<std::chrono::microseconds>(end_position - start_position).count());
  ROS_INFO(" - Tracking done in %d us", std::chrono::duration_cast<std::chrono::microseconds>(end_tracking - start_tracking).count());
#endif
  ROS_INFO("Raw measurments:");
  for (unsigned int i=0; i<distances.size(); i++){
    if (!bboxes[0][i].valid_) {
      continue;
    }
    ROS_INFO(" - Object %d:",i);
    ROS_INFO("   + Estimated distance: %.3f",distances[i]);
    ROS_INFO("   + Measured distance:  %.3f",cam2drone_);
    ROS_INFO("   + Estimated position: x %.3f y %.3f z %.3f",points[i][0],points[i][1],points[i][2]);
  }

  for (unsigned int i=0; i<bboxes[0].size(); i++) {
    if (!bboxes[0][i].valid_) {
      continue;
    }
    //const cv::Rect rect(bbox.x_min_ - padding_cols_, bbox.y_min_-padding_rows_, bbox.w_, bbox.h_);
    const cv::Rect rect(bboxes[0][i].x_min_, bboxes[0][i].y_min_, bboxes[0][i].w_, bboxes[0][i].h_);
    cv::rectangle(image, rect, ColorPalette[0], 3);
    cv::putText(image, std::to_string(distances[i]), cv::Point(bboxes[0][i].x_min_,bboxes[0][i].y_min_-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[0], 2);
  }

  for (auto & element : tracker_states) {
    cv::Rect rect(element.second[0] - element.second[7]/2, element.second[1]-element.second[6]/2, element.second[7], element.second[6]);
    cv::rectangle(image_tracker, rect, ColorPalette[element.first % 24], 3);
    cv::putText(image_tracker, std::to_string(element.first), cv::Point(element.second[0]-element.second[7]/2,element.second[1]-element.second[6]/2-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, ColorPalette[element.first % 24], 2);
  }
  
  //image.convertTo(image, CV_8UC3, 255.0f);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  detection_pub_.publish(image_ptr_out_);
  
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);

  if (points.size() > 0) {
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = points[0][0];
    pose.pose.position.y = points[0][1];
    pose.pose.position.z = points[0][2];
    drone_relative_pose_pub_.publish(pose);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetector rd;
  ros::spin();
  return 0;
}