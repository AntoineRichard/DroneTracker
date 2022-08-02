#include <string>
#include <vector>

#include <depth_image_extractor/KalmanFilter.h>
#include <depth_image_extractor/PoseEstimator.h>
#include <depth_image_extractor/ObjectDetection.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

# define M_PI           3.14159265358979323846  /* pi */

// State indices
# define PX 0
# define PY 1
# define PZ 2
# define VX 3
# define VY 4
# define VZ 5
# define WIDTH 6
# define AREA 7

class DecisionTree {
  private:
    float center_threshold_; 
    float area_threshold_; 
    float body_ratio_;
    float centroids_error(const std::vector<float>&, const std::vector<float>&);
    float area_ratio(const std::vector<float>&, const std::vector<float>&);
    float bodyShapeError(const std::vector<float>&, const std::vector<float>&);
  public:
    DecisionTree();
    DecisionTree(float, float, float);
    bool isMatch(const std::vector<float>&, const std::vector<float>&);
};

DecisionTree::DecisionTree() {
}

DecisionTree::DecisionTree(float center_threshold, float area_threshold, float body_ratio) {
  center_threshold_ = center_threshold;
  area_threshold_ = area_threshold;
  body_ratio_ = body_ratio;
}

float DecisionTree::centroids_error(const std::vector<float>& s1, const std::vector<float>& s2) {
   return sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]) + (s1[2] - s2[2])*(s1[2] - s2[2]));
}

float DecisionTree::area_ratio(const std::vector<float>& s1, const std::vector<float>& s2) {
  return (s1[6]*s1[7]) / (s2[6]*s2[7]);
}

float DecisionTree::bodyShapeError(const std::vector<float>& s1, const std::vector<float>& s2) {
  return (s1[6]/s1[7]) / (s2[6]/s2[7]);
}

bool DecisionTree::isMatch(const std::vector<float>& s1, const std::vector<float>& s2){
  if (centroids_error(s1,s2) < center_threshold_) {
    if ((1/area_threshold_ < area_ratio(s1,s2)) && (area_ratio(s1,s2) < area_threshold_)) {
      return true;
    }
  }
  return false;
}

/*struct Trace {
  unsigned int id;
  unsigned int frame_id;
  unsigned int skipped_frames;
  unsigned int lost_track;
  float px;
  float py;
  float pz;
  float vx;
  float vy;
  float vz;
  float width;
  float height;
  float p_px;
  float p_py;
  float p_pz;
  float p_vx;
  float p_vy;
  float p_vz;
  float p_height;
  float p_width;
}*/

class Drone {
  private:
    KalmanFilter KF_;
    unsigned int skipped_frame_;
    unsigned int nb_frames_;
    unsigned int id_;
  public:
    Drone();
    Drone(unsigned int, unsigned int, unsigned int);
    void newFrame();
    void update();
    void predict();
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
};

class Tracker {
  private:
    unsigned int track_id_count_;
    std::vector<Drone> Drones_;
  public:
    Tracker();
};



class ROSDetector {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Publisher image_pub_;

    int image_size_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;
    cv::Mat depth_image_;

    sensor_msgs::ImagePtr image_ptr_out_;
    ObjectDetector* OD_;
    PoseEstimator* PE_;

    void imageCallback(const sensor_msgs::ImageConstPtr&);
    void depthCallback(const sensor_msgs::ImageConstPtr&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);

  public:
    ROSDetector();
    ~ROSDetector();
};

ROSDetector::ROSDetector() : nh_("~"), it_(nh_), OD_(), PE_() {
  float nms_tresh, conf_tresh;
  int max_output_bbox_count;
  
  std::string default_path_to_engine("  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);None");
  std::string path_to_engine;


  nh_.param("nms_tresh",nms_tresh,0.45f);
  nh_.param("image_size",image_size_,640);
  nh_.param("conf_tresh",conf_tresh,0.25f);
  nh_.param("max_output_bbox_count", max_output_bbox_count, 1000);
  nh_.param("path_to_engine", path_to_engine, default_path_to_engine);
  OD_ = new ObjectDetector(path_to_engine, nms_tresh, conf_tresh,max_output_bbox_count, 2, image_size_);
  PE_ = new PoseEstimator(0.02, 0.15, 58.0, 87.0, 480, 640);
  
  padded_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC3);
  //depth_image_ = cv::Mat::zeros(image_size_, image_size_, CV_16UC1);

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetector::imageCallback, this);
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetector::depthCallback, this);
  image_pub_ = it_.advertise("/detection/image", 1);
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
    bboxes[0][j].x_min_ = bboxes[0][j].x_ - bboxes[0][j].w_/2;
    bboxes[0][j].x_max_ = bboxes[0][j].x_ + bboxes[0][j].w_/2;
    bboxes[0][j].y_min_ = bboxes[0][j].y_ - bboxes[0][j].h_/2;
    bboxes[0][j].y_max_ = bboxes[0][j].y_ + bboxes[0][j].h_/2;
  }
}

void ROSDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  //cv::Mat image_new;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  auto start = std::chrono::system_clock::now();
  padImage(image);
  std::vector<std::vector<BoundingBox>> bboxes(ObjectClass::NUM_CLASS);
  OD_->detectObjects(padded_image_, bboxes);
  adjustBoundingBoxes(bboxes);
  std::vector<float> distances;
  auto start1 = std::chrono::system_clock::now();
  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);
  std::vector<std::vector<float>> points;
  points = PE_->estimatePosition(distances, bboxes);
  auto end1 = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  ROS_INFO("Depth detection %d us", std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
  for (unsigned int i=0; i<distances.size(); i++){
    ROS_INFO("Distance of object %d : %.3fm",i,distances[i]);
    ROS_INFO("Object %d position: x %.3f y %.3f z %.3f",i,points[i][0],points[i][1],points[i][2]);
  }

  
  for (unsigned int i=0; i<bboxes[0].size(); i++) {
    if (!bboxes[0][i].valid_) {
      continue;
    }
    //const cv::Rect rect(bbox.x_min_ - padding_cols_, bbox.y_min_-padding_rows_, bbox.w_, bbox.h_);
    const cv::Rect rect(bboxes[0][i].x_min_, bboxes[0][i].y_min_, bboxes[0][i].w_, bboxes[0][i].h_);
    cv::rectangle(image, rect, cv::Scalar(0,255,0), 3);
    cv::putText(image, std::to_string(distances[i]), cv::Point(bboxes[0][i].x_min_,bboxes[0][i].y_min_-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0,255,0), 2);
  }
  
  //image.convertTo(image, CV_8UC3, 255.0f);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  image_pub_.publish(image_ptr_out_);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetector rd;
  ros::spin();
  return 0;
}