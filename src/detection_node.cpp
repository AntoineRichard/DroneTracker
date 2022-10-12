#include <string>
#include <vector>
#include <map>

#include <detect_and_track/DetectionUtils.h>

#include <detect_and_track/BoundingBox2D.h>
#include <detect_and_track/BoundingBoxes2D.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>


class ROSDetect : public Detect {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher detection_pub_;
#endif
    ros::Publisher bboxes_pub_;

    // Image parameters
    sensor_msgs::ImagePtr image_ptr_out_;

    void imageCallback(const sensor_msgs::ImageConstPtr&);

  public:
    ROSDetect();
    ~ROSDetect();
};

ROSDetect::ROSDetect() : nh_("~"), it_(nh_), Detect() {
  // Empty structs
  GlobalParameters glo_p;
  DetectionParameters det_p;
  NMSParameters nms_p;
  // Object detector parameters 
  std::string default_path_to_engine("None");
  std::string path_to_engine;
  std::vector<std::string> default_class_map {std::string("object")};
  // Global parameters
  nh_.param("image_rows", glo_p.image_height, 480);
  nh_.param("image_cols", glo_p.image_width, 640);
  // NMS parameters
  nh_.param("nms_tresh", nms_p.nms_thresh,0.45f);
  nh_.param("conf_tresh", nms_p.conf_thresh,0.25f);
  nh_.param("max_output_bbox_count", nms_p.max_output_bbox_count, 1000);
  // Model parameters
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);

  build(glo_p, det_p, nms_p);

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetect::imageCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  detection_pub_ = it_.advertise("/detection/raw_detection", 1);
#endif
  bboxes_pub_ = nh_.advertise<detect_and_track::BoundingBoxes2D>("/detection/bounding_boxes", 1);
}

ROSDetect::~ROSDetect() {
}

void ROSDetect::imageCallback(const sensor_msgs::ImageConstPtr& msg){
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
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
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  detectObjects(image, bboxes);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  printf("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingDetection();
#endif

#ifdef PUBLISH_DETECTION_IMAGE   
  generateDetectionImage(image, bboxes);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  detection_pub_.publish(image_ptr_out_);
#endif

  unsigned int counter = 0;
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;

  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      ros_bbox.min_x = bboxes[i][j].x_min_;
      ros_bbox.min_y = bboxes[i][j].y_min_;
      ros_bbox.height = bboxes[i][j].h_;
      ros_bbox.width = bboxes[i][j].w_;
      ros_bbox.class_id = i;
      ros_bbox.detection_id = counter;
      counter ++;
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = ros::Time::now();
  ros_bboxes.header.frame_id = cv_ptr->header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetect rd;
  ros::spin();
  return 0;
}