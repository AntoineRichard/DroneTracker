#include <string>
#include <vector>
#include <map>

#include <detect_and_track/DetectionUtils.h>

// Custom messages
#include <detect_and_track/BoundingBox2D.h>
#include <detect_and_track/BoundingBoxes2D.h>
#include <detect_and_track/PositionBoundingBox2D.h>
#include <detect_and_track/PositionBoundingBox2DArray.h>
#include <detect_and_track/PositionID.h>
#include <detect_and_track/PositionIDArray.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>

class ROSTrack2D : public Track2D {
  protected:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber bboxes_sub_;
    ros::Publisher bboxes_pub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher tracker_pub_;
#endif
    
    // Image parameters
    int num_classes_;
    sensor_msgs::ImagePtr image_ptr_out_;
    std_msgs::Header header_;
    cv::Mat image_;

    // dt update for Kalman 
    float dt_;
    ros::Time t1_;
    ros::Time t2_;   
    void ROSbboxes2bboxes(const detect_and_track::BoundingBoxes2DConstPtr&, std::vector<std::vector<BoundingBox>>&);
    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::Header&);
    void imageCallback(const sensor_msgs::ImageConstPtr&);
    void bboxesCallback(const detect_and_track::BoundingBoxes2DConstPtr&);

  public:
    ROSTrack2D();
    ~ROSTrack2D();
};