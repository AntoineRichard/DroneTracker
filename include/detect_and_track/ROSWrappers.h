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


class ROSDetect : public Detect {
  protected:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher detection_pub_;
#endif
    ros::Publisher bboxes_pub_;

    // Image parameters
    sensor_msgs::ImagePtr image_ptr_out_;

    virtual void imageCallback(const sensor_msgs::ImageConstPtr&);
    void publishDetectionImage(cv::Mat&, std::vector<std::vector<BoundingBox>>&);
    void publishDetections(std::vector<std::vector<BoundingBox>>&, std_msgs::Header&);

  public:
    ROSDetect();
    ~ROSDetect();
};

class ROSDetectAndLocate : public ROSDetect, public Locate {
  protected:
    image_transport::Subscriber depth_sub_;
    ros::Publisher pose_array_pub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher detection_pub_;
#endif
    ros::Subscriber depth_info_sub_;
#ifdef PUBLISH_DETECTION_WITH_POSITION
    ros::Publisher positions_bboxes_pub_;
#else
    ros::Publisher positions_pub_;
#endif

    // Image parameters
    cv::Mat depth_image_;

    virtual void imageCallback(const sensor_msgs::ImageConstPtr&) override;
    void depthInfoCallback(const sensor_msgs::CameraInfoConstPtr&);
    void depthCallback(const sensor_msgs::ImageConstPtr&);
    void publishDetectionsAndPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::Header&);
    void publishPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::Header&);

  public:
    ROSDetectAndLocate();
    ~ROSDetectAndLocate();
};

class ROSDetectTrack2DAndLocate : public ROSDetectAndLocate, public Track2D {
  protected:
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher tracker_pub_;
#endif

    // Image parameters
    cv::Mat depth_image_;

    // dt update for Kalman 
    float dt_;
    ros::Time t1_;
    ros::Time t2_;   

    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetectionsAndPositions(std::vector<std::map<unsigned int, std::vector<float>>>&,
                                       std::vector<std::map<unsigned int, std::vector<float>>>&,
                                       std_msgs::Header&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::Header&);
    void publishPositions(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::Header&);
    virtual void imageCallback(const sensor_msgs::ImageConstPtr&) override;

  public:
    ROSDetectTrack2DAndLocate();
    ~ROSDetectTrack2DAndLocate();
};