#include <string>
#include <vector>
#include <map>

#include <detect_and_track/DetectionUtils.h>

// Custom messages
#include <detect_and_track/msg/bounding_box2_d.hpp>
#include <detect_and_track/msg/bounding_boxes2_d.hpp>
#include <detect_and_track/msg/position_bounding_box2_d.hpp>
#include <detect_and_track/msg/position_bounding_box2_d_array.hpp>
#include <detect_and_track/msg/position_id.hpp>
#include <detect_and_track/msg/position_id_array.hpp>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/string.hpp>



class ROSDetect : public Detect, public rclcpp::Node {
  protected:
    image_transport::Subscriber image_sub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher detection_pub_;
#endif
    rclcpp::Publisher<detect_and_track::msg::BoundingBoxes2D>::SharedPtr bboxes_pub_;

    // Image parameters
    sensor_msgs::msg::Image::SharedPtr image_ptr_out_;

    virtual void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&);
    void publishDetectionImage(cv::Mat&, std::vector<std::vector<BoundingBox>>&);
    void publishDetections(std::vector<std::vector<BoundingBox>>&, std_msgs::msg::Header&);

  public:
    ROSDetect();
    ~ROSDetect();
};

class ROSDetectAndLocate : public ROSDetect, public Locate { // Should be using virtual classes
  protected:
    image_transport::Subscriber depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
#ifdef PUBLISH_DETECTION_WITH_POSITION
    rclcpp::Publisher<detect_and_track::msg::PositionBoundingBox2DArray>::SharedPtr positions_bboxes_pub_;
#else
    rclcpp::Publisher<detect_and_track::msg::PositionIDArray>::SharedPtr positions_pub_;
#endif
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;

    // Image parameters
    cv::Mat depth_image_;

    virtual void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&) override;
    void depthInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr);
    void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr&);
    void publishDetectionsAndPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::msg::Header&);
    void publishPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::msg::Header&);

  public:
    ROSDetectAndLocate();
    ~ROSDetectAndLocate();
};

class ROSTrack2D : public Track2D, public rclcpp::Node {
  protected:
    image_transport::Subscriber image_sub_;
    rclcpp::Subscription<detect_and_track::msg::BoundingBoxes2D>::SharedPtr bboxes_sub_;
    rclcpp::Publisher<detect_and_track::msg::BoundingBoxes2D>::SharedPtr bboxes_pub_;
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher tracker_pub_;
#endif
    
    // Image parameters
    int num_classes_;
    sensor_msgs::msg::Image::SharedPtr image_ptr_out_;
    std_msgs::msg::Header header_;
    cv::Mat image_;

    // dt update for Kalman 
    float dt_;
    rclcpp::Time t1_;
    rclcpp::Time t2_;   
    void ROSbboxes2bboxes(const detect_and_track::msg::BoundingBoxes2D::SharedPtr, std::vector<std::vector<BoundingBox>>&);
    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::msg::Header&);
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&);
    void bboxesCallback(const detect_and_track::msg::BoundingBoxes2D::SharedPtr);

  public:
    ROSTrack2D();
    ~ROSTrack2D();
};

class ROSDetectAndTrack2D : public ROSDetect, public Track2D { // should be using virtual classes
  protected:
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher tracker_pub_;
#endif
    // dt update for Kalman 
    float dt_;
    rclcpp::Time t1_;
    rclcpp::Time t2_;   

    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::msg::Header&);
    virtual void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&) override;

  public:
    ROSDetectAndTrack2D();
    ~ROSDetectAndTrack2D();
};

class ROSDetectTrack2DAndLocate : public ROSDetectAndLocate, public Track2D { // should be using virtual classes
  protected: 
#ifdef PUBLISH_DETECTION_IMAGE   
    image_transport::Publisher tracker_pub_;
#endif

    // dt update for Kalman 
    float dt_;
    rclcpp::Time t1_;
    rclcpp::Time t2_;   

    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetectionsAndPositions(std::vector<std::map<unsigned int, std::vector<float>>>&,
                                       std::vector<std::map<unsigned int, std::vector<float>>>&,
                                       std_msgs::msg::Header&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::msg::Header&);
    void publishPositions(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::msg::Header&);
    virtual void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&) override;

  public:
    ROSDetectTrack2DAndLocate();
    ~ROSDetectTrack2DAndLocate();
};

/*class ROSDetectTrack2DAndLocateTF : public ROSDetectTrack2DAndLocate {
  protected:
    // Transform parameters
    std::string global_frame_;
    std::string camera_frame_;
    std::vector<std::string> frames_to_track_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener listener_;

    virtual void imageCallback(const sensor_msgs::ImageConstPtr&) override;
  public:
    ROSDetectTrack2DAndLocateTF();
    ~ROSDetectTrack2DAndLocateTF();
};*/
