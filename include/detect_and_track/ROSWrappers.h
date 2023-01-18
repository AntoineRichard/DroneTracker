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
#include <sensor_msgs/Image.h>
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
    sensor_msgs::Image::Ptr image_ptr_out_;

    virtual void imageCallback(const sensor_msgs::Image::ConstPtr&);
    void publishDetectionImage(cv::Mat&, std::vector<std::vector<BoundingBox>>&);
    void publishDetections(std::vector<std::vector<BoundingBox>>&, std_msgs::Header&);

  public:
    ROSDetect();
    ~ROSDetect();
};

class ROSDetectAndLocate : public ROSDetect, public Locate { // Should be using virtual classes
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

    virtual void imageCallback(const sensor_msgs::Image::ConstPtr&) override;
    void depthInfoCallback(const sensor_msgs::CameraInfoConstPtr&);
    void depthCallback(const sensor_msgs::Image::ConstPtr&);
    void publishDetectionsAndPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::Header&);
    void publishPositions(std::vector<std::vector<BoundingBox>>&, std::vector<std::vector<std::vector<float>>>&, std_msgs::Header&);

  public:
    ROSDetectAndLocate();
    ~ROSDetectAndLocate();
};

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
    sensor_msgs::Image::Ptr image_ptr_out_;
    std_msgs::Header header_;
    cv::Mat image_;

    // dt update for Kalman 
    float dt_;
    ros::Time t1_;
    ros::Time t2_;   
    void ROSbboxes2bboxes(const detect_and_track::BoundingBoxes2D::ConstPtr&, std::vector<std::vector<BoundingBox>>&);
    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::Header&);
    void imageCallback(const sensor_msgs::Image::ConstPtr&);
    void bboxesCallback(const detect_and_track::BoundingBoxes2D::ConstPtr&);

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
    ros::Time t1_;
    ros::Time t2_;   

    void publishTrackingImage(cv::Mat&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>&,
                          std_msgs::Header&);
    virtual void imageCallback(const sensor_msgs::Image::ConstPtr&) override;

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
    virtual void imageCallback(const sensor_msgs::Image::ConstPtr&) override;

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
