#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//
class ImageConverter {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;

    void depthCallback(const sensor_msgs::ImageConstPtr&);

  public:
    ImageConverter();
};

ImageConverter::ImageConverter() : nh_("~"), it_(nh_) {
    image_sub_ = it_.subscribe("/camera/depth/image_rect_raw", 1, &ImageConverter::depthCallback, this);
}

void ImageConverter::depthCallback(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_ptr;
    //ROS_INFO("%s",msg->encoding.c_str());
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    ROS_INFO("row %d, col %d", cv_ptr->image.rows, cv_ptr->image.cols);
    cv::imwrite("/home/antoine/Documents/test.png",cv_ptr->image);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
