#include <detect_and_track/ROS2Wrappers.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ROSTrack2D>());
  rclcpp::shutdown();
  return 0;
}