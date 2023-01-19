#include <detect_and_track/ROS2Wrappers.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ROSDetectTrack2DAndLocate>());
  rclcpp::shutdown();
  return 0;
}