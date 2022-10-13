#include <detect_and_track/ROSWrappers.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetectAndLocate rdal;
  ros::spin();
  return 0;
}