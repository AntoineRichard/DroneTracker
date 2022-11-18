#include <detect_and_track/ROSWrappers.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetectAndTrack2D rdt2d;
  ros::spin();
  return 0;
}