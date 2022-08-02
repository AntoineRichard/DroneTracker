#ifndef PoseEstimator_H
#define PoseEstimator_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <depth_image_extractor/utils.h>

class PoseEstimator {
  private:
    float rejection_threshold_;
    float keep_threshold_;
    float vertical_fov_;
    float horizontal_fov_;
    int image_height_;
    int image_width_;

    std::vector<float> P_;

  public:
    PoseEstimator();
    PoseEstimator(float, float, float, float, int, int);
    std::vector<float> extractDistanceFromDepth(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
    std::vector<std::vector<float>> estimatePosition(const std::vector<float>& , const std::vector<std::vector<BoundingBox>>&);
};

#endif