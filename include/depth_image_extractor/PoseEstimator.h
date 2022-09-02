#ifndef PoseEstimator_H
#define PoseEstimator_H

#include <vector>
#include <execution>
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

    float fx_;
    float fy_;
    float fx_inv_;
    float fy_inv_;
    float cx_;
    float cy_;
    std::vector<float> K_;

  public:
    PoseEstimator();
    PoseEstimator(float, float, float, float, int, int);
    std::vector<std::vector<float>> extractDistanceFromDepth(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
    std::vector<std::vector<std::vector<float>>> estimatePosition(const std::vector<std::vector<float>>& , const std::vector<std::vector<BoundingBox>>&);
    void deprojectPixel2PointBrownConrady(const float&, const std::vector<float>&, std::vector<float>&);
    void deprojectPixel2PointPinHole(const float&, const std::vector<float>&, std::vector<float>& );
    void distancePixel2PointBrownConrady(const float&, const std::vector<float>&, std::vector<float>&);
    void distancePixel2PointPinHole(const float&, const std::vector<float>&, std::vector<float>& );
    void updateCameraParameters(float, float, float, float, std::vector<double>);
};

#endif