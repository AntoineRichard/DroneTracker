/**
 * @file PoseEstimator.h
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief The header of the pose estimation class.
 * @details This file implements a simple algorithm to estimate the position of objects inside bounding boxes.
 */

#ifndef PoseEstimator_H
#define PoseEstimator_H

#include <vector>
#include <execution>
#include <opencv2/opencv.hpp>
#include <detect_and_track/utils.h>
#include <stdio.h>

/**
 * @brief Estimates the position of detected objects.
 * @details This class uses the bounding box and a depth image to estimate the relative position of objects inside it.
 * To estimate the position, the algorithm calculates the distance between every point inside the bounding box and the center of the camera.
 * For this purpose two models can be used, an inverse pinhole, or an inverse plumb blob. This is set through compilation flags.
 * -DBROWNCONRADY enables the plumb blob model. This is not recommended by default it requires a gradient descent to compute the inverse of this model.
 * Hence it is heavier than the pinhole.
 * Then, the 5% closest points are removed as considered as potential outliers (too close, noise in the camera).
 * Of the remaining points, the 10% smallest are then averaged to get the distance to the object.
 * The distance is then used inside a pin-hole model as follows:
 * ix = x * fx + cx
 * iy = y * fy + cy
 * x = ix * z
 * y = iy * z
 * d = sqrt(x*x + y*y + z*z)
 * d = z*sqrt(ix*ix + iy*iy + 1)
 * z = d/sqrt(ix*ix + iy*iy + 1)
 */
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
    std::vector<std::map<unsigned int, float>> extractDistanceFromDepth(const cv::Mat&, const std::vector<std::map<unsigned int, std::vector<float>>>&);
    std::vector<std::vector<std::vector<float>>> estimatePosition(const std::vector<std::vector<float>>& , const std::vector<std::vector<BoundingBox>>&);
    std::vector<std::map<unsigned int, std::vector<float>>> estimatePosition(const std::vector<std::map<unsigned int, float>>& , const std::vector<std::map<unsigned int, std::vector<float>>>&);
    void deprojectPixel2PointBrownConrady(const float&, const std::vector<float>&, std::vector<float>&);
    void deprojectPixel2PointPinHole(const float&, const std::vector<float>&, std::vector<float>& );
    void distancePixel2PointBrownConrady(const float&, const std::vector<float>&, std::vector<float>&);
    float getDistance(const cv::Mat&, const float&, const float&, const float&, const float&);
    void distancePixel2PointPinHole(const float&, const std::vector<float>&, std::vector<float>& );
    void projectPixel2PointPinHole(const float&, const float&, const float&, float&, float&);
    void updateCameraParameters(float, float, float, float, std::vector<double>);
};

#endif