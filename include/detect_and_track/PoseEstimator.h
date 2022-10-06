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
 * @details This object implements different methods to estimate the position and distance of 
 * objects detected within RGB-D images. It also allows the user to choose between different
 * camera model which we detail bellow: \n
 *  - "pin_hole" model, a simple model that does not account for lens deformations.
 *  In most situations this model is sufficient. \n
 *  - "plumb_blob" model, a more complicated model that adds lens deformation to
 *  the Pin-Hole model. This model is slower, as to compute its inverse we need to
 * estimate it through a short gradient descent. \n
 * To estimate the distance and position we then provide 2 different modes: \n 
 *  - "center": this mode takes the distance at the center of the bounding box
 *  to estimate the distance and position of the objects. It's fast, but may be
 *  wrong on thin objects like drones where the center of the bounding can be empty. \n
 *  - "min_distance": this mode takes the filtered minimal distance to estimate the
 *  distance to the center of the dected object. This is will create an offset, as
 *  the distance measured is the smallest, and is sensitive to obstructions. However,
 *  it always ensure that the measured distance is the one of the object. \n 
 */
class PoseEstimator {
  private:
    float rejection_threshold_;
    float keep_threshold_;
    int image_height_;
    int image_width_;
    int distortion_model_;
    int position_mode_;

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