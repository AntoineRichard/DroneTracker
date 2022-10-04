/**
 * @file ObjectDetection.cpp
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief Header of the object detection
 * @details This file implements an object detection class using TensorRT.
 * This class is meant to be use with the Yolo v5 from ultralytics: https://github.com/ultralytics/yolov5
 */

#ifndef ObjectDetector_H
#define ObjectDetector_H

#include <string>
#include <vector>
#include <map>

#include <detect_and_track/ObjectDetection.h>
#include <detect_and_track/PoseEstimator.h>
#include <detect_and_track/Tracker.h>

#include <opencv2/opencv.hpp>

class Detect {
  private:
    // Image parameters
    int image_size_;
    int image_rows_;
    int image_cols_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;

    // Object detector parameters
    int num_classes_;
    std::vector<std::string> class_map_;

    ObjectDetector* OD_;

  public:
    Detect(std::string, int, int, int, int, float, float, int, bool);
    ~Detect();

    void peformDetection(const cv::Mat&, std::vector<std::vector<BoundingBox>>&);
    void generateDetectionImage(cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);
};

class DetectAndLocate : public Detect {
  private:
    // Image parameters
    cv::Mat depth_image_;

    PoseEstimator* PE_;

  public:
    DetectAndLocate(std::string, int, int, int, int, float, float, int, bool,
                    std::string, float, float, std::vector<float>,
                    std::vector<float>);
    ~DetectAndLocate();

    void locate(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&,
                std::vector<std::vector<float>>&);
    void updateCameraInfo(const std::vector<float>);
};

class DetectAndTrack2D : public Detect {
  private:
    // Tracker parameters
    std::vector<float> Q_;
    std::vector<float> R_;
    float dist_threshold_;
    float center_threshold_;
    float area_threshold_;
    float body_ratio_;
    bool use_dim_;
    bool use_vel_;
    int max_frames_to_skip_;
    float min_bbox_width_;
    float min_bbox_height_;
    float max_bbox_width_;
    float max_bbox_height_;

    // dt update for Kalman 
    float dt_;

    std::vector<Tracker2D*> Trackers_;

    void cast2states(std::vector<std::vector<std::vector<float>>>&,
                     const std::vector<std::vector<BoundingBox>>&);

  public:
    DetectAndTrack2D(std::string, int, int, int, int, float, float, int, bool,
                     std::vector<float>, std::vector<float>, float, float,
                     float, float, bool, bool, int, float, float, float,
                     float);
    ~DetectAndTrack2D();

    void track(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&,
               std::vector<std::map<unsigned int, std::vector<float>>>&);
};

class DetectTrackAndLocate2D : public DetectAndTrack2D {
  private:
    // Image parameters
    cv::Mat depth_image_;

    PoseEstimator* PE_;
  public:
    DetectAndLocate(std::string, int, int, int, int, float, float, int, bool,
                    std::string, float, float, std::vector<float>,
                    std::vector<float>);
    ~DetectAndLocate();

    void locate(const cv::Mat&,
                const std::vector<std::map<unsigned int, std::vector<float>>>&,
                std::vector<std::map<unsigned int, std::vector<float>>>&);
    void updateCameraInfo(const std::vector<float>);
}

#endif