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

#ifndef DetectionUtils_H
#define DetectionUtils_H

#include <string>
#include <stdio.h>
#include <vector>
#include <map>

#include <detect_and_track/Tracker.h>
#include <detect_and_track/utils.h>

#include <opencv2/opencv.hpp>

class Track2D {
  protected:
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

    //Profiling variables
#ifdef PROFILE
    std::chrono::time_point<std::chrono::system_clock> start_tracking_;
    std::chrono::time_point<std::chrono::system_clock> end_tracking_;
#endif

    // dt update for Kalman 
    float dt_;
    
    // Object detector parameters
    std::vector<std::string> class_map_;

    std::vector<Tracker2D*> Trackers_;

    void cast2states(std::vector<std::vector<std::vector<float>>>&,
                     const std::vector<std::vector<BoundingBox>>&);

  public:
    Track2D();
    Track2D(DetectionParameters&, KalmanParameters&, TrackingParameters&, BBoxRejectionParameters&);
    void buildTrack2D(DetectionParameters&, KalmanParameters&, TrackingParameters&, BBoxRejectionParameters&);
    ~Track2D();

    void track(const std::vector<std::vector<BoundingBox>>&,
              std::vector<std::map<unsigned int, std::vector<float>>>&);
    void track(const std::vector<std::vector<BoundingBox>>&,
              std::vector<std::map<unsigned int, std::vector<float>>>&,
              const float&);
    void generateTrackingImage(cv::Mat&, const std::vector<std::map<unsigned int, std::vector<float>>>);
    void printProfilingTracking();
};

#endif