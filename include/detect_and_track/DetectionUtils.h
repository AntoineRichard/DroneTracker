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

#include <detect_and_track/ObjectDetection.h>
#include <detect_and_track/PoseEstimator.h>
#include <detect_and_track/Tracker.h>
#include <detect_and_track/utils.h>

#include <opencv2/opencv.hpp>


class Detect {
  protected:
    // Image parameters
    int image_size_;
    int image_rows_;
    int image_cols_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;

    //Profiling variables
#ifdef PROFILE
    std::chrono::time_point start_image_;
    std::chrono::time_point end_image_;
    std::chrono::time_point start_detection_;
    std::chrono::time_point end_detection_;
#endif

    // Object detector parameters
    int num_classes_;
    std::vector<std::string> class_map_;

    ObjectDetector* OD_;

  public:
    Detect();
    Detect(GlobalParameters&, DetectionParameters&, NMSParameters&);
    void buildDetect(GlobalParameters&, DetectionParameters&, NMSParameters&);
    ~Detect();

    void detectObjects(const cv::Mat&, std::vector<std::vector<BoundingBox>>&);
    void generateDetectionImage(cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);
    void printProfilingDetection();
    void applyOnFolder(std::string, std::string, bool, bool, bool);
    void applyOnVideo(std::string, std::string, bool, bool, bool);
};


class Locate {
  protected:
    //Profiling variables
#ifdef PROFILE
    std::chrono::time_point start_distance_;
    std::chrono::time_point end_distance_;
    std::chrono::time_point start_position_;
    std::chrono::time_point end_position_;
#endif

    PoseEstimator* PE_;

  public:
    Locate();
    Locate(GlobalParameters&, LocalizationParameters&, CameraParameters&);
    void buildLocate(GlobalParameters&, LocalizationParameters&, CameraParameters&);
    ~Locate();

    void locate(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&,
                std::vector<std::vector<float>>&, std::vector<std::vector<std::vector<float>>>&);
    void locate(const cv::Mat&, const std::vector<std::map<unsigned int, std::vector<float>>>&,
                std::vector<std::map<unsigned int, float>>&, std::vector<std::map<unsigned int, std::vector<float>>>&);
    void updateCameraInfo(const std::vector<float>&, const std::vector<float>&);
    void printProfilingLocalization();
};


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
    std::chrono::time_point start_tracking_;
    std::chrono::time_point end_tracking_;
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
    ~Track2D();

    void track(const std::vector<std::vector<BoundingBox>>&,
              std::vector<std::map<unsigned int, std::vector<float>>>&);
    void track(const std::vector<std::vector<BoundingBox>>&,
              std::vector<std::map<unsigned int, std::vector<float>>>&,
              const float&);
    void generateTrackingImage(cv::Mat&, const std::vector<std::map<unsigned int, std::vector<float>>>);
    void printProfilingTracking();
};


class DetectAndLocate : public Detect, public Locate {
  public:
    DetectAndLocate();
    DetectAndLocate(GlobalParameters&, DetectionParameters&, NMSParameters&,
                    LocalizationParameters&, CameraParameters&);
    void applyOnFolder(std::string, std::string, bool, bool, bool);
    void applyOnVideo(std::string, std::string, bool, bool, bool);
};


class DetectAndTrack2D : public Detect, public Track2D {
  public:
    DetectAndTrack2D();
    DetectAndTrack2D(GlobalParameters&, DetectionParameters&, NMSParameters&,
                     KalmanParameters&, TrackingParameters&, BBoxRejectionParameters&);
    void applyOnFolder(std::string, std::string, bool, bool, bool);
    void applyOnVideo(std::string, std::string, bool, bool, bool);
};


class DetectTrack2DAndLocate : public Detect, public Locate, public Track2D {
  public:
    DetectTrack2DAndLocate();
    DetectTrack2DAndLocate(GlobalParameters&, DetectionParameters&, NMSParameters&,
                           KalmanParameters&, TrackingParameters&, BBoxRejectionParameters&,
                           LocalizationParameters&, CameraParameters&);
    void applyOnFolder(std::string, std::string, bool, bool, bool);
    void applyOnVideo(std::string, std::string, bool, bool, bool);
};
#endif