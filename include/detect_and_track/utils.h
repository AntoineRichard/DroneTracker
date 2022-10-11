/**
 * @file ObjectDetection.cpp
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief Header of the utilitary classes
 * @details This file implements various utilitary classes, such as bounding-box definition,
 * csv writer, or color maps.
 */

#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

/**
 * @brief A color palette.
 * 
 */
static const std::vector<cv::Scalar> ColorPalette{
      cv::Scalar(0, 64, 255),
      cv::Scalar(64, 255, 0),
      cv::Scalar(255, 0, 64),
      cv::Scalar(0, 255, 192),
      cv::Scalar(255, 192, 0),
      cv::Scalar(192, 0, 255),
      cv::Scalar(0, 192, 255),
      cv::Scalar(192, 255, 0),
      cv::Scalar(255, 0, 192),
      cv::Scalar(0, 255, 64),
      cv::Scalar(255, 64, 0),
      cv::Scalar(64, 0, 255),
      cv::Scalar(0, 128, 255),
      cv::Scalar(128, 255, 0),
      cv::Scalar(255, 0, 128),
      cv::Scalar(0, 255, 128),
      cv::Scalar(255, 128, 0),
      cv::Scalar(128, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 0, 255)
  };

/**
 * @brief A structure that stores basic parameters
 */
typedef struct GlobalParameters{
  int image_height; // The height of the image in pixels.
  int image_width; // The width of the image in pixels.
} GlobalParameters;

/**
 * @brief A structure that stores all the parameters related network.
 * 
 */
typedef struct DetectionParameters{
  std::string engine_path; // The path to the TensorRT engine file.
  int num_buffers; // The number of buffers (inputs/outputs) the network has. In most cases it should be 2.
  int num_classes; // The number of classes the network knows.
  std::vector<std::string> class_map; // An ordered vector containing the name of the classes.
} DetectionParameters;

/**
 * @brief A structure that stores all the parameters related to Non-Maximum-Supression.
 * 
 */
typedef struct NMSParameters{
  float nms_thresh;
  float conf_thresh;
  int max_output_bbox_count;
} NMSParameters;

/**
 * @brief A structure that stores all the parameters related to the camera.
 * 
 */
typedef struct CameraParameters{
  std::vector<float> camera_parameters;
  std::vector<float> lens_distortion;
  std::string distortion_model;
} CameraParameters;

/**
 * @brief A structure that stores all the parameters related to the position estimation algorithm.
 * 
 */
typedef struct LocalizationParameters{
  float reject_thresh;
  float keep_thresh;
  std::string mode;
} LocalizationParameters;

/**
 * @brief A structure that stores all the parameters related to the Kalman filters.
 * 
 */
typedef struct KalmanParameters{
  std::vector<float> Q;
  std::vector<float> R;
  bool use_dim;
  bool use_vel;
  float dt;
} KalmanParameters;

/**
 * @brief A structure that stores all the parameters related to the tracking algorithm.
 * 
 */
typedef struct TrackingParameters{
  float distance_thresh;
  float center_thresh;
  float body_ratio;
  float area_thresh;
  int max_frames_to_skip;
} TrackingParameters;

/**
 * @brief A structure that stores all the parameters used to reject bounding boxes before tracking.
 * 
 */
typedef struct BBoxRejectionParameters{
  int min_bbox_height;
  int max_bbox_height;
  int min_bbox_width;
  int max_bbox_width;
} BBoxRejectionParameters;

class csvWriter {
    private:
        std::ofstream ofs_;
        std::string separator_;
        std::string endline_;
        std::string filename_;
        unsigned int max_buffer_size_;
        std::vector<std::vector<float>> buffer_;

        void makeHeader(std::vector<std::string>);
        bool createFile();
        void writeToFile();
        void openFile();
        void closeFile();
    public:
        csvWriter(std::string&, std::string&, std::string&, std::vector<std::string>&, unsigned int&);
        ~csvWriter();

        void flush();
        bool addToBuffer(std::vector<float> data);
};

class RotatedBoundingBox {
    public:
        int class_id_;
        float confidence_;
        float x_; // center x
        float y_; // center y
        float w_; // width
        float h_; // height
        float cos_;
        float sin_;
        float theta_;
        float area_;

        float x1_;
        float x2_;
        float x3_;
        float x4_;
        float y1_;
        float y2_;
        float y3_;
        float y4_;
        
        bool valid_ = true;
        BoundingBox (float*, int&): 
        static bool sortComparisonFunction(const BoundingBox&, const BoundingBox&);
        float calculateIOU (const BoundingBox&);
        void compareWith(BoundingBox&, const float);
};

class BoundingBox {
    public:
        int class_id_;
        float confidence_;
        float x_; // center x
        float y_; // center y
        float w_; // width
        float h_; // height
        float x_min_;
        float x_max_;
        float y_min_;
        float y_max_;
        float area_;
        bool valid_ = true;

        BoundingBox (float* data);
        static bool sortComparisonFunction(const BoundingBox&, const BoundingBox&);
        float calculateIOU (const BoundingBox&);
        void compareWith(BoundingBox&, const float);

};

#endif