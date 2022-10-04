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

#ifndef ObjectDetection_H
#define ObjectDetection_H

#include <vector>
#include <string>
#include <execution>
#include <set>

#include <opencv2/opencv.hpp>
#include <stdio.h>

// CUDA/TENSOR_RT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <detect_and_track/logging.h>
#include <detect_and_track/utils.h>

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":"    \
                << __LINE__;                                                   \
      assert(0);                                                               \
    }                                                                          \
  }

/**
 * @brief An object that is used to detect objects in images.
 * @details An object that is used to detect objects in images.
 */
class ObjectDetector {
  private:
    // Non Maximum Supression (NMS) parameters
    float nms_tresh_;
    float conf_tresh_;
    size_t max_output_bbox_count_;

    // Model parameters
    std::string path_to_engine_;
    int image_size_;
    int input_size_;
    int output_size_;
    int buffer_size_;
    int num_classes_;

    // TensorRT primitives 
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *context_;
    nvinfer1::IRuntime *runtime_;

    //Logger
    sample::Logger gLogger_;

    // CPU/GPU data stream
    cudaStream_t stream_;

    // Buffers
    std::vector<void *> buffers_;
    std::shared_ptr<float[]> input_data_;
    std::shared_ptr<float[]> output_data_;

    void prepareEngine();
    size_t getSizeByDim(const nvinfer1::Dims&);
    void preprocessImage(cv::Mat&);
    void sendBufferToGPU();
    void getBufferFromGPU();
    void inferNetwork();
    virtual void nonMaximumSuppression(std::vector<std::vector<BoundingBox>>&);

  public:
    ObjectDetector();
    ObjectDetector(std::string,
                   float,
                   float,
                   size_t,
                   int,
                   int,
                   int);
    ~ObjectDetector();
    void detectObjects(cv::Mat, std::vector<std::vector<BoundingBox>>&);
};

/**
 * @brief An object that is used to detect objects in images.
 * @details An object that is used to detect objects in images.
 */
/** class ObjectDetectorRotation : public ObjectDetector {
  private:
    void nonMaximumSuppression(std::vector<std::vector<BoundingBox>>&) override;

  public:
    ObjectDetectorRotation();
    ObjectDetectorRotation(std::string,
                   float,
                   float,
                   size_t,
                   int,
                   int,
                   int);
    ~ObjectDetectorRotation();
}; **/

#endif