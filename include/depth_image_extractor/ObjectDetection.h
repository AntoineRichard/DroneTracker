#ifndef ObjectDetection_H
#define ObjectDetection_H

#include <vector>
#include <string>
#include <execution>
#include <set>

#include <opencv2/opencv.hpp>

// CUDA/TENSOR_RT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <depth_image_extractor/logging.h>
#include <depth_image_extractor/utils.h>

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":"    \
                << __LINE__;                                                   \
      assert(0);                                                               \
    }                                                                          \
  }

enum ObjectClass { CLASS_DRONE = 0, NUM_CLASS = 1 };

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
    void nonMaximumSuppression(std::vector<std::vector<BoundingBox>>&);

  public:
    ObjectDetector();
    ObjectDetector(std::string,
                   float,
                   float,
                   size_t,
                   int,
                   int);
    ~ObjectDetector();
    void detectObjects(cv::Mat, std::vector<std::vector<BoundingBox>>&);
};

#endif