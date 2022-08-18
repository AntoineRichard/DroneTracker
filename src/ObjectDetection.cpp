#include <fstream>
#include <iostream>
#include <sstream>

// CUDA/TENSOR_RT
#include <depth_image_extractor/ObjectDetection.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

ObjectDetector::ObjectDetector() {
}

ObjectDetector::ObjectDetector(std::string path_to_engine,
                   float nms_tresh,
                   float conf_tresh,
                   size_t max_output_bbox_count,
                   int buffer_size,
                   int image_size) {
  path_to_engine_ = path_to_engine;
  nms_tresh_ = nms_tresh;
  conf_tresh_ = conf_tresh;
  max_output_bbox_count_ = max_output_bbox_count;
  buffer_size_ = buffer_size;
  image_size_ = image_size;

  buffers_.resize(buffer_size_);

  prepareEngine();

  input_data_ = std::shared_ptr<float[]>(new float[input_size_]);
  output_data_ = std::shared_ptr<float[]>(new float[output_size_]);
}

ObjectDetector::~ObjectDetector(){
  // Release stream and buffers
  cudaStreamDestroy(stream_);
  CUDA_CHECK(cudaFree(buffers_[0])); //inputs
  CUDA_CHECK(cudaFree(buffers_[1])); //outputs
  // Destroy the engine
  context_->destroy();
  engine_->destroy();
  runtime_->destroy();
}

size_t ObjectDetector::getSizeByDim(const nvinfer1::Dims &dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];

    std::cout << dims.d[i] << ", ";
  }
  std::cout << std::endl;
  return size;
}

void ObjectDetector::prepareEngine() {
  ROS_INFO("engine_path = %s", path_to_engine_.c_str());
  std::ifstream engine_file(path_to_engine_, std::ios::binary);

  if (!engine_file.good()) {
    ROS_ERROR("no such engine file: %s", path_to_engine_.c_str());
    return;
  }

  char *trt_model_stream = nullptr;
  size_t trt_stream_size = 0;
  engine_file.seekg(0, engine_file.end);
  trt_stream_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  trt_model_stream = new char[trt_stream_size];
  assert(trt_model_stream);
  engine_file.read(trt_model_stream, trt_stream_size);
  engine_file.close();

  runtime_ = nvinfer1::createInferRuntime(gLogger_);
  assert(runtime_ != nullptr);
  engine_ = runtime_->deserializeCudaEngine(trt_model_stream, trt_stream_size);
  assert(engine_ != nullptr);
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr);
  if (engine_->getNbBindings() != buffer_size_) {
    ROS_ERROR("engine->getNbBindings() == %d, but should be %d",
              engine_->getNbBindings(), buffer_size_);
  }

  // get sizes of input and output and allocate memory required for input data
  // and for output data
  std::vector<nvinfer1::Dims> input_dims;
  std::vector<nvinfer1::Dims> output_dims;
  for (size_t i = 0; i < engine_->getNbBindings(); ++i) {
    const size_t binding_size =
        getSizeByDim(engine_->getBindingDimensions(i)) * sizeof(float);
    if (binding_size == 0) {
      ROS_ERROR("binding_size == 0");

      delete[] trt_model_stream;
      return;
    }

    cudaMalloc(&buffers_[i], binding_size);
    if (engine_->bindingIsInput(i)) {
      input_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Input layer, size = %lu", binding_size);
      input_size_ = (int) (binding_size / 4);
      ROS_INFO("Creating input buffer of size %d.", input_size_); 
    } else {
      output_dims.emplace_back(engine_->getBindingDimensions(i));
      ROS_INFO("Output layer, size = %lu", binding_size);
      output_size_ = (int) (binding_size / 4);
      ROS_INFO("Creating output buffer of size %d.", output_size_);
    }
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));
  delete[] trt_model_stream;
  ROS_INFO("Engine preparation finished");
}

void ObjectDetector::inferNetwork(){

  context_->enqueue(1, buffers_.data(), stream_, nullptr);

}

void ObjectDetector::sendBufferToGPU(){
  CUDA_CHECK(cudaMemcpyAsync(buffers_[0], input_data_.get(),
                             input_size_ * sizeof(float), cudaMemcpyHostToDevice,
                             stream_));

}

void ObjectDetector::getBufferFromGPU(){
  CUDA_CHECK(cudaMemcpyAsync(output_data_.get(), buffers_[1],
                             output_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);
}

void ObjectDetector::preprocessImage(cv::Mat& image){
  image.convertTo(image, CV_32FC3, 1.f / 255.f);
  int i = 0;
  for (int row = 0; row < image_size_; ++row) {
    for (int col = 0; col < image_size_; ++col) {
      input_data_.get()[i] = image.at<cv::Vec3f>(row, col)[0];
      input_data_.get()[i + image_size_ * image_size_] =
          image.at<cv::Vec3f>(row, col)[1];
      input_data_.get()[i + 2 * image_size_ * image_size_] =
          image.at<cv::Vec3f>(row, col)[2];
      ++i;
    }
  }
}

void ObjectDetector::detectObjects(cv::Mat image, std::vector<std::vector<BoundingBox>>& bboxes){
  bboxes.clear();
  preprocessImage(image);
  auto start_infer = std::chrono::system_clock::now();
  sendBufferToGPU();
  inferNetwork();
  getBufferFromGPU();
  auto end_infer = std::chrono::system_clock::now();
  nonMaximumSuppression(bboxes);
}

void ObjectDetector::nonMaximumSuppression(std::vector<std::vector<BoundingBox>> &bboxes) {
  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    bboxes[c].reserve(output_size_);
  }

  for (int i = 0; i < output_size_; i += 6) {
    const float conf = output_data_.get()[i + 4];
    if (conf > conf_tresh_) {
      assert(conf <= 1.0f);
      const int class_id = 0;
      bboxes[class_id].push_back(BoundingBox(output_data_.get() + i));
    }
  }

  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    std::sort(bboxes[c].begin(), bboxes[c].end(),
              BoundingBox::sortComparisonFunction);
    const size_t bboxes_size = bboxes[c].size();
    size_t valid_count = 0;

    for (size_t i = 0; i < bboxes_size && valid_count < max_output_bbox_count_;
         ++i) {
      if (!bboxes[c][i].valid_) {
        continue;
      }
      for (size_t j = i + 1; j < bboxes_size; ++j) {
        bboxes[c][i].compareWith(bboxes[c][j], nms_tresh_);
      }
      ++valid_count;
    }
  }
}