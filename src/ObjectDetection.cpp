/**
 * @file ObjectDetection.cpp
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief Source code of the object detection
 * @details This file implements an object detection class using TensorRT.
 * This class is meant to be use with the Yolo v5 from ultralytics: https://github.com/ultralytics/yolov5
 * @todo Enable the user to pick the buffer index for input and output.
 * @todo Enable the user to switch between float32 and float16.
 */

#include <fstream>
#include <iostream>
#include <sstream>

// CUDA/TENSOR_RT
#include <detect_and_track/ObjectDetection.h>

/**
 * @brief Constructs an object detector object.
 * @details Construct an object that can be used to detect objects inside images.
 * This code was built to run the Yolov5 implementation from Ultralytics, but it should
 * be able to run about any object detectors. The object detector uses TensorRT and CUDA to
 * run the network, after that the raw detections are sorted and filtered using Non Maximum
 * Suppression (NMS).
 * 
 */
ObjectDetector::ObjectDetector() {
}

/**
 * @brief Constructs an object detector object.
 * @details Construct an object that can be used to detect objects inside images.
 * This code was built to run the Yolov5 implementation from Ultralytics, but it should
 * be able to run about any object detectors. The object detector uses TensorRT and CUDA to
 * run the network, after that the raw detections are sorted and filtered using Non Maximum
 * Suppression (NMS).
 * 
 * @param path_to_engine The absolute path to the tensorRT engine; i.e the weights of the network after conversion to tensorRT.
 * @param nms_tresh Non Maximum Supression (NMS) threshold. 
 * @param conf_tresh Confidence threshold. The threshold that the detector uses to keep or reject detected bounding boxes.
 * The lower the value, the higher the chances of false positives, the higher the value, the higher the chances of false negatives.
 * @param max_output_bbox_count The maximum amount of bounding boxes that can be detected.
 * @param buffer_size The size of the GPU buffer, in most scenarios, this value should be set to 2. This corresponds to the number of buffers that will be used.
 * In our case, we use two, one for the input of the network, one for its output.
 * @param image_size The size of the image the network will process. The network processes square images (NxN), hence, only one value is required, the largest.
 * 
 */
ObjectDetector::ObjectDetector(std::string path_to_engine, float nms_tresh, float conf_tresh,
                   size_t max_output_bbox_count, int buffer_size, int image_size, int num_classes) {
  path_to_engine_ = path_to_engine;
  nms_tresh_ = nms_tresh;
  conf_tresh_ = conf_tresh;
  max_output_bbox_count_ = max_output_bbox_count;
  buffer_size_ = buffer_size;
  image_size_ = image_size;
  num_classes_ = num_classes;

  buffers_.resize(buffer_size_);

  prepareEngine();

  input_data_ = std::shared_ptr<float[]>(new float[input_size_]);
  output_data_ = std::shared_ptr<float[]>(new float[output_size_]);
}

/**
 * @brief Constructs an object detector object.
 * @details Construct an object that can be used to detect objects inside images.
 * This code was built to run the Yolov5 implementation from Ultralytics, but it should
 * be able to run about any object detectors. The object detector uses TensorRT and CUDA to
 * run the network, after that the raw detections are sorted and filtered using Non Maximum
 * Suppression (NMS).
 * 
 * @param image_size The size of the image the network will process. The network processes square images (NxN), hence, only one value is required, the largest. 
 * @param det_p A structure that holds all the parameters related to the network.
 * @param nms_p A structure that holds all the parameters related to the Non-Maximum-Supression. 
 * 
 */
ObjectDetector::ObjectDetector(int image_size, DetectionParameters& det_p, NMSParameters& nms_p) {
  path_to_engine_ = det_p.engine_path;
  nms_tresh_ = nms_p.nms_thresh;
  conf_tresh_ = nms_p.conf_thresh;
  max_output_bbox_count_ = nms_p.max_output_bbox_count;
  buffer_size_ = det_p.num_buffers;
  image_size_ = image_size;
  num_classes_ = det_p.num_classes;
  buffers_.resize(buffer_size_);

  prepareEngine();

  input_data_ = std::shared_ptr<float[]>(new float[input_size_]);
  output_data_ = std::shared_ptr<float[]>(new float[output_size_]);
}

/**
 * @brief Makes sure the memory of the GPU is released properly.
 * 
 */
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

/**
 * @brief Returns the size of the buffer
 * @details Returns the size the number of float the buffer must contains.
 * This is used to reserve the correct amount of VRAM on the GPU.
 *  
 * @param dims The reference to the dimension of the buffer.
 * @return The required number of floats to be stored on the GPU.
 * 
 */
size_t ObjectDetector::getSizeByDim(const nvinfer1::Dims& dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];

    std::cout << dims.d[i] << ", ";
  }
  std::cout << std::endl;
  return size;
}

/**
 * @brief Initializes the object detector.
 * @details This method loads the model, allocates the memory on the GPU, and instantiate
 * the TensorRT engine. If the object detection model used is different from Yolov5 there may
 * be some differences in the number of buffers (input and outputs of the network) as well
 * as which buffer is used for what. In YoloV5 there are two buffers, buffer 0 is the input
 * while buffer 1 is the output, however this depends on the network architecture and can change.
 * 
 */
void ObjectDetector::prepareEngine() {
  printf("[LOG   ] ObjectDetector::%s::l%d Engine_path = %s.\n", __func__, __LINE__, path_to_engine_.c_str());

  // Reads and loads the neural network
  std::ifstream engine_file(path_to_engine_, std::ios::binary);
  if (!engine_file.good()) {
    printf("[ERROR ] ObjectDetector::%s::l%d No such engine file: %s.\n",__func__, __LINE__, path_to_engine_.c_str());
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

  // 
  runtime_ = nvinfer1::createInferRuntime(gLogger_);
  assert(runtime_ != nullptr);
  engine_ = runtime_->deserializeCudaEngine(trt_model_stream, trt_stream_size);
  assert(engine_ != nullptr);
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr);
  if (engine_->getNbBindings() != buffer_size_) {
    printf("[ERROR ] ObjectDetector::%s::l%d engine->getNbBindings() == %d, but should be %d.\n", __func__, __LINE__,
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
      printf("[ERROR ] ObjectDetector::%s::l%d binding_size == 0.\n",__func__,__LINE__);

      delete[] trt_model_stream;
      return;
    }

    cudaMalloc(&buffers_[i], binding_size);
    if (engine_->bindingIsInput(i)) {
      input_dims.emplace_back(engine_->getBindingDimensions(i));
      printf("[LOG   ] ObjectDetector::%s::l%d Input layer, size = %lu.\n", __func__, __LINE__, binding_size);
      input_size_ = (int) (binding_size / 4);
      printf("[LOG   ] ObjectDetector::%s::l%d Creating input buffer of size %d.\n", __func__, __LINE__, input_size_); 
    } else {
      output_dims.emplace_back(engine_->getBindingDimensions(i));
      printf("[LOG   ] ObjectDetector::%s::l%d Output layer, size = %lu.\n", __func__, __LINE__, binding_size);
      output_size_ = (int) (binding_size / 4);
      printf("[LOG   ] ObjectDetector::%s::l%d Creating output buffer of size %d.\n", __func__, __LINE__, output_size_);
    }
  }

  CUDA_CHECK(cudaStreamCreate(&stream_));
  delete[] trt_model_stream;
  printf("[LOG   ] ObjectDetector::%s::l%d Engine preparation finished.\n", __func__, __LINE__);
}

/**
 * @brief Runs the network
 * @details This method applies the forward pass of the network. Before calling this method,
 * the input buffer needs to be filled. This can be achieved by using the method called sendBufferToGPU.
 * To collect the result, the method called getBufferFromGPU must be called.
 * 
 */
void ObjectDetector::inferNetwork(){

  context_->enqueue(1, buffers_.data(), stream_, nullptr);

}

/**
 * @brief Sends a batch to the GPU.
 * @details Sends a batch, a set of images or a single image, to the GPU.
 * Once this function has been called, the forward pass of the network can be applied.
 * See infereNetwork. This function sends the data inside input_data_ inside buffer_[0]
 * on the GPU. The index of the buffer may change depending on the architecture.
 * 
 */
void ObjectDetector::sendBufferToGPU(){
  CUDA_CHECK(cudaMemcpyAsync(buffers_[0], input_data_.get(),
                             input_size_ * sizeof(float), cudaMemcpyHostToDevice,
                             stream_));

}

/**
 * @brief Fetches data from the GPU.
 * @details Fetches the result of the network forward pass. This function should be called
 * after infereNetwork. This function fetches data from buffer_[1] on the GPU, and stores it
 * inside output_data_. The index of the buffer may change depending on the architecture.
 * 
 */
void ObjectDetector::getBufferFromGPU(){
  CUDA_CHECK(cudaMemcpyAsync(output_data_.get(), buffers_[1],
                             output_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);
}

/**
 * @brief Transforms the RGB image into a buffer.
 * @details Converts the uint8 RGB image into a float32 RGB image and stores it into a buffer
 * that can be copied onto the GPU. The copy of the buffer is done by sendBufferToGPU.
 * It is important to note that the image is converted to float32 whithin this function.
 * To leverage float16 operation, it may be beneficial to cast to float16 instead.
 * 
 * @param image The reference to the RGB image to be preprocessed.
 */
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

/**
 * @brief Applies the object detector.  
 * @details Applies the object detector on a given image and returns the bounding boxes.
 * This function encapsulates all the preprocessing and data-handling between the CPU and the GPU.
 * It first preprocesses the image, to send it to a local buffer, then, it is sent to the GPU,
 * the inference is applied, the result is collected from the GPU and the Non Maximum Supression
 * (NMS) algorithm is applied on top of it.
 * 
 * @param image The image to be processed by the network.
 * @param bboxes The reference to a vector of vectors of bounding boxes. 
 */
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

/**
 * @brief Filters the bounding boxes generated by the network.
 * @details Applies Non Maximum Supression (NMS) to filter the bounding boxes generated by the network.
 * First step it checks if the bounding boxes have a confidence superior to a given threshold.
 * Second, it applies the non-maximum supression to remove deuplicate detections and other outliers.
 * 
 * @param bboxes The reference to a vector of vectors of bounding boxes.
 */
void ObjectDetector::nonMaximumSuppression(std::vector<std::vector<BoundingBox>> &bboxes) {
  bboxes.resize(num_classes_);
  int class_id;
  float conf; 

  for (int c = 0; c < num_classes_; ++c) {
    bboxes[c].reserve(output_size_);
  }
  // Bounding box is min_x, min_y, width, height, conf, class1, class2, ...
  for (int i = 0; i < output_size_; i += (num_classes_ + 5)) {
    conf = output_data_.get()[i + 4];
    if (conf > conf_tresh_) {
      assert(conf <= 1.0f);
      // Get all the probabilities that this objects belong to a given class
      std::vector<float> probabilities(num_classes_);
      for (unsigned int j=0; i < num_classes_; j++){
        // 5 : minx, miny, width, height, conf
        // i : number of objects
        // j : number of classes
        probabilities[j] = output_data_.get()[4 + i + j]; 
      }
      // Take the maximum
      class_id = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
      // Save to bounding box
      bboxes[class_id].push_back(BoundingBox(output_data_.get() + i, class_id));
    }
  }
  // Non-maximum supression
  for (int c = 0; c < num_classes_; ++c) {
    std::sort(bboxes[c].begin(), bboxes[c].end(), sortComparisonFunction);
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

/**
 * @brief Filters the bounding boxes generated by the network.
 * @details Applies Non Maximum Supression (NMS) to filter the bounding boxes generated by the network.
 * First step it checks if the bounding boxes have a confidence superior to a given threshold.
 * Second, it applies the non-maximum supression to remove deuplicate detections and other outliers.
 * 
 * @param bboxes The reference to a vector of vectors of bounding boxes.
 */
/*void ObjectDetectorRotation::nonMaximumSuppression(std::vector<std::vector<RotatedBoundingBox>> &bboxes) {
  bboxes.resize(num_classes_);
  int class_id;
  float conf; 

  for (int c = 0; c < num_classes_; ++c) {
    bboxes[c].reserve(output_size_);
  }
  // Rotated bounding box is cx, cy, width, height, cos(theta), sin(theta), conf, class1, class2, ...
  for (int i = 0; i < output_size_; i += (num_classes_ + 7)) {
    conf = output_data_.get()[i + 4];
    if (conf > conf_tresh_) {
      assert(conf <= 1.0f);
      // Get all the probabilities that this objects belong to a given class
      std::vector<float> probabilities(num_classes_);
      for (unsigned int j=0; i < num_classes_; j++){
        // 7 : cx, cy, width, height, cos(theta), sin(theta), conf
        // i : number of objects
        // j : number of classes
        probabilities[j] = output_data_.get()[6 + i + j]; 
      }
      // Take the maximum
      class_id = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
      // Save to bounding box
      bboxes[class_id].push_back(RotatedBoundingBox(output_data_.get() + i, class_id));
    }
  }
  // Non-maximum supression
  for (int c = 0; c < num_classes_; ++c) {
    std::sort(bboxes[c].begin(), bboxes[c].end(), sortComparisonFunction);
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
}*/