#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <execution>
#include <set>

// EIGEN
#include <eigen3/Eigen/Dense>
#include <math.h>

// CUDA/TENSOR_RT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <depth_image_extractor/logging.h>
#include <depth_image_extractor/utils.h>

// ROS
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

# define M_PI           3.14159265358979323846  /* pi */

// State indices
# define PX 0
# define PY 1
# define PZ 2
# define VX 3
# define VY 4
# define VZ 5
# define WIDTH 6
# define AREA 7


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

  //delete[] input_data_;
  //delete[] output_data_;
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
      //ROS_INFO("%d",i);
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
       /*ROS_INFO(
           "object detected: class = %d, conf = %.4f, tx = %.4f, ty = %.4f, tw = %.4f, th = %.4f",
	   (int) 0,//(output_data.get()[i+6]>output_data.get()[i+5]),
           output_data_.get()[i+4],
           output_data_.get()[i+0],
           output_data_.get()[i+1],
           output_data_.get()[i+2],
           output_data_.get()[i+3]
       );*/
      assert(conf <= 1.0f);
      const int class_id = 0;
      //    (int)(output_data.get()[i + 6] > output_data.get()[i + 5]);
      bboxes[class_id].push_back(BoundingBox(output_data_.get() + i));
    }
  }

  for (int c = 0; c < ObjectClass::NUM_CLASS; ++c) {
    //ROS_INFO("class %d has %lu available bboxes", c, bboxes[c].size());
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
    //ROS_INFO("class %d has %lu valid bboxes", c, valid_count);
  }
}

class KalmanFilter {
  private:
    float dt_;
    bool use_dim_;
    bool use_z_;

    Eigen::VectorXf X_;
    Eigen::VectorXf Z_;
    Eigen::MatrixXf P_;
    Eigen::MatrixXf F_;
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;
    Eigen::MatrixXf I8_;
    Eigen::MatrixXf H_;

    void initialize();
    void getMeasurement(const std::vector<float>&);

  public:
    KalmanFilter();
    KalmanFilter(float, bool, bool);
    void resetFilter(const std::vector<float>&);
    void update();
    void predict();
    void correct(const std::vector<float>&);
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
};

KalmanFilter::KalmanFilter() {
  // State is [x, y, z, vx, vy, vz, h, w]
  initialize();
}

KalmanFilter::KalmanFilter(float dt, bool use_dim, bool use_z) {
  // State is [x, y, z, vx, vy, vz, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_z_ = use_z;
  initialize();
}

void KalmanFilter::initialize() {
  // State is [x, y, z, vx ,vy, vz, h, w]
  X_ = Eigen::VectorXf(8);
  P_ = Eigen::MatrixXf(8,8);
  Q_ = Eigen::MatrixXf(8,8);
  I8_ = Eigen::MatrixXf::Identity(8,8);

  Eigen::MatrixXf H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
  H_pos = Eigen::MatrixXf(2,8);
  H_pos_z = Eigen::MatrixXf(1,8);
  H_vel = Eigen::MatrixXf(2,8);
  H_vel_z = Eigen::MatrixXf(1,8);
  H_hw = Eigen::MatrixXf(2,8); // Why 3x8 -> area
  
  Eigen::MatrixXf R_pos, R_pos_z, R_vel, R_vel_z, R_hw;
  R_pos = Eigen::MatrixXf(2,8);
  R_pos_z = Eigen::MatrixXf(1,8);
  R_vel = Eigen::MatrixXf(2,8);
  R_vel_z = Eigen::MatrixXf(1,8);
  R_hw = Eigen::MatrixXf(2,8); // Why 3x8 -> area

  int h_size = 4;
  if (use_z_) {
    h_size += 2;
  }
  if (use_dim_) {
    h_size += 2; // Or 3?
  }
  H_ = Eigen::MatrixXf(h_size, 8);
  R_ = Eigen::MatrixXf(h_size, 8);
  Z_ = Eigen::VectorXf(h_size);

  X_ << 0, 0, 0, 0, 0, 0, 0, 0;

  P_ << 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0;
  
  Q_ << 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0;

  /*R_ << 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0;*/

  F_ << 1.0, 0, 0, dt_, 0, 0, 0, 0,
       0, 1.0, 0, 0, dt_, 0, 0, 0,
       0, 0, 1.0, 0, 0, dt_, 0, 0,
       0, 0, 0, 1.0, 0, 0, 0, 0,
       0, 0, 0, 0, 1.0, 0, 0, 0,
       0, 0, 0, 0, 0, 1.0, 0, 0,
       0, 0, 0, 0, 0, 0, 1.0, 0,
       0, 0, 0, 0, 0, 0, 0, 1.0;

  H_pos << 1.0, 0, 0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0, 0, 0;
  
  H_pos_z << 0, 0, 1.0, 0, 0, 0, 0, 0;

  H_vel << 0, 0, 0, 1.0, 0, 0, 0, 0,
            0, 0, 0, 0, 1.0, 0, 0, 0;
  
  H_vel_z << 0, 0, 0, 0, 0, 1.0, 0, 0; 

  H_hw << 0, 0, 0, 0, 0, 0, 1.0, 0,
           0, 0, 0, 0, 0, 0, 0, 1.0;

  if (use_z_){
    if (use_dim_){
      H_ << H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
      R_ << R_pos, R_pos_z, R_vel, R_vel_z, R_hw;
    } else {
      H_ << H_pos, H_pos_z, H_vel, H_vel_z;
      R_ << R_pos, R_pos_z, R_vel, R_vel_z;
    }
  } else {  
    if (use_dim_){
      H_ << H_pos, H_vel, H_hw;
      R_ << R_pos, R_vel, R_hw;
    } else {
      H_ << H_pos, H_vel;
      R_ << R_pos, R_vel;
    }
  }
}


void KalmanFilter::resetFilter(const std::vector<float>& initial_state) {
  X_ << initial_state[0],
        initial_state[1],
        initial_state[2],
        initial_state[3],
        initial_state[4],
        initial_state[5],
        initial_state[6],
        initial_state[7];

  P_ << 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0;
}

void KalmanFilter::getMeasurement(const std::vector<float>& measurement) {
  // State is [x, y, z, vx, vy, vz, h, w]
  Z_(0) = measurement[0];
  Z_(1) = measurement[1];
  Z_(3) = measurement[3];
  Z_(4) = measurement[4];
  if (use_z_) {
    Z_(2) = measurement[2];
    Z_(5) = measurement[5];
  }
  if (use_dim_) {
    Z_(6) = measurement[6];
    Z_(7) = measurement[7];
  }
}

void KalmanFilter::getState(std::vector<float>& state) {
  state[0] = X_(0);
  state[1] = X_(1);
  state[2] = X_(2);
  state[3] = X_(3);
  state[4] = X_(4);
  state[5] = X_(5);
  state[6] = X_(6);
  state[7] = X_(7);
}

void KalmanFilter::getUncertainty(std::vector<float>& uncertainty) {
  uncertainty[0] = P_(0,0);
  uncertainty[1] = P_(1,1);
  uncertainty[2] = P_(2,2);
  uncertainty[3] = P_(3,3);
  uncertainty[4] = P_(4,4);
  uncertainty[5] = P_(5,5);
  uncertainty[6] = P_(6,6);
  uncertainty[7] = P_(7,7);
}

void KalmanFilter::predict() {
  X_  = F_ * X_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::correct(const std::vector<float>& Z){
  Eigen::MatrixXf Y, S, K, I_KH;

  getMeasurement(Z);

  Y = Z_ - H_ * X_;
  S = R_ + H_ * P_ * H_.transpose();
  K = P_ * H_.transpose() * S.inverse();
  X_ = X_ + K * Y;
  I_KH = I8_ - K*H_;
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
}

/*class DecisionTree {
  private:
    float center_threshold_; 
    float area_threshold_; 
    float body_ratio_;
    float centroids_error(const std::vector<float>&, const std::vector<float>&);
    float area_ratio(const std::vector<float>&, const std::vector<float>&);
    float bodyShapeError(const std::vector<float>&, const std::vector<float>&);
  public:
    DecisionTree();
    DecisionTree(float, float, float);
    bool isMatch(const std::vector<float>&, const std::vector<float>&);
};

DecisionTree::DecisionTree() {
}

DecisionTree::DecisionTree(float center_threshold, float area_threshold, float body_ratio) {
  center_threshold_ = center_threshold;
  area_threshold_ = area_threshold;
  body_ratio_ = body_ratio;
}

float DecisionTree::centroids_error(const std::vector<float>& s1, const std::vector<float>& s2) {
   return sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]) + (s1[2] - s2[2])*(s1[2] - s2[2]));
}

float DecisionTree::area_ratio(const std::vector<float>& s1, const std::vector<float>& s2) {
  return (s1[6]*s1[7]) / (s2[6]*s2[7]);
}

float DecisionTree::bodyShapeError(const std::vector<float>& s1, const std::vector<float>& s2) {
  return (s1[6]/s1[7]) / (s2[6]/s2[7]);
}

bool DecisionTree::isMatch(const std::vector<float>& s1, const std::vector<float>& s2){
  if (centroids_error(s1,s2) < center_threshold_) {
    if ((1/area_threshold < area_ratio(s1,s2)) && (area_ratio(s1,s2) < area_threshold_)) {
      return true;
    }
  }
  return false;
}*/

/*struct Trace {
  unsigned int id;
  unsigned int frame_id;
  unsigned int skipped_frames;
  unsigned int lost_track;
  float px;
  float py;
  float pz;
  float vx;
  float vy;
  float vz;
  float width;
  float height;
  float p_px;
  float p_py;
  float p_pz;
  float p_vx;
  float p_vy;
  float p_vz;
  float p_height;
  float p_width;
}*/

/*class Drone {
  private:
    KalmanFilter KF_;
    unsigned int skipped_frame_;
    unsigned int nb_frames_;
    unsigned int id_;
  public:
    void newFrame();
    void update();
    void predict();
    getState(std::vector<float>&);
    getUncertainty(std::vector<float>&);
}

class Tracker {
  private:
    unsigned int track_id_count_;
    std::vector<Drone> Drones_;
  public:

}*/


class PoseEstimator {
  private:
    float rejection_threshold_;
    float keep_threshold_;
    float vertical_fov_;
    float horizontal_fov_;
    int image_height_;
    int image_width_;

    std::vector<float> P_;

  public:
    PoseEstimator();
    PoseEstimator(float, float, float, float, int, int);
    std::vector<float> extractDistanceFromDepth(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
    std::vector<std::vector<float>> estimatePosition(const std::vector<float>& , const std::vector<std::vector<BoundingBox>>&);
    //void updateCameraParameters();
    //<std::Vec3f> getPose(const cv::Mat&, const std::vector<std::vector<BoundingBox>>&);
};

PoseEstimator::PoseEstimator() {
}

PoseEstimator::PoseEstimator(float rejection_threshold, float keep_threshold, float vertical_fov, float horizontal_fov, int image_height, int image_width) {
  rejection_threshold_ = rejection_threshold;
  keep_threshold_ = keep_threshold;
  vertical_fov_ = vertical_fov * M_PI / 180;
  horizontal_fov_ = horizontal_fov * M_PI / 180;
  image_height_ = image_height;
  image_width_ = image_width;

  P_.resize(3,0.0);
}

std::vector<float> PoseEstimator::extractDistanceFromDepth(const cv::Mat& depth_image, const std::vector<std::vector<BoundingBox>>& bboxes){
  std::vector<float> distance_vector;
  if (depth_image.empty()) {
    for (unsigned int i=0; i < bboxes[0].size(); i++) {
      if (!bboxes[0][i].valid_) {
        distance_vector.push_back(-1);
        continue;
      }
      distance_vector.push_back(-1);
    }
    return distance_vector; 
  }
  size_t reject, keep;
  for (unsigned int i=0; i < bboxes[0].size(); i++) {
    if (!bboxes[0][i].valid_) {
      distance_vector.push_back(-1);
      continue;
    }
    std::vector<float> distances;
    for (unsigned int row = bboxes[0][i].y_min_; row<bboxes[0][i].y_max_; row++) {
      for (unsigned int col = bboxes[0][i].x_min_; col<bboxes[0][i].x_max_; col++) {
        distances.push_back(depth_image.at<float>(row,col));
      }
    }
    reject = distances.size() * rejection_threshold_;
    keep = distances.size() * keep_threshold_;
    std::sort(distances.begin(), distances.end(), std::less<float>());
    distance_vector.push_back(std::accumulate(distances.begin() + reject, distances.begin() + reject+keep,0.0) / keep);
  }
  return distance_vector;
}

std::vector<std::vector<float>> PoseEstimator::estimatePosition(const std::vector<float>& distances, const std::vector<std::vector<BoundingBox>>& bboxes) {
  std::vector<std::vector<float>> points;
  for (unsigned int i=0; i < bboxes[0].size(); i++) {
    float theta, phi;
    if (!bboxes[0][i].valid_) {
      P_[0] = 0;
      P_[1] = 0;
      P_[2] = 0;
      points.push_back(P_);
      continue;
    }
    phi = horizontal_fov_ * ((bboxes[0][i].x_ / image_width_) - 0.5);
    theta = vertical_fov_ * (0.5 - (bboxes[0][i].y_ / image_height_));
    ROS_INFO("%.3f, %.3f, %3.f, %3.f", phi, theta, bboxes[0][i].x_, bboxes[0][i].y_);
    P_[0] = distances[i] * cos(phi) * sin(theta);
    P_[1] = distances[i] * sin(phi) * sin(theta);
    P_[2] = distances[i] * cos(theta);
    points.push_back(P_);
  }
  return points;
}

class ROSDetector {
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    image_transport::Publisher image_pub_;

    int image_size_;
    int padding_rows_;
    int padding_cols_;
    cv::Mat padded_image_;
    cv::Mat depth_image_;

    sensor_msgs::ImagePtr image_ptr_out_;
    ObjectDetector* OD_;
    PoseEstimator* PE_;

    void imageCallback(const sensor_msgs::ImageConstPtr&);
    void depthCallback(const sensor_msgs::ImageConstPtr&);
    void adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>&);
    void padImage(const cv::Mat&);

  public:
    ROSDetector();
    ~ROSDetector();
};

ROSDetector::ROSDetector() : nh_("~"), it_(nh_), OD_(), PE_() {
  float nms_tresh, conf_tresh;
  int max_output_bbox_count;
  
  std::string default_path_to_engine("  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);None");
  std::string path_to_engine;


  nh_.param("nms_tresh",nms_tresh,0.45f);
  nh_.param("image_size",image_size_,640);
  nh_.param("conf_tresh",conf_tresh,0.25f);
  nh_.param("max_output_bbox_count", max_output_bbox_count, 1000);
  nh_.param("path_to_engine", path_to_engine, default_path_to_engine);
  OD_ = new ObjectDetector(path_to_engine, nms_tresh, conf_tresh,max_output_bbox_count, 2, image_size_);
  PE_ = new PoseEstimator(0.02, 0.15, 58.0, 87.0, 480, 640);
  
  padded_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC3);
  //depth_image_ = cv::Mat::zeros(image_size_, image_size_, CV_16UC1);

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetector::imageCallback, this);
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetector::depthCallback, this);
  image_pub_ = it_.advertise("/detection/image", 1);
}

ROSDetector::~ROSDetector() {
}

void ROSDetector::padImage(const cv::Mat& image) {
  float r;
  r = (float) image_size_ / std::max(image.rows, image.cols);
  ROS_INFO("%f",r);
  if (r != 1) {
    ROS_ERROR("Not implemented");
  } else {
    padding_rows_ = (image_size_ - image.rows)/2;
    padding_cols_ = (image_size_ - image.cols)/2;
    image.copyTo(padded_image_(cv::Range(padding_rows_,padding_rows_+image.rows),cv::Range(padding_cols_,padding_cols_+image.cols)));
  }
}

void ROSDetector::depthCallback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv_ptr->image.convertTo(depth_image_, CV_32F, 0.001);
}

void ROSDetector::adjustBoundingBoxes(std::vector<std::vector<BoundingBox>>& bboxes) {
  for (unsigned int j=0; j < bboxes[0].size(); j++) {
      if (!bboxes[0][j].valid_) {
        continue;
      }
    bboxes[0][j].x_ -= padding_cols_;
    bboxes[0][j].y_ -= padding_rows_;
    bboxes[0][j].x_min_ = bboxes[0][j].x_ - bboxes[0][j].w_/2;
    bboxes[0][j].x_max_ = bboxes[0][j].x_ + bboxes[0][j].w_/2;
    bboxes[0][j].y_min_ = bboxes[0][j].y_ - bboxes[0][j].h_/2;
    bboxes[0][j].y_max_ = bboxes[0][j].y_ + bboxes[0][j].h_/2;
  }
}

void ROSDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  //cv::Mat image_new;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  auto start = std::chrono::system_clock::now();
  padImage(image);
  std::vector<std::vector<BoundingBox>> bboxes(ObjectClass::NUM_CLASS);
  OD_->detectObjects(padded_image_, bboxes);
  adjustBoundingBoxes(bboxes);
  std::vector<float> distances;
  auto start1 = std::chrono::system_clock::now();
  distances = PE_->extractDistanceFromDepth(depth_image_, bboxes);
  std::vector<std::vector<float>> points;
  points = PE_->estimatePosition(distances, bboxes);
  auto end1 = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  ROS_INFO("Depth detection %d us", std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count());
  for (unsigned int i=0; i<distances.size(); i++){
    ROS_INFO("Distance of object %d : %.3fm",i,distances[i]);
    ROS_INFO("Object %d position: x %.3f y %.3f z %.3f",i,points[i][0],points[i][1],points[i][2]);
  }

  
  for (unsigned int i=0; i<bboxes[0].size(); i++) {
    if (!bboxes[0][i].valid_) {
      continue;
    }
    //const cv::Rect rect(bbox.x_min_ - padding_cols_, bbox.y_min_-padding_rows_, bbox.w_, bbox.h_);
    const cv::Rect rect(bboxes[0][i].x_min_, bboxes[0][i].y_min_, bboxes[0][i].w_, bboxes[0][i].h_);
    cv::rectangle(image, rect, cv::Scalar(0,255,0), 3);
    cv::putText(image, std::to_string(distances[i]), cv::Point(bboxes[0][i].x_min_,bboxes[0][i].y_min_-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0,255,0), 2);
  }
  
  //image.convertTo(image, CV_8UC3, 255.0f);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  image_pub_.publish(image_ptr_out_);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ROSDetector rd;
  ros::spin();
  return 0;
}