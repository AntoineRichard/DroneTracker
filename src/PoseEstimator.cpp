#include <depth_image_extractor/PoseEstimator.h>

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