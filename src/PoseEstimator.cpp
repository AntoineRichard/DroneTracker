/**
 * @file PoseEstimator.cpp
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief Source code of the pose estimation class.
 * @details This file implements a simple algorithm to estimate the position of objects inside bounding boxes.
 */

#include <detect_and_track/PoseEstimator.h>

/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
PoseEstimator::PoseEstimator() {
}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param rejection_threshold The amount of points considered as outliers.
 * @param keep_threshold The amount of points to average from.
 * @param vertical_fov The vertical FOV of the camera.
 * @param horizontal_fov The horizontal FOV of the camera.
 * @param image_height The height of the images the class will process.
 * @param image_width The width of the images the class will process.
 */
PoseEstimator::PoseEstimator(float rejection_threshold, float keep_threshold, float vertical_fov, float horizontal_fov, int image_height, int image_width) {
  rejection_threshold_ = rejection_threshold;
  keep_threshold_ = keep_threshold;
  vertical_fov_ = vertical_fov * M_PI / 180;
  horizontal_fov_ = horizontal_fov * M_PI / 180;
  image_height_ = image_height;
  image_width_ = image_width;

  K_.resize(5,0.0);
  fx_ = 607.7302246;
  cx_ = 327.8651123;
  fy_ = 606.1353759;
  cy_ = 246.6830596;
}

/**
 * @brief Update the camera parameters.
 * @details Updates the camera parameters.
 * 
 * @param cx The x principle point coordinate.
 * @param cy The y principle point coordinate.
 * @param fx The x scaling parameter.
 * @param fy The y scaling parameter.
 * @param K The reference to the plumb-blob model parameters.
 */
void PoseEstimator::updateCameraParameters(float cx, float cy, float fx, float fy, std::vector<double> K) {
  cx_ = cx;
  cy_ = cy;
  fx_ = fx;
  fy_ = fy;
  fx_inv_ = 1/fx_;
  fy_inv_ = 1/fy_;
}

/**
 * @brief Computes the position of the pixel in the camera's local frame.
 * @details Compute the position of the pixel in the camera's local frame using a plumb-blob model.
 * This type of model can take into account advanced lens deformations.
 * Explanation on how it works can be found here: https://calib.io/blogs/knowledge-base/camera-models
 * 
 * This function inverses this model using a gradient descent.
 * 
 * @param depth The reference to the depth value, also called z.
 * @param pixel The reference to the pixel, format: u,v or row,col.
 * @param point The reference to the point in which the result will be saved, format: x,y,z.
 */
void PoseEstimator::deprojectPixel2PointBrownConrady(const float& depth, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;

  float xo = x;
  float yo = y;
  // Brown-Conrady model used by the realsense.
  for (int i = 0; i < 10; i++) {
    float r2 = x * x + y * y;
    float icdist = (float)1 / (float)(1 + ((K_[4] * r2 + K_[1]) * r2 + K_[0]) * r2);
    float delta_x = 2 * K_[2] * x * y + K_[3] * (r2 + 2 * x * x);
    float delta_y = 2 * K_[3] * x * y + K_[2] * (r2 + 2 * y * y);
    x = (xo - delta_x) * icdist;
    y = (yo - delta_y) * icdist;
  }
  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

/**
 * @brief Computes the position of the pixel in the camera's local frame.
 * @details Compute the position of the pixel in the camera's local frame using a pin-hole model.
 * This type of model is fast and simple.
 * Explanation on how it works can be found here: https://calib.io/blogs/knowledge-base/camera-models
 * 
 * @param depth The reference to the depth value, also called z.
 * @param pixel The reference to the pixel, format: (u,v) or (col,row).
 * @param point The reference to the point in which the result will be saved, format: (x,y,z).
 */
void PoseEstimator::deprojectPixel2PointPinHole(const float& depth, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;
  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

/**
 * @brief Project a point in 3D to a pixel position.
 * @details Utility function to project the position of point inside an image.
 * 
 * @param x The x position of the object in the image frame.
 * @param y The y position of the object in the image frame.
 * @param z The z position of the object in the image frame.
 * @param u The column of the pixel.
 * @param v The row of the pixel.
 */
void PoseEstimator::projectPixel2PointPinHole(const float& x, const float& y, const float& z, float& u, float& v){
  u = (x/z)*fx_ + cx_ ;
  v = (y/z)*fy_ + cy_;
}

/**
 * @brief Computes the position of the pixel in the camera's local frame.
 * @details Computes the position of the pixel in the camera's local frame using a modified pin-hole model.
 * Instead of using z, we use the effective distance between the object and the point, hence, we must compute z using the following equations:
 * ix = x * fx + cx
 * iy = y * fy + cy
 * x = ix * z
 * y = iy * z
 * d = sqrt(x*x + y*y + z*z)
 * d = z*sqrt(ix*ix + iy*iy + 1)
 * z = d/sqrt(ix*ix + iy*iy + 1)
 * 
 * @param dist The distance between the object and the camera.
 * @param pixel The reference to the pixel, format: (u,v) or (col,row).
 * @param point The reference to the point in which the result will be saved, format: (x,y,z).
 */
void PoseEstimator::distancePixel2PointPinHole(const float& dist, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;
  float z = dist / sqrt(x*x + y*y + 1);
  point[0] = z * x;
  point[1] = z * y;
  point[2] = z;
}

/**
 * @brief Computes the position of the pixel in the camera's local frame.
 * @details Computes the position of the pixel in the camera's local frame using a modified plumb-blob model.
 * Instead of using z, we use the effective distance between the object and the point, hence, we must compute z using the following equations:
 * ix = x * fx + cx
 * iy = y * fy + cy
 * x = ix * z
 * y = iy * z
 * d = sqrt(x*x + y*y + z*z)
 * d = z*sqrt(ix*ix + iy*iy + 1)
 * z = d/sqrt(ix*ix + iy*iy + 1)
 * 
 * @param dist The distance between the object and the camera.
 * @param pixel The reference to the pixel, format: (u,v) or (col,row).
 * @param point The reference to the point in which the result will be saved, format: (x,y,z).
 */
void PoseEstimator::distancePixel2PointBrownConrady(const float& dist, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;

  float xo = x;
  float yo = y;
  // Brown-Conrady model used by the realsense.
  for (int i = 0; i < 10; i++) {
    float r2 = x * x + y * y;
    float icdist = (float)1 / (float)(1 + ((K_[4] * r2 + K_[1]) * r2 + K_[0]) * r2);
    float delta_x = 2 * K_[2] * x * y + K_[3] * (r2 + 2 * x * x);
    float delta_y = 2 * K_[3] * x * y + K_[2] * (r2 + 2 * y * y);
    x = (xo - delta_x) * icdist;
    y = (yo - delta_y) * icdist;
  }
  float z = dist / sqrt(x*x + y*y + 1);
  point[0] = z * x;
  point[1] = z * y;
  point[2] = z;
}

/**
 * @brief Computes the distance between an object and the camera.
 * @details Computes the distance between an object and the camera.
 * To do so, we first calculate the distance between every point inside the bounding box and the camera.
 * Then, the 5% closest points are removed as considered as potential outliers (too close, noise in the camera).
 * Of the remaining points, the 10% smallest are then averaged to get the distance to the object. 
 * 
 * @param depth_image The reference to the depth image to compute the distance from.
 * @param x_min The reference to the position of the bounding box's left side.
 * @param y_min The reference to the position of the bounding box's top side.
 * @param width The reference to the width of the bounding box.
 * @param height The reference to the height of the bounding box.
 * @return The distance to the object.
 */
float PoseEstimator::getDistance(const cv::Mat& depth_image, const float& x_min, const float& y_min, const float& width, const float& height ) {
  float z, d;
  size_t reject, keep;
  std::vector<float> distances(height*width, 0);
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);

  unsigned int c = 0;
  for (int row = (int) y_min; row < (int) y_min + height; row++) {
    for (int col = (int) x_min; col < (int) x_min + width; col++) {
      z = depth_image.at<float>(row, col);
      if (z != 0) {
        pixel[0] = row;
        pixel[1] = col;
#ifdef BROWNCONRADY
        deprojectPixel2PointBrownConrady(z, pixel, point);
#else
        deprojectPixel2PointPinHole(z, pixel, point);
#endif
        d = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
        distances[c] = d;
        c++;
      }
    }
  }
  reject = distances.size() * rejection_threshold_;
  keep = distances.size() * keep_threshold_;
  std::sort(distances.begin(), distances.end(), std::less<float>());
  return std::accumulate(distances.begin() + reject, distances.begin() + reject+keep,0.0) / keep;
}

/**
 * @brief Computes the distance to all the detected objects.
 * @details Computes the distance to all the detected objects.
 * 
 * @param depth_image The reference to the depth image to compute the distance from. 
 * @param bboxes The reference to the bounding boxes to compute the distance of.
 * @return The distance to the objects.
 */
std::vector<std::vector<float>> PoseEstimator::extractDistanceFromDepth(const cv::Mat& depth_image, const std::vector<std::vector<BoundingBox>>& bboxes){
  std::vector<float> distance_vector;
  std::vector<std::vector<float>> distance_vectors;
  int rows, cols;


  // If the image does not exist, return -1 as distance.
  if (depth_image.empty()) {
    for (unsigned int i=0; i < bboxes.size(); i++) {
      distance_vector.clear();
      for (unsigned int j=0; j < bboxes[i].size(); j++) {
        if (!bboxes[0][i].valid_) {
          distance_vector.push_back(-1);
          continue;
        }
      }
      distance_vectors.push_back(distance_vector);
    }
    return distance_vectors; 
  }

  // Computes the distance to all the objects.
  for (unsigned int i=0; i < bboxes.size(); i++) {
    distance_vector.clear();
    for (unsigned int j=0; j < bboxes[i].size(); j++) {
#ifdef PROFILE
    start_distance = std::chrono::system_clock::now();
#endif
      // If the bounding box is invalid use -1 as distance.
      if (!bboxes[i][j].valid_) {
        distance_vector.push_back(-1);
#ifdef PROFILE
        end_distance = std::chrono::system_clock::now();
        printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance done in %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
        continue;
      }
      distance_vector.push_back(getDistance(depth_image, bboxes[i][j].x_min_, bboxes[i][j].y_min_, bboxes[i][j].w_, bboxes[i][j].h_ ));
#ifdef PROFILE
      end_distance = std::chrono::system_clock::now();
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance time %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
    }
    distance_vectors.push_back(distance_vector);
  }
  return distance_vectors;
}

/**
 * @brief Computes the distance to all the detected objects.
 * @details Computes the distance to all the detected objects.
 * 
 * @param depth_image The reference to the depth image to compute the distance from. 
 * @param tracked_states The reference to the tracked states to compute the distance of.
 * @return The distance to the objects.
 */
std::vector<std::map<unsigned int, float>> PoseEstimator::extractDistanceFromDepth(const cv::Mat& depth_image, const std::vector<std::map<unsigned int, std::vector<float>>>& tracked_states){
  std::vector<std::map<unsigned int, float>> distance_maps;
  distance_maps.resize(tracked_states.size());
  std::vector<float> distances;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  float z, d;
  int rows, cols;

  // If the image does not exist, return -1 as distance.
  if (depth_image.empty()) {
    for (unsigned int i=0; i < tracked_states.size(); i++) {
      for (auto & element : tracked_states[i]) {
        distance_maps[i].insert(std::pair(element.first, -1));
      }
    }
    return distance_maps; 
  }

  // Computes the distance to all the objects.
  for (unsigned int i=0; i < tracked_states.size(); i++) {
    for (auto & element : tracked_states[i]) {
#ifdef PROFILE
    start_distance = std::chrono::system_clock::now();
#endif
      distance_maps[i].insert(std::pair(element.first, getDistance(depth_image, element.second[0], element.second[1], element.second[4], element.second[5])));
#ifdef PROFILE
      end_distance = std::chrono::system_clock::now();
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance time %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
    }
  }
  return distance_maps;
}

/**
 * @brief Estimates the position of all the objects. 
 * @details Estimates the position of all the objects using their distance to the camera and position in the image.
 * 
 * @param distances The reference to the distance of the objects.
 * @param bboxes The reference to the tracked states.
 * @return The position of all the objects. 
 */
std::vector<std::vector<std::vector<float>>> PoseEstimator::estimatePosition(const std::vector<std::vector<float>>& distances, const std::vector<std::vector<BoundingBox>>& bboxes) {
  std::vector<std::vector<std::vector<float>>> point_vectors;
  std::vector<std::vector<float>> point_vector;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  for (unsigned int i=0; i < bboxes.size(); i++) {
    point_vector.clear();
    for (unsigned int j=0; j < bboxes[i].size(); j++) {
      float theta, phi;
      if (!bboxes[i][j].valid_) {
        point[0] = 0;
        point[1] = 0;
        point[2] = 0;
        point_vector.push_back(point);
        continue;
      }
      pixel[0] = bboxes[i][j].x_;
      pixel[1] = bboxes[i][j].y_;
#ifdef BROWNCONRADY
      distancePixel2PointBrownConrady(distances[i][j], pixel, point);
#else
      distancePixel2PointPinHole(distances[i][j], pixel, point);
#endif
      point_vector.push_back(point);
    }
    point_vectors.push_back(point_vector);
  }
  return point_vectors;
}

/**
 * @brief Estimates the position of all the objects.
 * @details Estimates the position of all the objects using their distance to the camera and position in the image.
 * 
 * @param distances The reference to the distances of the objects.
 * @param tracked_states The reference to the tracked states.
 * @return The position of all the objects.
 */
std::vector<std::map<unsigned int, std::vector<float>>> PoseEstimator::estimatePosition(const std::vector<std::map<unsigned int, float>>& distances, const std::vector<std::map<unsigned int, std::vector<float>>>& tracked_states) {
  std::vector<std::map<unsigned int, std::vector<float>>> point_maps;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  point_maps.resize(tracked_states.size());
  for (unsigned int i=0; i < tracked_states.size(); i++) {
    for (auto & element : tracked_states[i]) {
      float theta, phi;
      pixel[0] = element.second[0];
      pixel[1] = element.second[1];
#ifdef BROWNCONRADY
      distancePixel2PointBrownConrady(distances[i].at(element.first), pixel, point);
#else
      distancePixel2PointPinHole(distances[i].at(element.first), pixel, point);
#endif
      point_maps[i].insert(std::pair(element.first, point));
    }
  }
  return point_maps;
}