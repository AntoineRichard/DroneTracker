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
 * @brief Builds an object dedicated to estimating the distance and position.
 * @details This object implements different methods to estimate the position and distance of 
 * objects detected within RGB-D images. It also allows the user to choose between different
 * camera model which we detail bellow: \n
 *  - "pin_hole" model, a simple model that does not account for lens deformations.
 *  In most situations this model is sufficient. \n
 *  - "plumb_blob" model, a more complicated model that adds lens deformation to
 *  the Pin-Hole model. This model is slower, as to compute its inverse we need to
 * estimate it through a short gradient descent. \n
 * To estimate the distance and position we then provide 2 different modes: \n 
 *  - "center": this mode takes the distance at the center of the bounding box
 *  to estimate the distance and position of the objects. It's fast, but may be
 *  wrong on thin objects like drones where the center of the bounding can be empty. \n
 *  - "min_distance": this mode takes the filtered minimal distance to estimate the
 *  distance to the center of the dected object. This is will create an offset, as
 *  the distance measured is the smallest, and is sensitive to obstructions. However,
 *  it always ensure that the measured distance is the one of the object. \n 
 * 
 */
PoseEstimator::PoseEstimator() {}

/**
 * @brief Builds an object dedicated to estimating the distance and position.
 * @details This object implements different methods to estimate the position and distance of 
 * objects detected within RGB-D images. It also allows the user to choose between different
 * camera model which we detail bellow: \n
 *  - "pin_hole" model, a simple model that does not account for lens deformations.
 *  In most situations this model is sufficient. \n
 *  - "plumb_blob" model, a more complicated model that adds lens deformation to
 *  the Pin-Hole model. This model is slower, as to compute its inverse we need to
 * estimate it through a short gradient descent. \n
 * To estimate the distance and position we then provide 2 different modes: \n 
 *  - "center": this mode takes the distance at the center of the bounding box
 *  to estimate the distance and position of the objects. It's fast, but may be
 *  wrong on thin objects like drones where the center of the bounding can be empty. \n
 *  - "min_distance": this mode takes the filtered minimal distance to estimate the
 *  distance to the center of the dected object. This is will create an offset, as
 *  the distance measured is the smallest, and is sensitive to obstructions. However,
 *  it always ensure that the measured distance is the one of the object. \n 
 * @param rejection_threshold The amount of points considered as outliers.
 * @param keep_threshold The amount of points to average from.
 * @param image_height The height of the images the class will process.
 * @param image_width The width of the images the class will process.
 * @param camera_parameters The parameters of the camera used in both the pin-hole,
 *  and plumb-blob model.The vector is organized as follows [fx,fy,cx,cy].
 * @param K The lens deformation parameters, used in the plumb-blob model.
 * A vector of size 5, organized as follows: [k1,k2,k3,k4,k5].
 * @param deformation_model The deformation model used to project object from the image frame
 *  to the local camera frame, pin_hole or plumb_blob. Note using plumb_blob with K = [0,0,0,0,0]
 *  is equivalent to using a pin-hole model.
 * @param position_mode The mode used to estimate distance to the object. Currently there are
 *  2 different modes: min_distance, center.
 */
PoseEstimator::PoseEstimator(float rejection_threshold, float keep_threshold, int image_height, int image_width,
                             std::vector<float>& camera_parameters, std::vector<float>& K,
                             std::string distortion_model, std::string position_mode) {
  rejection_threshold_ = rejection_threshold;
  keep_threshold_ = keep_threshold;
  image_height_ = image_height;
  image_width_ = image_width;
  
  if (distortion_model == "pin_hole") {
    distortion_model_ = 0;
  } else if (distortion_model == "plumb_blob") {
    distortion_model_ = 1;
  } else {
    distortion_model_ = 0;
  }

  if (position_mode == "min_distance") {
    position_mode_ = 0;
  } else if (position_mode == "center") {
    position_mode_ = 1;
  } else if (position_mode == "average_min_distance") {
    position_mode_ = 2;
  } else {
    position_mode_ = 0;
  }

  K_.resize(5,0.0);
  K_ = K;
  fx_ = camera_parameters[0];
  fy_ = camera_parameters[1];
  cx_ = camera_parameters[2];
  cy_ = camera_parameters[3];
  fx_inv_ = 1/fx_;
  fy_inv_ = 1/fy_;
}

/**
 * @brief Builds an object dedicated to estimating the distance and position.
 * @details This object implements different methods to estimate the position and distance of 
 * objects detected within RGB-D images. It also allows the user to choose between different
 * camera model which we detail bellow: \n
 *  - "pin_hole" model, a simple model that does not account for lens deformations.
 *  In most situations this model is sufficient. \n
 *  - "plumb_blob" model, a more complicated model that adds lens deformation to
 *  the Pin-Hole model. This model is slower, as to compute its inverse we need to
 * estimate it through a short gradient descent. \n
 * To estimate the distance and position we then provide 2 different modes: \n 
 *  - "center": this mode takes the distance at the center of the bounding box
 *  to estimate the distance and position of the objects. It's fast, but may be
 *  wrong on thin objects like drones where the center of the bounding can be empty. \n
 *  - "min_distance": this mode takes the filtered minimal distance to estimate the
 *  distance to the center of the dected object. This is will create an offset, as
 *  the distance measured is the smallest, and is sensitive to obstructions. However,
 *  it always ensure that the measured distance is the one of the object. \n 
 * @param glo_p A structure that holds some basic parameters.
 * @param loc_p A structure that holds the parameters related to the position estimation of detected objects.
 * @param cam_p A structure that holds the parameters related to the camera.
 */
PoseEstimator::PoseEstimator(GlobalParameters& glo_p, LocalizationParameters& loc_p, CameraParameters& cam_p) {
  rejection_threshold_ = loc_p.reject_thresh;
  keep_threshold_ = loc_p.keep_thresh;
  image_height_ = glo_p.image_height;
  image_width_ = glo_p.image_width;
  
  if (cam_p.distortion_model == "pin_hole") {
    distortion_model_ = 0;
  } else if (cam_p.distortion_model == "plumb_blob") {
    distortion_model_ = 1;
  } else {
    distortion_model_ = 0;
  }

  if (loc_p.mode == "pin_hole") {
    position_mode_ = 0;
  } else if (loc_p.mode == "plumb_blob") {
    position_mode_ = 1;
  } else {
    position_mode_ = 0;
  }

  K_.resize(5,0.0);
  K_ = cam_p.lens_distortion;
  fx_ = cam_p.camera_parameters[0];
  fy_ = cam_p.camera_parameters[1];
  cx_ = cam_p.camera_parameters[2];
  cy_ = cam_p.camera_parameters[3];
  fx_inv_ = 1/fx_;
  fy_inv_ = 1/fy_;
}

/**
 * @brief Update the camera parameters.
 * @details Updates the camera parameters.
 * 
 * @param cx The x principle point coordinate.
 * @param cy The y principle point coordinate.
 * @param fx The x scaling parameter.
 * @param fy The y scaling parameter.
 * @param lens_parameters The reference to the plumb-blob model parameters.
 */
void PoseEstimator::updateCameraParameters(const std::vector<float>& camera_parameters, const std::vector<float>& lens_parameters) {
  fx_ = camera_parameters[0];
  fy_ = camera_parameters[1];
  cx_ = camera_parameters[2];
  cy_ = camera_parameters[3];
  fx_inv_ = 1/fx_;
  fy_inv_ = 1/fy_;
  K_ = lens_parameters;
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
 * Instead of using z, we use the effective distance between the object and the point, hence, we must compute
 *  z using the following equations: \n
 * ix = x * fx + cx \n
 * iy = y * fy + cy \n
 * x = ix * z \n
 * y = iy * z \n
 * d = sqrt(x*x + y*y + z*z) \n
 * d = z*sqrt(ix*ix + iy*iy + 1) \n
 * z = d/sqrt(ix*ix + iy*iy + 1) \n
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
 * Instead of using z, we use the effective distance between the object and the point, hence, we must compute
 *  z using the following equations: \n
 * ix = x * fx + cx \n
 * iy = y * fy + cy \n
 * x = ix * z \n
 * y = iy * z \n
 * d = sqrt(x*x + y*y + z*z) \n
 * d = z*sqrt(ix*ix + iy*iy + 1) \n
 * z = d/sqrt(ix*ix + iy*iy + 1) \n
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
 * Selects the method to compute the distance using user input.
 * There are currently 3 options:
 *  - getMinDistance
 *  - getAverageMinDistance
 *  - getCenterDistance
 * 
 * @param depth_image The reference to the depth image to compute the distance from.
 * @param x_min The reference to the position of the bounding box's left side.
 * @param y_min The reference to the position of the bounding box's top side.
 * @param width The reference to the width of the bounding box.
 * @param height The reference to the height of the bounding box.
 * @return The distance to the object.
 */
float PoseEstimator::getDistance(const cv::Mat& depth_image, const int& x_min, const int& y_min, const int& width, const int& height) {
  float distance;
  if (position_mode_ == 0) {
    distance = getMinDistance(depth_image, x_min, y_min, width, height);
  } else if (position_mode_ == 1) {
    distance = getMinAverageDistance(depth_image, x_min, y_min, width, height);
  } else if (position_mode_ == 2) {
    distance = getCenterDistance(depth_image, x_min, y_min, width, height);
  }
  return distance;
}

/**
 * @brief Computes the distance between an object and the camera.
 * @details Computes the distance between an object and the camera.
 * To do so, we first calculate the distance between every point inside the bounding box and the camera.
 * Then, we look for the closest points and use it as the distance to the object.
 * This method can be usefull for thin objects with large foot prints such as drones.
 * With drones the center of the bounding box rarely lands on the drone itself,
 * and averaging the N% closest points can lead to selecting points that belong to a wall or else.
 * 
 * @param depth_image The reference to the depth image to compute the distance from.
 * @param x_min The reference to the position of the bounding box's left side.
 * @param y_min The reference to the position of the bounding box's top side.
 * @param width The reference to the width of the bounding box.
 * @param height The reference to the height of the bounding box.
 * @return The distance to the object.
 */
float PoseEstimator::getMinDistance(const cv::Mat& depth_image, const int& x_min, const int& y_min, const int& width, const int& height ) {
  float z, d, distance;
  size_t reject, keep;
  std::vector<float> distances((int) (height*width), 0);
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  unsigned int c = 0;
  for (int col = x_min; col < (x_min + width); col++) {
    for (int row = y_min; row < (y_min + height); row++) {
      z = depth_image.at<float>(row, col);
      if ((z > 0.3) && (z < 10.0)) { // TODO: These values could be parameters.
        pixel[0] = (float) col;
        pixel[1] = (float) row;
        if (distortion_model_ == 1) {
          deprojectPixel2PointBrownConrady(z, pixel, point);
        } else {
          deprojectPixel2PointPinHole(z, pixel, point);
        }
        d = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
        distances[c] = d;
        c++;
      }
    }
  }
  std::vector<float> short_distances(distances.begin(), distances.begin() + c);
  return *std::min_element(short_distances.begin(), short_distances.end());
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
float PoseEstimator::getMinAverageDistance(const cv::Mat& depth_image, const int& x_min, const int& y_min, const int& width, const int& height ) {
  float z, d, distance;
  size_t reject, keep;
  std::vector<float> distances((int) (height*width), 0);
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  unsigned int c = 0;

  for (int col = x_min; col < (x_min + width); col++) {
    for (int row = y_min; row < (y_min + height); row++) {
      z = depth_image.at<float>(row, col);
      if ((z > 0.3) && (z < 10.0)) {
        pixel[0] = (float) col;
        pixel[1] = (float) row;
        if (distortion_model_ == 1) {
          deprojectPixel2PointBrownConrady(z, pixel, point);
        } else {
          deprojectPixel2PointPinHole(z, pixel, point);
        }
        d = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
        distances[c] = d;
        c++;
      }
    }
  }
  std::vector<float> short_distances(distances.begin(), distances.begin() + c);
  reject = short_distances.size() * rejection_threshold_;
  keep = short_distances.size() * keep_threshold_;
  std::sort(short_distances.begin(), short_distances.end(), std::less<float>());
  return std::accumulate(short_distances.begin() + reject, short_distances.begin() + reject+keep, 0.0)/keep;
}

/**
 * @brief Computes the distance between an object and the camera.
 * @details Computes the distance between an object and the camera.
 * To do so, we first use the distance measured at the center of the object.
 * 
 * @param depth_image The reference to the depth image to compute the distance from.
 * @param x_min The reference to the position of the bounding box's left side.
 * @param y_min The reference to the position of the bounding box's top side.
 * @param width The reference to the width of the bounding box.
 * @param height The reference to the height of the bounding box.
 * @return The distance to the object.
 */
float PoseEstimator::getCenterDistance(const cv::Mat& depth_image, const int& x_min, const int& y_min, const int& width, const int& height ) {
  float z, d;
  int col, row;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  col = x_min + width / 2;
  row = y_min + height / 2;
  z = depth_image.at<float>(col, row);
  pixel[0] = (float) col;
  pixel[1] = (float) row;
  if (distortion_model_ == 1) {
    deprojectPixel2PointBrownConrady(z, pixel, point);
  } else {
    deprojectPixel2PointPinHole(z, pixel, point);
  }
  d = sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
  return d;
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
#ifdef DEBUG_POSE
    printf("\e[1;31m[DEBUG  ]\e[0m PoseEstimator::%s::l%d - Depth image hasn't been received yet. Setting distance to -1.\n", __func__, __LINE__);
#endif
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
    auto start_distance = std::chrono::system_clock::now();
#endif
      // If the bounding box is invalid use -1 as distance.
      if (!bboxes[i][j].valid_) {
#ifdef DEBUG_POSE
        printf("\e[1;33m[DEBUG  ]\e[0m PoseEstimator::%s::l%d - Bounding box is invalid, setting distance to -1.\n", __func__, __LINE__);
#endif
        distance_vector.push_back(-1);
#ifdef PROFILE
        auto end_distance = std::chrono::system_clock::now();
        printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance done in %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
        continue;
      }
      distance_vector.push_back(getDistance(depth_image, (int) bboxes[i][j].x_min_, (int) bboxes[i][j].y_min_, (int) bboxes[i][j].w_, (int) bboxes[i][j].h_ ));
#ifdef PROFILE
      auto end_distance = std::chrono::system_clock::now();
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
  float z, d, dist;
  int rows, cols;

  // If the image does not exist, return -1 as distance.
  if (depth_image.empty()) {
#ifdef DEBUG_POSE
    printf("\e[1;33m[DEBUG  ]\e[0m PoseEstimator::%s::l%d - Depth image hasn't been received yet. Setting distance to -1.\n", __func__, __LINE__);
#endif
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
    auto start_distance = std::chrono::system_clock::now();
#endif
    dist = getDistance(depth_image, (int) element.second[0], (int) element.second[1], (int) element.second[4], (int) element.second[5]);
    distance_maps[i].insert(std::pair(element.first, dist));
#ifdef PROFILE
      auto end_distance = std::chrono::system_clock::now();
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance time %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
#ifdef DEBUG_POSE
      printf("\e[1;33m[DEBUG  ]\e[0m PoseEstimator::%s::l%d - Distance of tracked object %d is %.3f.\n", __func__, __LINE__, element.first, dist);
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
      if (distortion_model_ == 1){
        distancePixel2PointBrownConrady(distances[i][j], pixel, point);
      } else {
        distancePixel2PointPinHole(distances[i][j], pixel, point);
      }
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
      printf("%d",element.first);
      pixel[0] = element.second[0];
      pixel[1] = element.second[1];
      if (distortion_model_ == 1){
        distancePixel2PointBrownConrady(distances[i].at(element.first), pixel, point);
      } else {
        distancePixel2PointPinHole(distances[i].at(element.first), pixel, point);
      }
      point_maps[i].insert(std::pair(element.first, point));
    }
  }
  return point_maps;
}