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

  if (position_mode == "pin_hole") {
    position_mode_ = 0;
  } else if (position_mode == "plumb_blob") {
    position_mode_ = 1;
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
  cx_ = camera_parameters[0];
  cy_ = camera_parameters[1];
  fx_ = camera_parameters[2];
  fy_ = camera_parameters[3];
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