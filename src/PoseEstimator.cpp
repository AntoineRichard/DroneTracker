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

  K_.resize(5,0.0);
  fx_ = 607.7302246;
  cx_ = 327.8651123;
  fy_ = 606.1353759;
  cy_ = 246.6830596;
}

void PoseEstimator::updateCameraParameters(float cx, float cy, float fx, float fy, std::vector<double> K) {
  cx_ = cx;
  cy_ = cy;
  fx_ = fx;
  fy_ = fy;
  fx_inv_ = 1/fx_;
  fy_inv_ = 1/fy_;
}

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

void PoseEstimator::deprojectPixel2PointPinHole(const float& depth, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;
  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

void PoseEstimator::projectPixel2PointPinHole(const float& x, const float& y, const float& z, float& u, float& v){
  u = (x/z)*fx_ + cx_ ;
  v = (y/z)*fy_ + cy_;
}

void PoseEstimator::distancePixel2PointPinHole(const float& depth, const std::vector<float>& pixel, std::vector<float>& point){
  float x = (pixel[0] - cx_) / fx_;
  float y = (pixel[1] - cy_) / fy_;
  float z = depth / sqrt(x*x + y*y + 1);
  point[0] = z * x;
  point[1] = z * y;
  point[2] = z;
}

void PoseEstimator::distancePixel2PointBrownConrady(const float& depth, const std::vector<float>& pixel, std::vector<float>& point){
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
  float z = depth / sqrt(x*x + y*y + 1);
  point[0] = z * x;
  point[1] = z * y;
  point[2] = z;
}

std::vector<std::vector<float>> PoseEstimator::extractDistanceFromDepth(const cv::Mat& depth_image, const std::vector<std::vector<BoundingBox>>& bboxes){
  std::vector<float> distance_vector;
  std::vector<std::vector<float>> distance_vectors;
  std::vector<float> distances;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  float z, d;
  int rows, cols;

#ifdef PROFILE
  std::chrono::time_point<std::chrono::system_clock> start_distance;
  std::chrono::time_point<std::chrono::system_clock> end_distance;
  std::chrono::time_point<std::chrono::system_clock> start_measure;
  std::chrono::time_point<std::chrono::system_clock> end_measure;
#endif

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

  size_t reject, keep;
  for (unsigned int i=0; i < bboxes.size(); i++) {
    distance_vector.clear();
    for (unsigned int j=0; j < bboxes[i].size(); j++) {
#ifdef PROFILE
    start_distance = std::chrono::system_clock::now();
#endif
      if (!bboxes[i][j].valid_) {
        distance_vector.push_back(-1);
#ifdef PROFILE
        end_distance = std::chrono::system_clock::now();
        printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance done in %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
#endif
        continue;
      }
#ifdef PROFILE
      start_measure = std::chrono::system_clock::now();
#endif
      rows = bboxes[i][j].h_;//y_max_ - bboxes[i][j].y_min_;
      cols = bboxes[i][j].w_;//x_max_ - bboxes[i][j].x_min_;
      distances.clear();
      distances.resize(rows*cols, 0);
      unsigned int c = 0;
      for (int row = (int) bboxes[0][i].y_min_; row < (int) bboxes[0][i].y_max_; row++) {
        for (int col = (int) bboxes[0][i].x_min_; col < (int) bboxes[0][i].x_max_; col++) {
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
            //distances.push_back(d);
            c++;
          }
        }
      }
#ifdef PROFILE
      end_measure = std::chrono::system_clock::now();
#endif
      reject = distances.size() * rejection_threshold_;
      keep = distances.size() * keep_threshold_;
      std::sort(distances.begin(), distances.end(), std::less<float>());
      distance_vector.push_back(std::accumulate(distances.begin() + reject, distances.begin() + reject+keep,0.0) / keep);
#ifdef PROFILE
      end_distance = std::chrono::system_clock::now();
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance time %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d   + measure time %ld us\n", __func__, __LINE__,  std::chrono::duration_cast<std::chrono::microseconds>(end_measure - start_measure).count());
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d   + sort time %ld us\n", __func__, __LINE__,  std::chrono::duration_cast<std::chrono::microseconds>(end_distance - end_measure).count());
#endif
    }
    distance_vectors.push_back(distance_vector);
  }
  return distance_vectors;
}

std::vector<std::map<unsigned int, float>> PoseEstimator::extractDistanceFromDepth(const cv::Mat& depth_image, const std::vector<std::map<unsigned int, std::vector<float>>>& tracked_states){
  std::vector<std::map<unsigned int, float>> distance_maps;
  distance_maps.resize(tracked_states.size());
  std::vector<float> distances;
  std::vector<float> point(3,0);
  std::vector<float> pixel(2,0);
  float z, d;
  int rows, cols;

#ifdef PROFILE
  std::chrono::time_point<std::chrono::system_clock> start_distance;
  std::chrono::time_point<std::chrono::system_clock> end_distance;
  std::chrono::time_point<std::chrono::system_clock> start_measure;
  std::chrono::time_point<std::chrono::system_clock> end_measure;
#endif

  if (depth_image.empty()) {
    for (unsigned int i=0; i < tracked_states.size(); i++) {
      for (auto & element : tracked_states[i]) {
        distance_maps[i].insert(std::pair(element.first, -1));
      }
    }
    return distance_maps; 
  }

  size_t reject, keep;
  for (unsigned int i=0; i < tracked_states.size(); i++) {
    for (auto & element : tracked_states[i]) {
#ifdef PROFILE
      start_distance = std::chrono::system_clock::now();
      start_measure = std::chrono::system_clock::now();
#endif
      distances.clear();
      cols = element.second[4];
      rows = element.second[5];
      unsigned int c = 0;
      //distances.resize(rows*cols, 0);
      for (int row = (int) element.second[1] - rows/2; row < (int) element.second[1] + rows/2; row++) {
        for (int col = (int) element.second[0] - cols/2; col < (int) element.second[0] + cols/2; col++) {
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
            if (std::isnan(d)) {
              continue;
            }
            distances.push_back(d);
            //distances[c] = d;
            //c++;
          }
        }
      }
#ifdef PROFILE
      end_measure = std::chrono::system_clock::now();
#endif
      reject = distances.size() * rejection_threshold_;
      keep = distances.size() * keep_threshold_;
      std::sort(distances.begin(), distances.end(), std::less<float>()); 
      distance_maps[i].insert(std::pair(element.first, std::accumulate(distances.begin() + reject, distances.begin() + reject+keep,0.0) / keep));
#ifdef PROFILE
      end_distance = std::chrono::system_clock::now();
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d - Obj %d distance time %ld us\n", __func__, __LINE__,  i, std::chrono::duration_cast<std::chrono::microseconds>(end_distance - start_distance).count());
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d   + measure time %ld us\n", __func__, __LINE__,  std::chrono::duration_cast<std::chrono::microseconds>(end_measure - start_measure).count());
      printf("\e[1;34m[PROFILE]\e[0m PoseEstimator::%s::l%d   + sort time %ld us\n", __func__, __LINE__,  std::chrono::duration_cast<std::chrono::microseconds>(end_distance - end_measure).count());
#endif
    }
  }
  return distance_maps;
}


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