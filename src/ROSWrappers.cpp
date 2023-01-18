#include <detect_and_track/ROSWrappers.h>

/**
 * @brief Constructs a ROS node to perform object detection.
 * @details This class wrapps around the object detector and integrates
 * all the ROS related components required to run the network on ROS video
 * streams. This node can perform multi-class detection and outputs the detected
 * bounding boxes, and optionnaly, images of the bounding-boxes overlayed on top
 * of the original image.
 * 
 */
ROSDetect::ROSDetect() : nh_("~"), it_(nh_), Detect() {
  // Empty structs
  GlobalParameters glo_p;
  DetectionParameters det_p;
  NMSParameters nms_p;

  // Global parameters
  nh_.param("image_rows", glo_p.image_height, 480);
  nh_.param("image_cols", glo_p.image_width, 640);
  // NMS parameters
  nh_.param("nms_thresh", nms_p.nms_thresh,0.45f);
  nh_.param("conf_thresh", nms_p.conf_thresh,0.25f);
  nh_.param("max_output_bbox_count", nms_p.max_output_bbox_count, 1000);
  // Model parameters
  std::string default_path_to_engine("None");
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);
  // Initializes the detector
  buildDetect(glo_p, det_p, nms_p);

  // Creates the subscribers and publishers
  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetect::imageCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  detection_pub_ = it_.advertise("detection_image", 1);
#endif
  bboxes_pub_ = nh_.advertise<detect_and_track::BoundingBoxes2D>("bounding_boxes", 1);
}

ROSDetect::~ROSDetect() {
}

/**
 * @brief 
 * @details
 * 
 * @param image 
 * @param bboxes 
 */
void ROSDetect::publishDetectionImage(cv::Mat& image, std::vector<std::vector<BoundingBox>>& bboxes) {
  generateDetectionImage(image, bboxes);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  detection_pub_.publish(image_ptr_out_);
}

/**
 * @brief 
 * @details
 * 
 * @param bboxes 
 * @param header 
 */
void ROSDetect::publishDetections(std::vector<std::vector<BoundingBox>>& bboxes, std_msgs::Header& header) {
  unsigned int counter = 0;
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;

  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      ros_bbox.min_x = bboxes[i][j].x_min_;
      ros_bbox.min_y = bboxes[i][j].y_min_;
      ros_bbox.height = bboxes[i][j].h_;
      ros_bbox.width = bboxes[i][j].w_;
      ros_bbox.class_id = i;
      ros_bbox.detection_id = counter;
      counter ++;
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

/**
 * @brief
 * @details
 * 
 * @param msg 
 */
void ROSDetect::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
#endif
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  detectObjects(image, bboxes);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingDetection();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  publishDetectionImage(image, bboxes);
#endif
  publishDetections(bboxes, cv_ptr->header);
}


/**
 * @brief Construct a new ROSDetectAndLocate::ROSDetectAndLocate object
 * 
 */
ROSDetectAndLocate::ROSDetectAndLocate() : ROSDetect(), Locate() {
  // Empty structs
  GlobalParameters glo_p;
  LocalizationParameters loc_p;
  CameraParameters cam_p;

  // Global parameters
  nh_.param("image_rows", glo_p.image_height, 480);
  nh_.param("image_cols", glo_p.image_width, 640);
  // Localization parameters
  std::string position_mode("min_distance");
  nh_.param("rejection_threshold", loc_p.reject_thresh, 0.1f);
  nh_.param("keep_threshold", loc_p.keep_thresh, 0.1f);
  nh_.param("position_mode", loc_p.mode, position_mode);
  // Camera parameters
  std::string distortion_model("pin_hole");
  std::vector<float> P(5,0);
  std::vector<float> K(5,0);
  nh_.param("camera_parameters", cam_p.camera_parameters, P);
  nh_.param("K", cam_p.lens_distortion, K);
  nh_.param("lens_distortion_model", cam_p.distortion_model, distortion_model);
  // Initialize the position estimator
  buildLocate(glo_p, loc_p, cam_p);
  
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetectAndLocate::depthCallback, this);
  depth_info_sub_ = nh_.subscribe("/camera/aligned_depth_to_color/camera_info", 1, &ROSDetectAndLocate::depthInfoCallback, this);
  pose_array_pub_ = nh_.advertise<geometry_msgs::PoseArray>("detection_pose_array", 1);
#ifdef PUBLISH_DETECTION_WITH_POSITION
  positions_bboxes_pub_ = nh_.advertise<detect_and_track::PositionBoundingBox2DArray>("bounding_boxes_with_positions",1);
#else
  positions_pub_ = nh_.advertise<detect_and_track::PositionIDArray>("detection_positions",1);
#endif
}

ROSDetectAndLocate::~ROSDetectAndLocate() {
}

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSDetectAndLocate::depthInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
  std::vector<double> P{msg->K[2], msg->K[5], msg->K[0], msg->K[4]};
  std::vector<float> Pf(msg->P.begin(), msg->P.end());
  std::vector<float> K(msg->D.begin(), msg->D.end());
  updateCameraInfo(Pf, K);
}

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSDetectAndLocate::depthCallback(const sensor_msgs::ImageConstPtr& msg){
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

#ifdef PUBLISH_DETECTION_WITH_POSITION
/**
 * @brief 
 * 
 * @param bboxes 
 * @param points 
 * @param header 
 */
void ROSDetectAndLocate::publishDetectionsAndPositions(std::vector<std::vector<BoundingBox>>& bboxes,
                                                   std::vector<std::vector<std::vector<float>>>& points,
                                                   std_msgs::Header& header) {
  detect_and_track::PositionBoundingBox2DArray ros_bboxes;
  detect_and_track::PositionBoundingBox2D ros_bbox;
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<detect_and_track::PositionBoundingBox2D> vec_ros_bboxes;
  std::vector<geometry_msgs::Pose> poses;
  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      // BBox + Position
      ros_bbox.bbox.min_x = bboxes[i][j].x_min_;
      ros_bbox.bbox.min_y = bboxes[i][j].y_min_;
      ros_bbox.bbox.height = bboxes[i][j].h_;
      ros_bbox.bbox.width = bboxes[i][j].w_;
      ros_bbox.bbox.class_id = i;
      ros_bbox.position.x = points[i][j][0];
      ros_bbox.position.y = points[i][j][1];
      ros_bbox.position.z = points[i][j][2];
      vec_ros_bboxes.push_back(ros_bbox);
      // Pose Array (nice for visualization)
      pose.position.x = points[i][j][0];
      pose.position.y = points[i][j][1];
      pose.position.z = points[i][j][2];
      pose.orientation.w = 1.0;
      poses.push_back(pose);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;
  pose_array.header = ros_bboxes.header;
  pose_array.poses = poses;
  
  positions_bboxes_pub_.publish(ros_bboxes);
  pose_array_pub_.publish(pose_array);
}

#else
/**
 * @brief 
 * 
 * @param bboxes 
 * @param points 
 * @param header 
 */
void ROSDetectAndLocate::publishPositions(std::vector<std::vector<BoundingBox>>& bboxes,
                                          std::vector<std::vector<std::vector<float>>>& points,
                                          std_msgs::Header& header) {
  unsigned int counter = 0;
  detect_and_track::PositionIDArray id_positions;
  detect_and_track::PositionID id_position;
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<detect_and_track::PositionID> vec_id_positions;
  std::vector<geometry_msgs::Pose> poses;

  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      // Position
      id_position.position.x = points[i][j][0];
      id_position.position.y = points[i][j][1];
      id_position.position.z = points[i][j][2];
      id_position.detection_id = counter;
      vec_id_positions.push_back(id_position);
      // Pose Array (nice for visualization)
      pose.position.x = points[i][j][0];
      pose.position.y = points[i][j][1];
      pose.position.z = points[i][j][2];
      pose.orientation.w = 1.0;
      counter ++;
      poses.push_back(pose);
    }
  }
  id_positions.header.stamp = header.stamp;
  id_positions.header.frame_id = header.frame_id;
  id_positions.positions = vec_id_positions;
  pose_array.header = id_positions.header;
  pose_array.poses = poses;
  
  positions_pub_.publish(id_positions);
  pose_array_pub_.publish(pose_array);
}
#endif

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSDetectAndLocate::imageCallback(const sensor_msgs::ImageConstPtr& msg){
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
#endif
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  std::vector<std::vector<float>> distances;
  std::vector<std::vector<std::vector<float>>> points;
  detectObjects(image, bboxes);
  locate(depth_image_, bboxes, distances, points);

#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingDetection();
  printProfilingLocalization();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  publishDetectionImage(image, bboxes);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  publishDetectionsAndPositions(bboxes, points, cv_ptr->header);
#else
  publishDetections(bboxes,cv_ptr->header);
  publishPositions(bboxes, points, cv_ptr->header);
#endif
}


/**
 * @brief Construct a new ROSDetectTrack2DAndLocate::ROSDetectTrack2DAndLocate object
 * 
 */
ROSDetectTrack2DAndLocate::ROSDetectTrack2DAndLocate() : ROSDetectAndLocate(), Track2D() {
  DetectionParameters det_p;
  KalmanParameters kal_p;
  TrackingParameters tra_p;
  BBoxRejectionParameters bbo_p;
  // Model parameters
  std::string default_path_to_engine("None");
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);
  // Kalman parameters
  std::vector<float> default_Q {9.0, 9.0, 200.0, 200.0, 5.0, 5.0};
  std::vector<float> default_R {2.0, 2.0, 200.0, 200.0, 2.0, 2.0};
  Q_.resize(6);
  R_.resize(6);
  nh_.param("Q", kal_p.Q, default_Q);
  nh_.param("R", kal_p.R, default_R);
  nh_.param("use_vel", kal_p.use_vel, false);
  nh_.param("use_dim", kal_p.use_vel, true);
  // Tracking parameters
  nh_.param("center_threshold", tra_p.center_thresh, 80.0f);
  nh_.param("dist_threshold", tra_p.distance_thresh, 150.0f);
  nh_.param("body_ratio", tra_p.body_ratio, 0.5f);
  nh_.param("area_threshold", tra_p.area_thresh, 2.0f);
  nh_.param("dt", tra_p.dt, 0.02f);
  nh_.param("max_frames_to_skip", tra_p.max_frames_to_skip, 10);
  // BBox rejection
  nh_.param("min_bbox_width", bbo_p.min_bbox_width, 60);
  nh_.param("max_bbox_width", bbo_p.max_bbox_width, 400);
  nh_.param("min_bbox_height", bbo_p.min_bbox_height, 60);
  nh_.param("max_bbox_height", bbo_p.max_bbox_height, 300);
  buildTrack2D(det_p, kal_p, tra_p, bbo_p);

#ifdef PUBLISH_DETECTION_IMAGE
  tracker_pub_ = it_.advertise("tracking_image", 1);
#endif
}

ROSDetectTrack2DAndLocate::~ROSDetectTrack2DAndLocate(){}

/**
 * @brief 
 * 
 * @param image_tracker 
 * @param tracker_states 
 */
void ROSDetectTrack2DAndLocate::publishTrackingImage(cv::Mat& image_tracker,
                                                    std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states) {
  generateTrackingImage(image_tracker, tracker_states);
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);
}

#ifdef PUBLISH_DETECTION_WITH_POSITION
/**
 * @brief 
 * 
 * @param tracker_states 
 * @param points 
 * @param header 
 */
void ROSDetectTrack2DAndLocate::publishDetectionsAndPositions(std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states,
                                                              std::vector<std::map<unsigned int, std::vector<float>>>& points,
                                                              std_msgs::Header& header){
  detect_and_track::PositionBoundingBox2DArray ros_bboxes;
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<geometry_msgs::Pose> poses;
  detect_and_track::PositionBoundingBox2D ros_bbox;
  std::vector<detect_and_track::PositionBoundingBox2D> vec_ros_bboxes;
  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.bbox.height = element.second[5];
      ros_bbox.bbox.width = element.second[4];
      ros_bbox.bbox.class_id = i;
      ros_bbox.bbox.detection_id = element.first;
      ros_bbox.position.x = points[i].at(element.first)[0];
      ros_bbox.position.y = points[i].at(element.first)[1];
      ros_bbox.position.z = points[i].at(element.first)[2];
      pose.position.x = points[i].at(element.first)[0];
      pose.position.y = points[i].at(element.first)[1];
      pose.position.z = points[i].at(element.first)[2];
      pose.orientation.w = 1.0;
      vec_ros_bboxes.push_back(ros_bbox);
      poses.push_back(pose);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;
  pose_array.header = ros_bboxes.header;
  pose_array.poses = poses;
  
  positions_bboxes_pub_.publish(ros_bboxes);
  pose_array_pub_.publish(pose_array);
}

#else

/**
 * @brief 
 * 
 * @param tracker_states 
 * @param header 
 */
void ROSDetectTrack2DAndLocate::publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states,
                                                  std_msgs::Header& header) {
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;

  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.height = element.second[5];
      ros_bbox.width = element.second[4];
      ros_bbox.class_id = i;
      ros_bbox.detection_id = element.first;
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

/**
 * @brief 
 * 
 * @param points 
 * @param header 
 */
void ROSDetectTrack2DAndLocate::publishPositions(std::vector<std::map<unsigned int, std::vector<float>>>& points,
                                                 std_msgs::Header& header) {
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
  std::vector<geometry_msgs::Pose> poses;
  for (unsigned int i=0; i<points.size(); i++) {
    for (auto & element : points[i]) {
      pose.position.x = element.second[0];
      pose.position.y = element.second[1];
      pose.position.z = element.second[2];
      pose.orientation.w = 1.0;
      poses.push_back(pose);
    }
  }
  pose_array.header.stamp = header.stamp;
  pose_array.header.frame_id = header.frame_id;
  pose_array.poses = poses;
  pose_array_pub_.publish(pose_array);
}
#endif

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSDetectTrack2DAndLocate::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  t2_ = t1_;
  t1_ = ros::Time::now();
  ros::Duration dt = t1_ - t2_; 
  dt_ = (float) (dt.toSec());
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
#endif
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  std::vector<std::map<unsigned int, std::vector<float>>> tracker_states;
  std::vector<std::map<unsigned int, std::vector<float>>> points;
  std::vector<std::map<unsigned int, float>> distances;
  tracker_states.resize(num_classes_);
  detectObjects(image, bboxes);
  cv::Mat image_tracker = image.clone();
  track(bboxes, tracker_states, dt_);
  locate(depth_image_, tracker_states, distances, points);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingDetection();
  printProfilingTracking();
  printProfilingLocalization();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  publishDetectionImage(image, bboxes);
  publishTrackingImage(image_tracker, tracker_states);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  publishDetectionsAndPositions(tracker_states, points, cv_ptr->header);
#else
  publishDetections(tracker_states, cv_ptr->header);
  publishPositions(points, cv_ptr->header);
#endif
}

/**
 * @brief Construct a new ROSDetectAndTrack2D::ROSDetectAndTrack2D object
 * 
 */
ROSDetectAndTrack2D::ROSDetectAndTrack2D() : ROSDetect(), Track2D() {
  DetectionParameters det_p;
  KalmanParameters kal_p;
  TrackingParameters tra_p;
  BBoxRejectionParameters bbo_p;
  // Model parameters
  std::string default_path_to_engine("None");
  std::string path_to_engine;
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);
  // Kalman parameters
  std::vector<float> default_Q {9.0, 9.0, 200.0, 200.0, 5.0, 5.0};
  std::vector<float> default_R {2.0, 2.0, 200.0, 200.0, 2.0, 2.0};
  Q_.resize(6);
  R_.resize(6);
  nh_.param("Q", kal_p.Q, default_Q);
  nh_.param("R", kal_p.R, default_R);
  nh_.param("use_vel", kal_p.use_vel, false);
  nh_.param("use_dim", kal_p.use_vel, true);
  // Tracking parameters
  nh_.param("center_threshold", tra_p.center_thresh, 80.0f);
  nh_.param("dist_threshold", tra_p.distance_thresh, 150.0f);
  nh_.param("body_ratio", tra_p.body_ratio, 0.5f);
  nh_.param("area_threshold", tra_p.area_thresh, 2.0f);
  nh_.param("dt", tra_p.dt, 0.02f);
  nh_.param("max_frames_to_skip", tra_p.max_frames_to_skip, 10);
  // BBox rejection
  nh_.param("min_bbox_width", bbo_p.min_bbox_width, 60);
  nh_.param("max_bbox_width", bbo_p.max_bbox_width, 400);
  nh_.param("min_bbox_width", bbo_p.min_bbox_height, 60);
  nh_.param("max_bbox_width", bbo_p.max_bbox_height, 300);
  buildTrack2D(det_p, kal_p, tra_p, bbo_p);

#ifdef PUBLISH_DETECTION_IMAGE
  tracker_pub_ = it_.advertise("tracking_image", 1);
#endif
}

ROSDetectAndTrack2D::~ROSDetectAndTrack2D(){}

/**
 * @brief 
 * 
 * @param image_tracker 
 * @param tracker_states 
 */
void ROSDetectAndTrack2D::publishTrackingImage(cv::Mat& image_tracker,
                                               std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states) {
  generateTrackingImage(image_tracker, tracker_states);
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);
}

/**
 * @brief 
 * 
 * @param tracker_states 
 * @param header 
 */
void ROSDetectAndTrack2D::publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states,
                                            std_msgs::Header& header) {
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;

  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.height = element.second[5];
      ros_bbox.width = element.second[4];
      ros_bbox.class_id = i;
      ros_bbox.detection_id = element.first;
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSDetectAndTrack2D::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  t2_ = t1_;
  t1_ = ros::Time::now();
  ros::Duration dt = t1_ - t2_; 
  dt_ = (float) (dt.toSec());
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
#endif
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  std::vector<std::map<unsigned int, std::vector<float>>> tracker_states;
  tracker_states.resize(num_classes_);
  detectObjects(image, bboxes);
  cv::Mat image_tracker = image.clone();
  track(bboxes, tracker_states, dt_);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingDetection();
  printProfilingTracking();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  publishDetectionImage(image, bboxes);
  publishTrackingImage(image_tracker, tracker_states);
#endif
  publishDetections(tracker_states, cv_ptr->header);
}

/**
 * @brief Construct a new ROSTrack2D::ROSTrack2D object
 * 
 */
ROSTrack2D::ROSTrack2D() : nh_("~"), it_(nh_), Track2D() {
  DetectionParameters det_p;
  KalmanParameters kal_p;
  TrackingParameters tra_p;
  BBoxRejectionParameters bbo_p;
  // Model parameters
  std::string default_path_to_engine("None");
  std::string path_to_engine;
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);
  num_classes_ = det_p.num_classes;
  // Kalman parameters
  std::vector<float> default_Q {9.0, 9.0, 200.0, 200.0, 5.0, 5.0};
  std::vector<float> default_R {2.0, 2.0, 200.0, 200.0, 2.0, 2.0};
  Q_.resize(6);
  R_.resize(6);
  nh_.param("Q", kal_p.Q, default_Q);
  nh_.param("R", kal_p.R, default_R);
  nh_.param("use_vel", kal_p.use_vel, false);
  nh_.param("use_dim", kal_p.use_vel, true);
  // Tracking parameters
  nh_.param("center_threshold", tra_p.center_thresh, 80.0f);
  nh_.param("dist_threshold", tra_p.distance_thresh, 150.0f);
  nh_.param("body_ratio", tra_p.body_ratio, 0.5f);
  nh_.param("area_threshold", tra_p.area_thresh, 2.0f);
  nh_.param("dt", tra_p.dt, 0.02f);
  nh_.param("max_frames_to_skip", tra_p.max_frames_to_skip, 10);
  // BBox rejection
  nh_.param("min_bbox_width", bbo_p.min_bbox_width, 60);
  nh_.param("max_bbox_width", bbo_p.max_bbox_width, 400);
  nh_.param("min_bbox_width", bbo_p.min_bbox_height, 60);
  nh_.param("max_bbox_width", bbo_p.max_bbox_height, 300);
  buildTrack2D(det_p, kal_p, tra_p, bbo_p);

  cv::Mat image_;

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSTrack2D::imageCallback, this);
  bboxes_sub_ = nh_.subscribe("bounding_boxes", 1, &ROSTrack2D::bboxesCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  tracker_pub_ = it_.advertise("tracking_image", 1);
#endif
  bboxes_pub_ = nh_.advertise<detect_and_track::BoundingBoxes2D>("tracking_bounding_boxes", 1);
}

ROSTrack2D::~ROSTrack2D(){}

/**
 * @brief 
 * 
 * @param image_tracker 
 * @param tracker_states 
 */
void ROSTrack2D::publishTrackingImage(cv::Mat& image_tracker,
                                               std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states) {
  generateTrackingImage(image_tracker, tracker_states);
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);
}

/**
 * @brief 
 * 
 * @param tracker_states 
 * @param header 
 */
void ROSTrack2D::publishDetections(std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states,
                                            std_msgs::Header& header) {
  detect_and_track::BoundingBoxes2D ros_bboxes;
  detect_and_track::BoundingBox2D ros_bbox;
  std::vector<detect_and_track::BoundingBox2D> vec_ros_bboxes;

  for (unsigned int i=0; i<tracker_states.size(); i++) {
    for (auto & element : tracker_states[i]) {
      ros_bbox.min_x = element.second[0] - element.second[4]/2;
      ros_bbox.min_y = element.second[1] - element.second[5]/2;
      ros_bbox.height = element.second[5];
      ros_bbox.width = element.second[4];
      ros_bbox.class_id = i;
      ros_bbox.detection_id = element.first;
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

void ROSTrack2D::ROSbboxes2bboxes(const detect_and_track::BoundingBoxes2DConstPtr& msg, std::vector<std::vector<BoundingBox>>& bboxes){
  bboxes.resize(num_classes_);
  for (unsigned int i=0; i < msg->bboxes.size(); i++) { 
    bboxes[msg->bboxes[i].class_id].push_back(BoundingBox(msg->bboxes[i].min_x,
                                                          msg->bboxes[i].min_y,
                                                          msg->bboxes[i].width,
                                                          msg->bboxes[i].height,
                                                          msg->bboxes[i].conf,
                                                          msg->bboxes[i].class_id));
  }
}

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSTrack2D::bboxesCallback(const detect_and_track::BoundingBoxes2DConstPtr& msg){
  t2_ = t1_;
  t1_ = ros::Time::now();
  ros::Duration dt = t1_ - t2_; 
  dt_ = (float) (dt.toSec());
#ifdef PROFILE
  auto start_inference = std::chrono::system_clock::now();
#endif
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  std::vector<std::map<unsigned int, std::vector<float>>> tracker_states;
  tracker_states.resize(num_classes_);
  ROSbboxes2bboxes(msg, bboxes);
  track(bboxes, tracker_states, dt_);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
  printProfilingTracking();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  cv::Mat image_tracker = image_.clone();
  publishTrackingImage(image_tracker, tracker_states);
#endif
  publishDetections(tracker_states, header_);
}

/**
 * @brief 
 * 
 * @param msg 
 */
void ROSTrack2D::imageCallback(const sensor_msgs::ImageConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  image_ = cv_ptr->image;
  header_ = cv_ptr->header;
  cv::cvtColor(image_, image_, cv::COLOR_BGR2RGB);
}