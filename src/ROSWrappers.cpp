#include <detect_and_track/ROSWrappers.h>

ROSDetect::ROSDetect() : nh_("~"), it_(nh_), Detect() {
  // Empty structs
  GlobalParameters glo_p;
  DetectionParameters det_p;
  NMSParameters nms_p;
  // Global parameters
  nh_.param("image_rows", glo_p.image_height, 480);
  nh_.param("image_cols", glo_p.image_width, 640);
  // NMS parameters
  nh_.param("nms_tresh", nms_p.nms_thresh,0.45f);
  nh_.param("conf_tresh", nms_p.conf_thresh,0.25f);
  nh_.param("max_output_bbox_count", nms_p.max_output_bbox_count, 1000);
  // Model parameters
  std::string default_path_to_engine("None");
  std::string path_to_engine;
  std::vector<std::string> default_class_map {std::string("object")};
  nh_.param("path_to_engine", det_p.engine_path, default_path_to_engine);
  nh_.param("num_classes", det_p.num_classes, 1);
  nh_.param("class_map", det_p.class_map, default_class_map);
  nh_.param("num_buffers", det_p.num_buffers, 2);

  buildDetect(glo_p, det_p, nms_p);

  image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &ROSDetect::imageCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  detection_pub_ = it_.advertise("/detection/raw_detection", 1);
#endif
  bboxes_pub_ = nh_.advertise<detect_and_track::BoundingBoxes2D>("/detection/bounding_boxes", 1);
}

ROSDetect::~ROSDetect() {
}

void ROSDetect::publishDetectionImage(cv::Mat& image, std::vector<std::vector<BoundingBox>>& bboxes) {
  generateDetectionImage(image, bboxes);
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image).toImageMsg();
  detection_pub_.publish(image_ptr_out_);
}

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

  buildLocate(glo_p, loc_p, cam_p);
  
  depth_sub_ = it_.subscribe("/camera/aligned_depth_to_color/image_raw", 1, &ROSDetectAndLocate::depthCallback, this);
  depth_info_sub_ = nh_.subscribe("/camera/aligned_depth_to_color/camera_info", 1, &ROSDetectAndLocate::depthInfoCallback, this);
#ifdef PUBLISH_DETECTION_IMAGE
  detection_pub_ = it_.advertise("/detection/raw_detection", 1);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  positions_bboxes_pub_ = nh_.advertise<detect_and_track::PositionBoundingBox2DArray>("/detection/positions_bboxes",1);
#else
  positions_pub_ = nh_.advertise<detect_and_track::PositionIDArray>("/detection/positions",1);
#endif
}

ROSDetectAndLocate::~ROSDetectAndLocate() {
}

void ROSDetectAndLocate::depthInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
  std::vector<double> P{msg->K[2], msg->K[5], msg->K[0], msg->K[4]};
  std::vector<float> Pf(msg->P.begin(), msg->P.end());
  std::vector<float> K(msg->D.begin(), msg->D.end());
  updateCameraInfo(Pf, K);
}

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
void ROSDetectAndLocate::publishDetectionsAndPositions(std::vector<std::vector<BoundingBox>>& bboxes,
                                                   std::vector<std::vector<std::vector<float>>>& points,
                                                   std_msgs::Header& header) {
  detect_and_track::PositionBoundingBox2DArray ros_bboxes;
  detect_and_track::PositionBoundingBox2D ros_bbox;
  std::vector<detect_and_track::PositionBoundingBox2D> vec_ros_bboxes;
  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      ros_bbox.bbox.min_x = bboxes[i][j].x_min_;
      ros_bbox.bbox.min_y = bboxes[i][j].y_min_;
      ros_bbox.bbox.height = bboxes[i][j].h_;
      ros_bbox.bbox.width = bboxes[i][j].w_;
      ros_bbox.bbox.class_id = i;
      ros_bbox.position.x = points[i][j][0];
      ros_bbox.position.y = points[i][j][1];
      ros_bbox.position.z = points[i][j][2];
      vec_ros_bboxes.push_back(ros_bbox);
    }
  }
  ros_bboxes.header.stamp = header.stamp;
  ros_bboxes.header.frame_id = header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;
  
  positions_bboxes_pub_.publish(ros_bboxes);
}

#else

void ROSDetectAndLocate::publishPositions(std::vector<std::vector<BoundingBox>>& bboxes,
                                          std::vector<std::vector<std::vector<float>>>& points,
                                          std_msgs::Header& header) {
  unsigned int counter = 0;
  detect_and_track::PositionIDArray id_positions;
  detect_and_track::PositionID id_position;
  std::vector<detect_and_track::PositionID> vec_id_positions;

  for (unsigned int i=0; i<bboxes.size(); i++) {
    for (unsigned int j=0; j<bboxes[i].size(); j++) {
      if (!bboxes[i][j].valid_) {
        continue;
      }
      id_position.position.x = points[i][j][0];
      id_position.position.y = points[i][j][1];
      id_position.position.z = points[i][j][2];
      id_position.detection_id = counter;
      counter ++;
      vec_id_positions.push_back(id_position);
    }
  }
  id_positions.header.stamp = header.stamp;
  id_positions.header.frame_id = header.frame_id;
  id_positions.positions = vec_id_positions;

  positions_pub_.publish(id_positions);
}
#endif


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

void ROSDetectTrack2DAndLocate::publishTrackerImage(cv::Mat& tracker_image,
                                                    std::vector<std::map<unsigned int, std::vector<float>>>& tracker_states) {
  generateTrackingImage(image_tracker, tracker_states);
  cv::cvtColor(image_tracker, image_tracker, cv::COLOR_RGB2BGR);
  std_msgs::Header image_ptr_out_header;
  image_ptr_out_header.stamp = ros::Time::now();
  image_ptr_out_ = cv_bridge::CvImage(image_ptr_out_header, "bgr8", image_tracker).toImageMsg();
  tracker_pub_.publish(image_ptr_out_);
}

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
  ros_bboxes.header.stamp = cv_ptr->header.stamp;
  ros_bboxes.header.frame_id = cv_ptr->header.frame_id;
  ros_bboxes.bboxes = vec_ros_bboxes;

  bboxes_pub_.publish(ros_bboxes);
}

void ROSDetectTrack2DAndLocate::publishPositions(std::vector<std::map<unsigned int, std::vector<float>>>& points,
                                                 std_msgs::Header& header) {
  geometry_msgs::PoseArray pose_array;
  geometry_msgs::Pose pose;
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
  cv::Mat image_tracker = image.clone();
  std::vector<std::vector<BoundingBox>> bboxes(num_classes_);
  std::vector<std::map<unsigned int, std::vector<float>>> tracker_states;
  std::vector<std::map<unsigned int, std::vector<float>>> points;
  std::vector<std::map<unsigned int, float>> distances;
  tracker_states.resize(num_classes_);
  detectObjects(image, bboxes);
  track(bboxes, tracker_states, dt_);
  locate(depth_image_, bboxes, distances, points);
#ifdef PROFILE
  auto end_inference = std::chrono::system_clock::now();
  ROS_INFO("Full inference done in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_tracking - start_image).count());
  printProfilingDetection();
  printProfilingTracking();
  printProfilingLocalization();
#endif

#ifdef PUBLISH_DETECTION_IMAGE
  publishDetectionImage(image, bboxes);
  publishTrackingImage(image, tracker_states);
#endif
#ifdef PUBLISH_DETECTION_WITH_POSITION
  publishDetectionsAndPositions(tracker_states, points, cv_ptr->header);
#else
  publishDetections(tracker_states, cv_ptr->header);
  publishPositions(points, cv_ptr->header);
#endif
}