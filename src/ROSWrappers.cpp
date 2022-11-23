#include <detect_and_track/ROSWrappers.h>

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
  image_ok_ = false;
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
  if (image_ok_) {
    cv::Mat image_tracker = image_.clone();
    publishTrackingImage(image_tracker, tracker_states);
  }
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
  image_ok_ = true;
}