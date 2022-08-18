#include <depth_image_extractor/KalmanFilter.h>

#ifdef DEBUG_KALMAN
#include <ros/ros.h>
#endif

KalmanFilter::KalmanFilter() {
  // State is [x, y, z, vx, vy, vz, h, w]
}

KalmanFilter::KalmanFilter(const float& dt, const bool& use_dim, const bool& use_z, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx, vy, vz, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_z_ = use_z;
  use_vel_ = use_vel_;
  initialize(Q,R);
}

void KalmanFilter::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx ,vy, vz, h, w]
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Creating new Kalman filter.", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(8);
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting state to 0.", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0, 0, 0;

  // Process noise
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting Q.", __func__, __LINE__); 
#endif
  Q_ = Eigen::MatrixXf::Zero(8,8);
  Q_(0,0) = Q[0];
  Q_(1,1) = Q[1];
  Q_(2,2) = Q[2];
  Q_(3,3) = Q[3];
  Q_(4,4) = Q[4];
  Q_(5,5) = Q[5];
  Q_(6,6) = Q[6];
  Q_(7,7) = Q[7];

  // Dynamics
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting F.", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(8,8);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting P.", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(8,8);
  P_ = Q_*Q_;

  // Helper matrices
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting I8.", __func__, __LINE__); 
#endif
  I8_ = Eigen::MatrixXf::Identity(8,8);

  // Observation noise
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting R.", __func__, __LINE__); 
#endif
  buildR(R);
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting H.", __func__, __LINE__); 
#endif
  buildH();
}

void KalmanFilter::buildH(){
  int h_size = 2;
  if (use_vel_) {
    h_size += 2;
  }
  if (use_z_) {
    h_size ++;
    if (use_vel_) {
      h_size ++;
    }
  }
  if (use_dim_) {
    h_size += 2; // Or 3?
  }
  H_ = Eigen::MatrixXf(h_size, 8);
  Eigen::MatrixXf H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
  H_pos = Eigen::MatrixXf::Zero(2,8);
  H_pos_z = Eigen::MatrixXf::Zero(1,8);
  H_vel = Eigen::MatrixXf::Zero(2,8);
  H_vel_z = Eigen::MatrixXf::Zero(1,8);
  H_hw = Eigen::MatrixXf::Zero(2,8); // Why 3x8 -> area

  H_pos << 1.0, 0, 0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0, 0, 0;
  
  H_pos_z << 0, 0, 1.0, 0, 0, 0, 0, 0;

  H_vel << 0, 0, 0, 1.0, 0, 0, 0, 0,
            0, 0, 0, 0, 1.0, 0, 0, 0;
  
  H_vel_z << 0, 0, 0, 0, 0, 1.0, 0, 0; 

  H_hw << 0, 0, 0, 0, 0, 0, 1.0, 0,
           0, 0, 0, 0, 0, 0, 0, 1.0;

  H_ = Eigen::MatrixXf::Zero(h_size,8);
  if (use_vel_){
    if (use_z_){
      if (use_dim_){
        H_ << H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
      } else {
        H_ << H_pos, H_pos_z, H_vel, H_vel_z;
      }
    } else {  
      if (use_dim_){
        H_ << H_pos, H_vel, H_hw;
      } else {
        H_ << H_pos, H_vel;
      }
    }
  } else {
    if (use_z_){
      if (use_dim_){
        H_ << H_pos, H_pos_z, H_hw;
      } else {
        H_ << H_pos, H_pos_z;
      }
    } else {  
      if (use_dim_){
        H_ << H_pos, H_hw;
      } else {
        H_ << H_pos;
      }
    }
  }
}

void KalmanFilter::buildR(const std::vector<float>& R){
  int r_size = 2;
  if (use_vel_) {
    r_size +=2;
  }
  if (use_z_) {
    r_size ++;
    if (use_vel_) {
      r_size ++;
    }
  }
  if (use_dim_) {
    r_size += 2; // Or 3?
  }
  Z_ = Eigen::VectorXf::Zero(r_size);
  R_ = Eigen::MatrixXf::Zero(r_size, r_size);
  R_(0,0) = R[0];
  R_(1,1) = R[1];
  if (use_vel_) {
    if (use_z_){
      R_(2,2) = R[2];
      R_(3,3) = R[3];
      R_(4,4) = R[4];
      R_(5,5) = R[5];
      if (use_dim_){
        R_(6,6) = R[6];
        R_(7,7) = R[7];
      }
    } else {  
      R_(2,2) = R[3];
      R_(3,3) = R[4];
      if (use_dim_){
        R_(4,4) = R[6];
        R_(5,5) = R[7];
      }
    }
  } else {
    if (use_z_){
      R_(2,2) = R[2];
      if (use_dim_){
        R_(3,3) = R[6];
        R_(4,4) = R[7];
      }
    } else {  
      if (use_dim_){
        R_(2,2) = R[6];
        R_(3,3) = R[7];
      }
    }
  }
}

void KalmanFilter::resetFilter(const std::vector<float>& initial_state) {
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Reinitializing state and covariance.", __func__, __LINE__); 
  ROS_INFO("KalmanFilter::%s::l%d Setting state.", __func__, __LINE__); 
#endif
  X_ << initial_state[0],
        initial_state[1],
        initial_state[2],
        initial_state[3],
        initial_state[4],
        initial_state[5],
        initial_state[6],
        initial_state[7];

#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Setting P_.", __func__, __LINE__); 
#endif
  P_ = Q_*Q_;
}

void KalmanFilter::getMeasurement(const std::vector<float>& measurement) {
  // State is [x, y, z, vx, vy, vz, h, w]
  Z_(0) = measurement[0];
  Z_(1) = measurement[1];
  if (use_vel_) {
    if (use_z_) {
      Z_(2) = measurement[2];
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
      Z_(5) = measurement[5];
      if (use_dim_) {
        Z_(6) = measurement[6];
        Z_(7) = measurement[7];
      }
    } else {
      Z_(2) = measurement[3];
      Z_(3) = measurement[4];
      if (use_dim_) {
        Z_(4) = measurement[6];
        Z_(5) = measurement[7];
      }
    }
  } else {
    if (use_z_) {
      Z_(2) = measurement[2];
      if (use_dim_) {
        Z_(3) = measurement[6];
        Z_(4) = measurement[7];
      }
    } else {
      if (use_dim_) {
        Z_(2) = measurement[6];
        Z_(3) = measurement[7];
      }
    }
  }
}

void KalmanFilter::updateF(const float& dt) {
  F_(0,3) = dt;
  F_(1,4) = dt;
  F_(2,5) = dt;
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

void KalmanFilter::predict(const float& dt) {
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Updating.", __func__, __LINE__); 
#endif
  updateF(dt);
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d F matrix:", __func__, __LINE__);
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5), F_(0,6), F_(0,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5), F_(1,6), F_(1,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5), F_(2,6), F_(2,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5), F_(3,6), F_(3,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5), F_(4,6), F_(4,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5), F_(5,6), F_(5,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(6,0), F_(6,1), F_(6,2), F_(6,3), F_(6,4), F_(6,5), F_(6,6), F_(6,7));
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, F_(7,0), F_(7,1), F_(7,2), F_(7,3), F_(7,4), F_(7,5), F_(7,6), F_(7,7));
  ROS_INFO("KalmanFilter::%s::l%d X (state) before update:", __func__, __LINE__);
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5), X_(6), X_(7));
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d X (state) after update:", __func__, __LINE__);
  ROS_INFO("KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5), X_(6), X_(7));
#endif
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::predict() {
  X_  = F_ * X_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::correct(const std::vector<float>& Z){
  Eigen::MatrixXf Y, S, K, I_KH;
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Measuring.", __func__, __LINE__); 
#endif
  getMeasurement(Z);
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Computing Y.", __func__, __LINE__); 
#endif
  Y = Z_ - H_ * X_;
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d Computing S.", __func__, __LINE__); 
  ROS_INFO("KalmanFilter::%s::l%d R_ : %d, %d", __func__, __LINE__,R_.rows(), R_.cols());
  ROS_INFO("KalmanFilter::%s::l%d H_ : %d, %d", __func__, __LINE__,H_.rows(), H_.cols());
  ROS_INFO("KalmanFilter::%s::l%d P_ : %d, %d", __func__, __LINE__,P_.rows(), P_.cols());
#endif
  S = R_ + (H_ * (P_ * H_.transpose()));
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d S : %d, %d", __func__, __LINE__,S.rows(), S.cols());
  ROS_INFO("KalmanFilter::%s::l%d Computing K.", __func__, __LINE__); 
#endif
  K = P_ * H_.transpose() * S.inverse();
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d K : %d, %d", __func__, __LINE__,K.rows(), K.cols());
  ROS_INFO("KalmanFilter::%s::l%d Computing X.", __func__, __LINE__); 
#endif
  X_ = X_ + K * Y;
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d X_ : %d, %d", __func__, __LINE__,X_.rows(), X_.cols());
  ROS_INFO("KalmanFilter::%s::l%d Computing I_KH.", __func__, __LINE__); 
#endif
  I_KH = I8_ - K*H_;
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d IKH_ : %d, %d", __func__, __LINE__,I_KH.rows(), I_KH.cols());
  ROS_INFO("KalmanFilter::%s::l%d Computing P.", __func__, __LINE__); 
#endif
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
#ifdef DEBUG_KALMAN
  ROS_INFO("KalmanFilter::%s::l%d P_ : %d, %d", __func__, __LINE__,P_.rows(), P_.cols());
#endif
}