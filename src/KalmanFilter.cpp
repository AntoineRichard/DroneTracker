#include <depth_image_extractor/KalmanFilter.h>

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