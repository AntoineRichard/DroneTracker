#include <depth_image_extractor/KalmanFilter.h>

BaseKalmanFilter::BaseKalmanFilter() {
  // State is [x, y, z, vx, vy, vz, h, w]
}

BaseKalmanFilter::BaseKalmanFilter(const float& dt, const bool& use_dim, const bool& use_vel,
                                   const std::vector<float>& Q, const std::vector<float>& R) {}

BaseKalmanFilter::~BaseKalmanFilter() {
  printf("Base Kalman Filter Descructor");
}

KalmanFilter2D::~KalmanFilter2D() {
  printf("2D Kalman Filter Descructor");
}

void BaseKalmanFilter::getState(std::vector<float>& state) {
  state.resize(X_.size());
  for (unsigned int i=0; i < state.size(); i++){
    state[i] = X_(i);
  }
}

void BaseKalmanFilter::getUncertainty(std::vector<float>& uncertainty) {
  uncertainty.resize(X_.size());
  for (unsigned int i=0; i < uncertainty.size(); i++){
    uncertainty[i] = P_(i,i);
  }
}

void BaseKalmanFilter::predict(const float& dt) {
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Updating.\n", __func__, __LINE__); 
#endif
  updateF(dt);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d F matrix:\n", __func__, __LINE__);
  printF();
  printf("[DEBUG] KalmanFilter::%s::l%d X (state) before update:\n", __func__, __LINE__);
  printX();
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d X (state) after update:\n", __func__, __LINE__);
  printX();
#endif
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void BaseKalmanFilter::predict() {
  X_  = F_ * X_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void BaseKalmanFilter::correct(const std::vector<float>& Z){
  Eigen::MatrixXf Y, S, K, I_KH;
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Measuring.\n", __func__, __LINE__); 
#endif
  getMeasurement(Z);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Computing Y.\n", __func__, __LINE__); 
#endif
  Y = Z_ - H_ * X_;
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Computing S.\n", __func__, __LINE__); 
  printf("[DEBUG] KalmanFilter::%s::l%d R_ : %ld, %ld\n", __func__, __LINE__,R_.rows(), R_.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d H_ : %ld, %ld\n", __func__, __LINE__,H_.rows(), H_.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d P_ : %ld, %ld\n", __func__, __LINE__,P_.rows(), P_.cols());
#endif
  S = R_ + (H_ * (P_ * H_.transpose()));
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d S : %ld, %ld\n", __func__, __LINE__,S.rows(), S.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d Computing K.\n", __func__, __LINE__); 
#endif
  K = P_ * H_.transpose() * S.inverse();
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d K : %ld, %ld\n", __func__, __LINE__,K.rows(), K.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d Computing X.\n", __func__, __LINE__); 
#endif
  X_ = X_ + K * Y;
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d X_ : %ld, %ld\n", __func__, __LINE__,X_.rows(), X_.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d Computing I_KH.\n", __func__, __LINE__); 
#endif
  I_KH = I_ - K*H_;
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d IKH_ : %ld, %ld\n", __func__, __LINE__,I_KH.rows(), I_KH.cols());
  printf("[DEBUG] KalmanFilter::%s::l%d Computing P.\n", __func__, __LINE__); 
#endif
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d P_ : %ld, %ld\n", __func__, __LINE__,P_.rows(), P_.cols());
#endif
}

void BaseKalmanFilter::initialize(const std::vector<float>& Q, const std::vector<float>& R) {}
void BaseKalmanFilter::buildH(){}
void BaseKalmanFilter::buildR(const std::vector<float>& R){}
void BaseKalmanFilter::resetFilter(const std::vector<float>& initial_state) {}
void BaseKalmanFilter::updateF(const float& dt) {}
void BaseKalmanFilter::getMeasurement(const std::vector<float>& measurement) {}
void BaseKalmanFilter::printF() {}
void BaseKalmanFilter::printX() {}

KalmanFilter2D::KalmanFilter2D() {
  // State is [u, v, vu, vv, h, w]
}

KalmanFilter2D::KalmanFilter2D(const float& dt, const bool& use_dim, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, vu, vv, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_vel_ = use_vel;
  initialize(Q,R);
}

void KalmanFilter2D::printF() {
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5));
}

void KalmanFilter2D::printX() {
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5));
}

void KalmanFilter2D::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, vu ,vv, h, w]
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Creating new 2D Kalman filter.\n", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(6);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting state to 0.\n", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0;

  // Process noise
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting Q.\n", __func__, __LINE__); 
#endif
  Q_ = Eigen::MatrixXf::Zero(6,6);
  Q_(0,0) = Q[0];
  Q_(1,1) = Q[1];
  Q_(2,2) = Q[2];
  Q_(3,3) = Q[3];
  Q_(4,4) = Q[4];
  Q_(5,5) = Q[5];

  // Dynamics
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting F.\n", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(6,6);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting P.\n", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(6,6);
  P_ = Q_*Q_;

  // Helper matrices
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting I.\n", __func__, __LINE__); 
#endif
  I_ = Eigen::MatrixXf::Identity(6,6);

  // Observation noise
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting R.\n", __func__, __LINE__); 
#endif
  buildR(R);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting H.\n", __func__, __LINE__); 
#endif
  buildH();
}

void KalmanFilter2D::buildH(){
  int h_size = 2;
  if (use_vel_) {
    h_size += 2;
  }
  if (use_dim_) {
    h_size += 2;
  }
  H_ = Eigen::MatrixXf(h_size, 6);
  Eigen::MatrixXf H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
  H_pos = Eigen::MatrixXf::Zero(2,6);
  H_vel = Eigen::MatrixXf::Zero(2,6);
  H_hw = Eigen::MatrixXf::Zero(2,6);

  H_pos << 1.0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0;

  H_vel << 0, 0, 1.0, 0, 0, 0,
            0, 0, 0, 1.0, 0, 0;

  H_hw << 0, 0, 0, 0, 1.0, 0,
           0, 0, 0, 0, 0, 1.0;

  H_ = Eigen::MatrixXf::Zero(h_size,6);
  if (use_vel_){
    if (use_dim_){
      H_ << H_pos, H_vel, H_hw;
    } else {
      H_ << H_pos, H_vel;
    }
  } else {
    if (use_dim_){
      H_ << H_pos, H_hw;
    } else {
      H_ << H_pos;
    }
  }
}

void KalmanFilter2D::buildR(const std::vector<float>& R){
  int r_size = 2;
  if (use_vel_) {
    r_size +=2;
  }
  if (use_dim_) {
    r_size += 2;
  }
  Z_ = Eigen::VectorXf::Zero(r_size);
  R_ = Eigen::MatrixXf::Zero(r_size, r_size);
  R_(0,0) = R[0];
  R_(1,1) = R[1];
  if (use_vel_) {
    if (use_dim_) {
      R_(2,2) = R[2];
      R_(3,3) = R[3];
      R_(4,4) = R[4];
      R_(5,5) = R[5]; 
    } else {
      R_(2,2) = R[2];
      R_(3,3) = R[3];
    }
  } else {
    if (use_dim_) {
      R_(2,2) = R[4];
      R_(3,3) = R[5];
    }
  }
}

void KalmanFilter2D::resetFilter(const std::vector<float>& initial_state) {
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Reinitializing state and covariance.\n", __func__, __LINE__); 
  printf("[DEBUG] KalmanFilter::%s::l%d Setting state.\n", __func__, __LINE__); 
#endif
  X_ << initial_state[0],
        initial_state[1],
        initial_state[2],
        initial_state[3],
        initial_state[4],
        initial_state[5];

#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting P_.\n", __func__, __LINE__); 
#endif
  P_ = Q_*Q_;
}

void KalmanFilter2D::getMeasurement(const std::vector<float>& measurement) {
  // State is [u, v, vu, vv, h, w]
  Z_(0) = measurement[0];
  Z_(1) = measurement[1];
  if (use_vel_) {
    if (use_dim_) {
      Z_(2) = measurement[2];
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
      Z_(5) = measurement[5];
    } else {
      Z_(2) = measurement[2];
      Z_(3) = measurement[3];
    }
  } else {
    if (use_dim_) {
      Z_(2) = measurement[4];
      Z_(3) = measurement[5];
    }
  }
}

void KalmanFilter2D::updateF(const float& dt) {
  F_(0,2) = dt;
  F_(1,3) = dt;
}


KalmanFilter3D::KalmanFilter3D() {
  // State is [x, y, z, vx, vy, vz, h, w]
}

KalmanFilter3D::KalmanFilter3D(const float& dt, const bool& use_dim, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx, vy, vz, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_vel_ = use_vel_;
  initialize(Q,R);
}

void KalmanFilter3D::printF() {
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5), F_(0,6), F_(0,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5), F_(1,6), F_(1,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5), F_(2,6), F_(2,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5), F_(3,6), F_(3,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5), F_(4,6), F_(4,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5), F_(5,6), F_(5,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(6,0), F_(6,1), F_(6,2), F_(6,3), F_(6,4), F_(6,5), F_(6,6), F_(6,7));
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(7,0), F_(7,1), F_(7,2), F_(7,3), F_(7,4), F_(7,5), F_(7,6), F_(7,7));
}

void KalmanFilter3D::printX() {
  printf("[DEBUG] KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5), X_(6), X_(7));
}

void KalmanFilter3D::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx ,vy, vz, h, w]
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Creating new 3D Kalman filter.\n", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(8);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting state to 0.\n", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0, 0, 0;

  // Process noise
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting Q.\n", __func__, __LINE__); 
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
  printf("[DEBUG] KalmanFilter::%s::l%d Setting F.\n", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(8,8);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting P.\n", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(8,8);
  P_ = Q_*Q_;

  // Helper matrices
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting I.\n", __func__, __LINE__); 
#endif
  I_ = Eigen::MatrixXf::Identity(8,8);

  // Observation noise
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting R.\n", __func__, __LINE__); 
#endif
  buildR(R);
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Setting H.\n", __func__, __LINE__); 
#endif
  buildH();
}

void KalmanFilter3D::buildH(){
  int h_size = 3;
  if (use_vel_) {
    h_size += 3;
  }
  if (use_dim_) {
    h_size += 2;
  }
  H_ = Eigen::MatrixXf(h_size, 8);
  Eigen::MatrixXf H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
  H_pos = Eigen::MatrixXf::Zero(3,8);
  H_vel = Eigen::MatrixXf::Zero(3,8);
  H_hw = Eigen::MatrixXf::Zero(2,8);

  H_pos << 1.0, 0, 0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0, 0, 0,
            0, 0, 1.0, 0, 0, 0, 0, 0;

  H_vel << 0, 0, 0, 1.0, 0, 0, 0, 0,
            0, 0, 0, 0, 1.0, 0, 0, 0,
            0, 0, 0, 0, 0, 1.0, 0, 0; 

  H_hw << 0, 0, 0, 0, 0, 0, 1.0, 0,
           0, 0, 0, 0, 0, 0, 0, 1.0;

  H_ = Eigen::MatrixXf::Zero(h_size,8);
  if (use_vel_){
      if (use_dim_){
        H_ << H_pos, H_vel, H_hw;
      } else {
        H_ << H_pos, H_vel;
      }
  } else {
      if (use_dim_){
        H_ << H_pos, H_hw;
      } else {
        H_ << H_pos;
      }
  }
}

void KalmanFilter3D::buildR(const std::vector<float>& R){
  int r_size = 3;
  if (use_vel_) {
    r_size +=3;
  }
  if (use_dim_) {
    r_size += 2;
  }
  Z_ = Eigen::VectorXf::Zero(r_size);
  R_ = Eigen::MatrixXf::Zero(r_size, r_size);
  R_(0,0) = R[0];
  R_(1,1) = R[1];
  R_(2,2) = R[2];
  if (use_vel_) {
    if (use_dim_) {
      R_(3,3) = R[3];
      R_(4,4) = R[4];
      R_(5,5) = R[5];
      R_(6,6) = R[6];
      R_(7,7) = R[7];
    } else {
      R_(3,3) = R[3];
      R_(4,4) = R[4];
      R_(5,5) = R[5];
    }
  } else {
    if (use_dim_) {
      R_(3,3) = R[6];
      R_(4,4) = R[7]; 
    }
  }
}

void KalmanFilter3D::resetFilter(const std::vector<float>& initial_state) {
#ifdef DEBUG_KALMAN
  printf("[DEBUG] KalmanFilter::%s::l%d Reinitializing state and covariance.\n", __func__, __LINE__); 
  printf("[DEBUG] KalmanFilter::%s::l%d Setting state.\n", __func__, __LINE__); 
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
  printf("[DEBUG] KalmanFilter::%s::l%d Setting P_.\n", __func__, __LINE__); 
#endif
  P_ = Q_*Q_;
}

void KalmanFilter3D::getMeasurement(const std::vector<float>& measurement) {
  // State is [x, y, z, vx, vy, vz, h, w]
  Z_(0) = measurement[0];
  Z_(1) = measurement[1];
  Z_(2) = measurement[2];
  if (use_vel_) {
    if (use_dim_) {
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
      Z_(5) = measurement[5];
      Z_(6) = measurement[6];
      Z_(7) = measurement[7];
    } else {
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
      Z_(5) = measurement[5];
    }
  } else {
    if (use_dim_) {
      Z_(3) = measurement[6];
      Z_(4) = measurement[7];
    }
  }
}

void KalmanFilter3D::updateF(const float& dt) {
  F_(0,3) = dt;
  F_(1,4) = dt;
  F_(2,5) = dt;
}