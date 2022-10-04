/**
 * @file KalmanFilter.cpp
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief The source code of the Kalman filter classes.
 * @details This file implements a set of kalman filter for object tracking.
 */

#include <detect_and_track/KalmanFilter.h>

/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
BaseKalmanFilter::BaseKalmanFilter() {}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param dt The default time-delta in between two updates.
 * @param use_dim A flag to indicate if the filter should observe the height and width of the tracked object.
 * @param use_vel A flag to indicate if the filter should observe the velocity of the tracked object.
 * @param Q A reference to a vector of floats containing the process noise for each variable.
 * @param R A reference to a vector of floats containing the observation noise for each measurement.
 */
BaseKalmanFilter::BaseKalmanFilter(const float& dt, const bool& use_dim, const bool& use_vel,
                                   const std::vector<float>& Q, const std::vector<float>& R) {}

/**
 * @brief Default destructor.
 * @details Default destructor.
 * 
 */
BaseKalmanFilter::~BaseKalmanFilter() {}

/**
 * @brief Accessor function to get the current state.
 * @details Accessor function to get the current state.
 * 
 * @param state A reference to the vector containing the state.
 */
void BaseKalmanFilter::getState(std::vector<float>& state) {
  state.resize(X_.size());
  for (unsigned int i=0; i < state.size(); i++){
    state[i] = X_(i);
  }
}

/**
 * @brief Accessor function to get the current uncertainty on the state.
 * @details Accessor function to get the current uncertainty on the state.
 * 
 * @param uncertainty A reference to the vector containing the uncertainty.
 */
void BaseKalmanFilter::getUncertainty(std::vector<float>& uncertainty) {
  uncertainty.resize(X_.size());
  for (unsigned int i=0; i < uncertainty.size(); i++){
    uncertainty[i] = P_(i,i);
  }
}

/**
 * @brief The prediction function of the linear Kalman filter.
 * @details This function implements the prediction step of a Kalman Filter.
 * First F is updated, as well as the state X using the following formula:
 * X = F*X, where F is the dynamics of the system expressed as a Matrix.
 * Then P, the uncertainty on the state is updated:
 * P = F*P*F^T + Q, where Q is the process noise.
 * 
 * @param dt The time delta in second in between two updates.
 */
void BaseKalmanFilter::predict(const float& dt) {
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Updating.\n", __func__, __LINE__); 
#endif
  updateF(dt);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d F matrix:\n", __func__, __LINE__);
  printF();
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) before update:\n", __func__, __LINE__);
  printX();
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) after update:\n", __func__, __LINE__);
  printX();
#endif
  P_ = F_ * P_ * F_.transpose() + Q_;
}

/**
 * @brief The prediction function of the linear Kalman filter.
 * @details This function implements the prediction step of a Kalman filter.
 * First the state X is updated using the following formula:
 * X = F*X, where F is the dynamics of the system expressed as a Matrix.
 * Then P, the uncertainty on the state, is updated:
 * P = F*P*F^T + Q, where Q is the process noise.
 * 
 */
void BaseKalmanFilter::predict() {
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d F matrix:\n", __func__, __LINE__);
  printF();
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) before update:\n", __func__, __LINE__);
  printX();
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) after update:\n", __func__, __LINE__);
  printX();
#endif
  P_ = F_ * P_ * F_.transpose() + Q_;
}

/**
 * @brief The correction function of the linear Kalman filter.
 * @details This function implements the correction step of a Kalman filter.
 * First, the measurment Z is acquired, and the innovation Y is computed using the following formula:
 * Y = Z - H*X, where H is the 
 * Then, X is corrected using the new measurement:
 * S = R + H*P*H^T, where R is the noise on the measurment
 * K = P*H*S^-1
 * X = X + K*Y
 * Finally the uncertainty on the state is adjusted:
 * IKH = I - K*H
 * P = IKH*P*IKH^T + K*R*K^T
 * 
 * @param Z The reference to the measurement vector.
 */
void BaseKalmanFilter::correct(const std::vector<float>& Z){
  Eigen::MatrixXf Y, S, K, I_KH;
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Measuring.\n", __func__, __LINE__); 
#endif
  getMeasurement(Z);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing Y.\n", __func__, __LINE__); 
#endif
  Y = Z_ - H_ * X_;
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing S.\n", __func__, __LINE__); 
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d R_ : %ld, %ld\n", __func__, __LINE__,R_.rows(), R_.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d H_ : %ld, %ld\n", __func__, __LINE__,H_.rows(), H_.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d P_ : %ld, %ld\n", __func__, __LINE__,P_.rows(), P_.cols());
#endif
  S = R_ + (H_ * (P_ * H_.transpose()));
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d S : %ld, %ld\n", __func__, __LINE__,S.rows(), S.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing K.\n", __func__, __LINE__); 
#endif
  K = P_ * H_.transpose() * S.inverse();
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d K : %ld, %ld\n", __func__, __LINE__,K.rows(), K.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing X.\n", __func__, __LINE__); 
#endif
  X_ = X_ + K * Y;
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X_ : %ld, %ld\n", __func__, __LINE__,X_.rows(), X_.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing I_KH.\n", __func__, __LINE__); 
#endif
  I_KH = I_ - K*H_;
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d IKH_ : %ld, %ld\n", __func__, __LINE__,I_KH.rows(), I_KH.cols());
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Computing P.\n", __func__, __LINE__); 
#endif
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d P_ : %ld, %ld\n", __func__, __LINE__,P_.rows(), P_.cols());
#endif
}

/**
 * @brief Instantiates all the variables in the filter. 
 * @details This function instantiates the following variables:
 * X the state, P the covariance, Q the process noise, R the observation noise, F the dynamics,
 * H the observation matrix, and I an identity matrix.
 * To be implemented in child classes.
 * 
 * @param Q The reference to the process noise vector.
 * @param R The reference to the measurement noise vector.
 */
void BaseKalmanFilter::initialize(const std::vector<float>& Q, const std::vector<float>& R) {}

/**
 * @brief Instantiates the observation matrix H. 
 * @details The function builds the observatiin matrix H.
 * To be implemented in child classes.

 * @param R The reference to the vector containing the noise of the measurement.
 */
void BaseKalmanFilter::buildH(){}

/**
 * @brief Instantiates the measurement noise R. 
 * @details The function builds the measurement noise matrix R.
 * To be implemented in child classes.

 * @param R The reference to the vector containing the noise of the measurement.
 */
void BaseKalmanFilter::buildR(const std::vector<float>& R){}

/**
 * @brief Resets the state and covariance.
 * @details This function resets the state and covariance of the filter.
 * The state is initialiazed to the value given by the user, the covariance is set to Q*Q.
 * 
 * @param initial_state The refence to the vector containing the value of the state should be reseted to.
 */
void BaseKalmanFilter::resetFilter(const std::vector<float>& initial_state) {
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Reinitializing state and covariance.\n", __func__, __LINE__); 
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting state.\n", __func__, __LINE__); 
#endif
  for (unsigned int i=0; i<initial_state.size(); i++) {
    X_(i) = initial_state[i];
  }
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting P_.\n", __func__, __LINE__); 
#endif
  P_ = Q_*Q_;
}

/**
 * @brief Updates the dynamics based on dt.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * To be implemented in child classes.
 * 
 * @param dt 
 */
void BaseKalmanFilter::updateF(const float& dt) {}

void BaseKalmanFilter::getMeasurement(const std::vector<float>& measurement) {}

/**
 * @brief Helper function to display the matrix F.
 * @details Helper function to display the matrix F.
 * To be implemented in child classes.
 * 
 */
void BaseKalmanFilter::printF() {}

/**
 * @brief Helper function to display the state X.
 * @details Helper function to display the state X.
 * To be implemented in child classes.
 * 
 */
void BaseKalmanFilter::printX() {}

/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
BaseExtendedKalmanFilter::BaseExtendedKalmanFilter() {}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param dt The default time-delta in between two updates.
 * @param use_dim A flag to indicate if the filter should observe the height and width of the tracked object.
 * @param use_vel A flag to indicate if the filter should observe the velocity of the tracked object.
 * @param Q A reference to a vector of floats containing the process noise for each variable.
 * @param R A reference to a vector of floats containing the observation noise for each measurement.
 */
BaseExtendedKalmanFilter::BaseExtendedKalmanFilter(const float& dt, const bool& use_dim, const bool& use_vel,
                                   const std::vector<float>& Q, const std::vector<float>& R) {}

/**
 * @brief Default destructor.
 * @details Default destructor.
 * 
 */
BaseExtendedKalmanFilter::~BaseExtendedKalmanFilter() {}

/**
 * @brief The prediction function of the linear Kalman filter.
 * @details This function implements the prediction step of a Kalman Filter.
 * First F is updated, as well as the state X using the following formula:
 * X = F*X, where F is the dynamics of the system expressed as a Matrix.
 * Then P, the uncertainty on the state is updated:
 * P = F*P*F^T + Q, where Q is the process noise.
 * 
 * @param dt The time delta in second in between two updates.
 */
void BaseExtendedKalmanFilter::predict(const float& dt) {
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Updating.\n", __func__, __LINE__); 
#endif
  updatedFdX(dt);
  updateF(dt);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d F matrix:\n", __func__, __LINE__);
  printF();
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) before update:\n", __func__, __LINE__);
  printX();
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) after update:\n", __func__, __LINE__);
  printX();
#endif
  P_ = dFdX_ * P_ * dFdX_.transpose() + Q_;
}

/**
 * @brief The prediction function of the linear Kalman filter.
 * @details This function implements the prediction step of a Kalman filter.
 * First the state X is updated using the following formula:
 * X = F*X, where F is the dynamics of the system expressed as a Matrix.
 * Then P, the uncertainty on the state, is updated:
 * P = F*P*F^T + Q, where Q is the process noise.
 * 
 */
void BaseExtendedKalmanFilter::predict() {
  updatedFdX();
  updateF();
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d F matrix:\n", __func__, __LINE__);
  printF();
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) before update:\n", __func__, __LINE__);
  printX();
#endif
  X_  = F_ * X_;
# ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d X (state) after update:\n", __func__, __LINE__);
  printX();
#endif
  P_ = dFdX_ * P_ * dFdX_.transpose() + Q_;
}

/**
 * @brief Instantiates all the variables in the filter. 
 * @details This function instantiates the following variables:
 * X the state, P the covariance, Q the process noise, R the observation noise, F the dynamics,
 * H the observation matrix, and I an identity matrix.
 * To be implemented in child classes.
 * 
 * @param Q The reference to the process noise vector.
 * @param R The reference to the measurement noise vector.
 */
//void BaseExtendedKalmanFilter::initialize(const std::vector<float>& Q, const std::vector<float>& R) {}

/**
 * @brief Instantiates the observation matrix H. 
 * @details The function builds the observatiin matrix H.
 * To be implemented in child classes.

 * @param R The reference to the vector containing the noise of the measurement.
 */
//void BaseExtendedKalmanFilter::buildH(){}

/**
 * @brief Instantiates the measurement noise R. 
 * @details The function builds the measurement noise matrix R.
 * To be implemented in child classes.

 * @param R The reference to the vector containing the noise of the measurement.
 */
//void BaseExtendedKalmanFilter::buildR(const std::vector<float>& R){}

/**
 * @brief Resets the state and covariance.
 * @details This function resets the state and covariance of the filter.
 * The state is initialiazed to the value given by the user, the covariance is set to Q*Q.
 * 
 * @param initial_state The refence to the vector containing the value of the state should be reseted to.
 */
//void BaseExtendedKalmanFilter::resetFilter(const std::vector<float>& initial_state) {}

/**
 * @brief Updates the dynamics based on dt.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * To be implemented in child classes.
 * 
 * @param dt 
 */
//void BaseExtendedKalmanFilter::updateF(const float& dt) {}

/**
 * @brief Updates the derivative of the dynamics.
 * @details Updates the value of the dynamics' derivative.
 * To be implemented in child classes.
 * 
 * @param dt 
 */
void BaseExtendedKalmanFilter::updatedFdX(const float& dt) {}

/**
 * @brief Updates the derivative of the dynamics.
 * @details Updates the value of the dynamics' derivative.
 * To be implemented in child classes.
 * 
 */
void BaseExtendedKalmanFilter::updatedFdX() {}

/**
 * @brief Updates the dynamics.
 * @details Updates the value of the dynamics'.
 * To be implemented in child classes.
 * 
 * @param dt
 */
void BaseExtendedKalmanFilter::updateF(const float& dt) {}

/**
 * @brief Updates the dynamics.
 * @details Updates the value of the dynamics'.
 * To be implemented in child classes.
 * 
 */
void BaseExtendedKalmanFilter::updateF() {}

//void BaseExtendedKalmanFilter::getMeasurement(const std::vector<float>& measurement) {}

/**
 * @brief Helper function to display the matrix dFdX.
 * @details Helper function to display the matrix dFdX.
 * To be implemented in child classes.
 * 
 */
void BaseExtendedKalmanFilter::printdFdX() {}

/**
 * @brief Helper function to display the matrix F.
 * @details Helper function to display the matrix F.
 * To be implemented in child classes.
 * 
 */
//void BaseExtendedKalmanFilter::printF() {}

/**
 * @brief Helper function to display the state X.
 * @details Helper function to display the state X.
 * To be implemented in child classes.
 * 
 */
//void BaseExtendedKalmanFilter::printX() {}

/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
KalmanFilter2D::KalmanFilter2D() {
  // State is [u, v, vu, vv, h, w]
}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param dt The default time-delta in between two updates.
 * @param use_dim A flag to indicate if the filter should observe the height and width of the tracked object.
 * @param use_vel A flag to indicate if the filter should observe the velocity of the tracked object.
 * @param Q A reference to a vector of floats containing the process noise for each variable (R6).
 * @param R A reference to a vector of floats containing the observation noise for each measurement (R6).
 */
KalmanFilter2D::KalmanFilter2D(const float& dt, const bool& use_dim, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, vu, vv, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_vel_ = use_vel;
  initialize(Q,R);
}

/**
 * @brief Default destructor.
 * @details Default destructor.
 * 
 */
KalmanFilter2D::~KalmanFilter2D() {}

/**
 * @brief Helper function to display the matrix F.
 * @details Helper function to display the matrix F.
 * 
 */
void KalmanFilter2D::printF() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5));
}

/**
 * @brief Helper function to display the state X.
 * @details Helper function to display the state X.
 * 
 */
void KalmanFilter2D::printX() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5));
}

/**
 * @brief Instantiates all the variables in the filter. 
 * @details This function instantiates the following variables:
 * X the state (R6), P the covariance (R6x6), Q the process noise (R6x6), R the observation noise (R?x?), F the dynamics (R6x6),
 * H the observation matrix (R?x6), and I an identity matrix (R6x6).
 * F, the dynamics is computed using the following equations:
 * x_t+1 = x_t + vx_t * dt
 * y_t+1 = y_t + vy_t * dt
 * vx_t+1 = vx_t
 * vy_t+1 = vy_t
 * h_t+1 = h_t
 * w_t+1 = w_t
 * 
 * @param Q The reference to the process noise vector (R6).
 * @param R The reference to the measurement noise vector (R6).
 */
void KalmanFilter2D::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, vu ,vv, h, w]
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Creating new 2D Kalman filter.\n", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(6);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting state to 0.\n", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0; // state is initialized a 0

  // Process noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting Q.\n", __func__, __LINE__); 
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
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting F.\n", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(6,6);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting P.\n", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(6,6);
  P_ = Q_*Q_; // P is initialialized at Q*Q

  // Helper matrices
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting I.\n", __func__, __LINE__); 
#endif
  I_ = Eigen::MatrixXf::Identity(6,6);

  // Observation noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting R.\n", __func__, __LINE__); 
#endif
  buildR(R); // R is modular and depends on user choices. (use_dim, use_vel)
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting H.\n", __func__, __LINE__); 
#endif
  buildH(); // H is modular and depends on user choices. (use_dim, use_vel)
}

/**
 * @brief Instantiates the Observation matrix H.
 * @details This function builds the observation matrix H based on the selected parameters.
 * If the user selected use_vel, then the velocities will be observed and hence included in the observation matrix.
 * If the user selected use_dim, then the height and width will be observed and hence included in the observation matrix.
 * 
 */
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

/**
 * @brief Instantiates the measurement noise R. 
 * @details The function builds the measurement noise matrix R.
 * The size of this matrix changes based on the selected parameters parameters.
 * If the user selected use_vel, then the noise on the observed velocities will be added to the matrix.
 * If the user selected use_dim, then the noise on the observed height and width will be added to the matrix.
 * 
 * @param R The reference to the vector containing the noise of the measurement (R6).
 */
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

/**
 * @brief Converts the std::vector to a eigen::MatrixXf. 
 * @details Converts the measurement (std::vector<float>) into a eigen::MatrixXf.
 * It also ensures that the correct quantities are being observed.
 * The user must give a vector of full size even if the values are not observed (use 0s instead).
 * 
 * @param measurement The reference to the measurement vector (R6).
 */
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

/**
 * @brief Updates the dynamics based on dt.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * We recall that x_t+1 = x_t + vx_t * dt, x_t being the position at time t, and vx_t being the velocity at time t.
 * 
 * @param dt 
 */
void KalmanFilter2D::updateF(const float& dt) {
  F_(0,2) = dt;
  F_(1,3) = dt;
}

/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
KalmanFilter2DH::KalmanFilter2DH() {
  // State is [u, v, theta, v, vtheta, h, w]
}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param dt The default time-delta in between two updates.
 * @param use_dim A flag to indicate if the filter should observe the height and width of the tracked object.
 * @param use_vel A flag to indicate if the filter should observe the velocity of the tracked object.
 * @param Q A reference to a vector of floats containing the process noise for each variable (R7).
 * @param R A reference to a vector of floats containing the observation noise for each measurement (R7).
 */
KalmanFilter2DH::KalmanFilter2DH(const float& dt, const bool& use_dim, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, theta, v, vtheta, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_vel_ = use_vel;
  initialize(Q,R);
}

/**
 * @brief Default destructor.
 * @details Default destructor.
 * 
 */
KalmanFilter2DH::~KalmanFilter2DH() {}

/**
 * @brief Helper function to display the matrix F.
 * @details Helper function to display the matrix F.
 * 
 */
void KalmanFilter2DH::printF() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5), F_(0,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5), F_(1,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5), F_(2,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5), F_(3,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5), F_(4,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5), F_(5,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(6,0), F_(6,1), F_(6,2), F_(6,3), F_(6,4), F_(6,5), F_(6,6));
}

/**
 * @brief Helper function to display the derivative of the F matrix.
 * @details Helper function to display the derivative of the F matrix.
 * 
 */
void KalmanFilter2DH::printdFdX() {
  /*for (unsigned int i=0; i < dFdX_.rows(); i++) {
    printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d", __func__, __LINE__);
    for (unsigned int j=0; j < dFdX_.cols(); i++) {
      printf(" %.3f", dFdX_(i,j));
    }
    printf("\n");
  }*/
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(0,0), dFdX_(0,1), dFdX_(0,2), dFdX_(0,3), dFdX_(0,4), dFdX_(0,5), dFdX_(0,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(1,0), dFdX_(1,1), dFdX_(1,2), dFdX_(1,3), dFdX_(1,4), dFdX_(1,5), dFdX_(1,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(2,0), dFdX_(2,1), dFdX_(2,2), dFdX_(2,3), dFdX_(2,4), dFdX_(2,5), dFdX_(2,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(3,0), dFdX_(3,1), dFdX_(3,2), dFdX_(3,3), dFdX_(3,4), dFdX_(3,5), dFdX_(3,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(4,0), dFdX_(4,1), dFdX_(4,2), dFdX_(4,3), dFdX_(4,4), dFdX_(4,5), dFdX_(4,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(5,0), dFdX_(5,1), dFdX_(5,2), dFdX_(5,3), dFdX_(5,4), dFdX_(5,5), dFdX_(5,6));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, dFdX_(6,0), dFdX_(6,1), dFdX_(6,2), dFdX_(6,3), dFdX_(6,4), dFdX_(6,5), dFdX_(6,6));
}

/**
 * @brief Helper function to display the state X.
 * @details Helper function to display the state X.
 * 
 */
void KalmanFilter2DH::printX() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5), X_(6));
}

/**
 * @brief Instantiates all the variables in the filter. 
 * @details This function instantiates the following variables:
 * X the state (R7), P the covariance (R7x7), Q the process noise (R7x7), R the observation noise (R?x?), F the dynamics (R7x7),
 * H the observation matrix (R?x6), and I an identity matrix (R7x7).
 * F, the dynamics is computed using the following equations:
 * x_t+1 = x_t + v_t * cos(theta_t) * dt
 * y_t+1 = y_t + v_t * sin(theta_t) * dt
 * theta_t+1 = theta_t + vtheta_t * dt
 * vx_t+1 = vx_t
 * vtheta_t+1 = vtheta_t
 * h_t+1 = h_t
 * w_t+1 = w_t
 * 
 * @param Q The reference to the process noise vector (R7).
 * @param R The reference to the measurement noise vector (R7).
 */
void KalmanFilter2DH::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [u, v, theta, v, vtheta, h, w]
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Creating new 2D Kalman filter.\n", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(7);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting state to 0.\n", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0, 0; // State is initialized a 0.
  dFdX_ = Eigen::MatrixXf::Identity(7,7); // Initializes the derivative matrix with ones on the diagonal.

  // Process noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting Q.\n", __func__, __LINE__); 
#endif
  Q_ = Eigen::MatrixXf::Zero(7,7);  
  Q_(0,0) = Q[0];
  Q_(1,1) = Q[1];
  Q_(2,2) = Q[2];
  Q_(3,3) = Q[3];
  Q_(4,4) = Q[4];
  Q_(5,5) = Q[5];
  Q_(6,6) = Q[6];

  // Dynamics
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting F.\n", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(7,7);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting P.\n", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(7,7);
  P_ = Q_*Q_; // P is initialialized at Q*Q

  // Helper matrices
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting I.\n", __func__, __LINE__); 
#endif
  I_ = Eigen::MatrixXf::Identity(7,7);

  // Observation noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting R.\n", __func__, __LINE__); 
#endif
  buildR(R); // R is modular and depends on user choices. (use_dim, use_vel)
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting H.\n", __func__, __LINE__); 
#endif
  buildH(); // H is modular and depends on user choices. (use_dim, use_vel)
}

/**
 * @brief Instantiates the Observation matrix H.
 * @details This function builds the observation matrix H based on the selected parameters.
 * If the user selected use_vel, then the velocities will be observed and hence included in the observation matrix.
 * If the user selected use_dim, then the height and width will be observed and hence included in the observation matrix.
 * 
 */
void KalmanFilter2DH::buildH(){
  int h_size = 3;
  if (use_vel_) {
    h_size += 2;
  }
  if (use_dim_) {
    h_size += 2;
  }
  H_ = Eigen::MatrixXf(h_size, 7);
  Eigen::MatrixXf H_pos, H_pos_z, H_vel, H_vel_z, H_hw;
  H_pos = Eigen::MatrixXf::Zero(3,7);
  H_vel = Eigen::MatrixXf::Zero(2,7);
  H_hw = Eigen::MatrixXf::Zero(2,7);

  H_pos << 1.0, 0, 0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0, 0, 0,
            0, 0, 1.0, 0, 0, 0, 0;

  H_vel << 0, 0, 0, 1.0, 0, 0, 0,
           0, 0, 0, 0, 1.0, 0, 0;

  H_hw << 0, 0, 0, 0, 0, 1.0, 0,
           0, 0, 0, 0, 0, 0, 1.0;

  H_ = Eigen::MatrixXf::Zero(h_size,7);
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

/**
 * @brief Instantiates the measurement noise R. 
 * @details The function builds the measurement noise matrix R.
 * The size of this matrix changes based on the selected parameters parameters.
 * If the user selected use_vel, then the noise on the observed velocities will be added to the matrix.
 * If the user selected use_dim, then the noise on the observed height and width will be added to the matrix.
 * 
 * @param R The reference to the vector containing the noise of the measurement (R6).
 */
void KalmanFilter2DH::buildR(const std::vector<float>& R){
  int r_size = 3;
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
  R_(2,2) = R[2];
  if (use_vel_) {
    if (use_dim_) {
      R_(3,3) = R[3];
      R_(4,4) = R[4];
      R_(5,5) = R[5];
      R_(6,6) = R[6]; 
    } else {
      R_(3,3) = R[3];
      R_(4,4) = R[4];
    }
  } else {
    if (use_dim_) {
      R_(3,3) = R[5];
      R_(4,4) = R[6];
    }
  }
}

/**
 * @brief Converts the std::vector to a eigen::MatrixXf. 
 * @details Converts the measurement (std::vector<float>) into a eigen::MatrixXf.
 * It also ensures that the correct quantities are being observed.
 * The user must give a vector of full size even if the values are not observed (use 0s instead).
 * 
 * @param measurement The reference to the measurement vector (R7).
 */
void KalmanFilter2DH::getMeasurement(const std::vector<float>& measurement) {
  // State is [u, v, theta, v, vtheta, h, w]
  Z_(0) = measurement[0];
  Z_(1) = measurement[1];
  Z_(2) = measurement[2];
  if (use_vel_) {
    if (use_dim_) {
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
      Z_(5) = measurement[5];
      Z_(6) = measurement[6];
    } else {
      Z_(3) = measurement[3];
      Z_(4) = measurement[4];
    }
  } else {
    if (use_dim_) {
      Z_(3) = measurement[5];
      Z_(4) = measurement[6];
    }
  }
}

/**
 * @brief Updates the dynamics based on dt.
 * @details Updates the value of the dynamics such that the propagation of the velocity is correct.
 * We recall that:
 * x_t+1 = x_t + v_t * cos(theta_t) * dt
 * y_t+1 = y_t + v_t * sin(theta_t) * dt
 * theta_t+1 = theta_t + vtheta * dt
 * With (x_t, y_t) the position at time t, v_t and vtheta the linear and angular velocity at time t, and theta_t the angle at time t.
 * @param dt The delta of time between two updates.
 */
void KalmanFilter2DH::updateF(const float& dt) {
  F_(0,3) = dt*std::cos(X_[2]);
  F_(1,3) = dt*std::sin(X_[2]);
  F_(2,4) = dt;
}

/**
 * @brief Updates the dynamics.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * We recall that:
 * x_t+1 = x_t + v_t * cos(theta_t) * dt
 * y_t+1 = y_t + v_t * sin(theta_t) * dt
 * theta_t+1 = theta_t + vtheta * dt
 * With (x_t, y_t) the position at time t, v_t and vtheta the linear and angular velocity at time t, and theta_t the angle at time t.
 * 
 */
void KalmanFilter2DH::updateF() {
  F_(0,3) = dt_*std::cos(X_[2]);
  F_(1,3) = dt_*std::sin(X_[2]);
  F_(2,4) = dt_;
}

/**
 * @brief Updates the derivative of the dynamics.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * We recall that:
 * x_t+1 = x_t + v_t * cos(theta_t) * dt
 * y_t+1 = y_t + v_t * sin(theta_t) * dt
 * theta_t+1 = theta_t + vtheta * dt
 * With (x_t, y_t) the position at time t, v_t and vtheta the linear and angular velocity at time t, and theta_t the angle at time t.
 * Hence, the jacobian is as follows:
 * dx / dtheta  = -sin(theta) * v * dt
 * dx / v = cos(theta) * dt
 * dy / dtheta = cos(theta) * v * dt
 * dy / v = sin(theta) * dt
 * dtheta / dvtheta = dt
 * 
 */
void KalmanFilter2DH::updatedFdX() {
  dFdX_(0,2) = -std::sin(X_[2]) *  X_[3] * dt_;
  dFdX_(0,4) = std::cos(X_[2]) * dt_;
  dFdX_(1,2) = std::cos(X_[2]) *  X_[3] * dt_;
  dFdX_(1,4) = std::sin(X_[2]) * dt_;
  dFdX_(3,4) = dt_;
}

/**
 * @brief Updates the derivative of the dynamics.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * We recall that:
 * x_t+1 = x_t + v_t * cos(theta_t) * dt
 * y_t+1 = y_t + v_t * sin(theta_t) * dt
 * theta_t+1 = theta_t + vtheta * dt
 * With (x_t, y_t) the position at time t, v_t and vtheta the linear and angular velocity at time t, and theta_t the angle at time t.
 * Hence, the jacobian is as follows:
 * dx / dtheta  = -sin(theta) * v * dt
 * dx / v = cos(theta) * dt
 * dy / dtheta = cos(theta) * v * dt
 * dy / v = sin(theta) * dt
 * dtheta / dvtheta = dt
 * 
 * @param dt The delta of time between two updates.
 * 
 */
void KalmanFilter2DH::updatedFdX(const float& dt) {
  dFdX_(0,2) = -std::sin(X_[2]) *  X_[3] * dt;
  dFdX_(0,4) = std::cos(X_[2]) * dt;
  dFdX_(1,2) = std::cos(X_[2]) *  X_[3] * dt;
  dFdX_(1,4) = std::sin(X_[2]) * dt;
  dFdX_(3,4) = dt;
}


/**
 * @brief Default constructor.
 * @details Default constructor.
 * 
 */
KalmanFilter3D::KalmanFilter3D() {}

/**
 * @brief Prefered constructor.
 * @details Prefered constructor.
 * 
 * @param dt The default time-delta in between two updates.
 * @param use_dim A flag to indicate if the filter should observe the height and width of the tracked object.
 * @param use_vel A flag to indicate if the filter should observe the velocity of the tracked object.
 * @param Q A reference to a vector of floats containing the process noise for each variable (R8).
 * @param R A reference to a vector of floats containing the observation noise for each measurement (R8).
 */
KalmanFilter3D::KalmanFilter3D(const float& dt, const bool& use_dim, const bool& use_vel, const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx, vy, vz, h, w]
  dt_ = dt;
  use_dim_ = use_dim;
  use_vel_ = use_vel_;
  initialize(Q,R);
}

/**
 * @brief Default destructor.
 * @details Default destructor.
 * 
 */
KalmanFilter3D::~KalmanFilter3D() {}

/**
 * @brief Helper function to display the matrix F.
 * @details Helper function to display the matrix F.
 * 
 */
void KalmanFilter3D::printF() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(0,0), F_(0,1), F_(0,2), F_(0,3), F_(0,4), F_(0,5), F_(0,6), F_(0,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(1,0), F_(1,1), F_(1,2), F_(1,3), F_(1,4), F_(1,5), F_(1,6), F_(1,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(2,0), F_(2,1), F_(2,2), F_(2,3), F_(2,4), F_(2,5), F_(2,6), F_(2,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(3,0), F_(3,1), F_(3,2), F_(3,3), F_(3,4), F_(3,5), F_(3,6), F_(3,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(4,0), F_(4,1), F_(4,2), F_(4,3), F_(4,4), F_(4,5), F_(4,6), F_(4,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(5,0), F_(5,1), F_(5,2), F_(5,3), F_(5,4), F_(5,5), F_(5,6), F_(5,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(6,0), F_(6,1), F_(6,2), F_(6,3), F_(6,4), F_(6,5), F_(6,6), F_(6,7));
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, F_(7,0), F_(7,1), F_(7,2), F_(7,3), F_(7,4), F_(7,5), F_(7,6), F_(7,7));
}

/**
 * @brief Helper function to display the matrix X.
 * @details Helper function to display the matrix X.
 * 
 */
void KalmanFilter3D::printX() {
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", __func__, __LINE__, X_(0), X_(1), X_(2), X_(3), X_(4), X_(5), X_(6), X_(7));
}

/**
 * @brief Instantiates all the variables in the filter. 
 * @details This function instantiates the following variables:
 * X the state (R8), P the covariance (R8x8), Q the process noise (R8x8), R the observation noise (R?x?), F the dynamics (R8x8),
 * H the observation matrix (R?x8), and I an identity matrix (R8x8).
 * F, the dynamics is computed using the following equations:
 * x_t+1 = x_t + vx_t * dt
 * y_t+1 = y_t + vy_t * dt
 * z_t+1 = z_t + vz_t * dt
 * vx_t+1 = vx_t
 * vy_t+1 = vy_t
 * vz_t+1 = vz_t
 * h_t+1 = h_t
 * w_t+1 = w_t
 * 
 * @param Q The reference to the process noise vector (R8).
 * @param R The reference to the measurement noise vector (R8).
 */
void KalmanFilter3D::initialize(const std::vector<float>& Q, const std::vector<float>& R) {
  // State is [x, y, z, vx ,vy, vz, h, w]
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Creating new 3D Kalman filter.\n", __func__, __LINE__); 
#endif
  X_ = Eigen::VectorXf(8);
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting state to 0.\n", __func__, __LINE__); 
#endif
  X_ << 0, 0, 0, 0, 0, 0, 0, 0; // state is initialized a 0

  // Process noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting Q.\n", __func__, __LINE__); 
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
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting F.\n", __func__, __LINE__); 
#endif
  F_ = Eigen::MatrixXf::Identity(8,8);
  updateF(dt_);

  // Covariance
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting P.\n", __func__, __LINE__); 
#endif
  P_ = Eigen::MatrixXf::Zero(8,8);
  P_ = Q_*Q_; // P is initialialized at Q*Q

  // Helper matrices
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting I.\n", __func__, __LINE__); 
#endif
  I_ = Eigen::MatrixXf::Identity(8,8);

  // Observation noise
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting R.\n", __func__, __LINE__); 
#endif
  buildR(R); // R is modular and depends on user choices. (use_dim, use_vel)
#ifdef DEBUG_KALMAN
  printf("\e[1;33m[DEBUG  ]\e[0m KalmanFilter::%s::l%d Setting H.\n", __func__, __LINE__); 
#endif
  buildH(); // H is modular and depends on user choices. (use_dim, use_vel)
}

/**
 * @brief Instantiates the Observation matrix H.
 * @details This function builds the observation matrix H based on the selected parameters.
 * If the user selected use_vel, then the velocities will be observed and hence included in the observation matrix.
 * If the user selected use_dim, then the height and width will be observed and hence included in the observation matrix.
 * 
 */
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

/**
 * @brief Instantiates the measurement noise R. 
 * @details The function builds the measurement noise matrix R.
 * The size of this matrix changes based on the selected parameters parameters.
 * If the user selected use_vel, then the noise on the observed velocities will be added to the matrix.
 * If the user selected use_dim, then the noise on the observed height and width will be added to the matrix.
 * 
 * @param R The reference to the vector containing the noise of the measurement (R8).
 */
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

/**
 * @brief Converts the std::vector to a eigen::MatrixXf. 
 * @details Converts the measurement (std::vector<float>) into a eigen::MatrixXf.
 * It also ensures that the correct quantities are being observed.
 * The user must give a vector of full size even if the values are not observed (use 0s instead).
 * 
 * @param measurement The reference to the measurement vector (R8).
 */
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

/**
 * @brief Updates the dynamics based on dt.
 * @details Updates the dt value in the dynamics such that the propagation of the velocity is correct.
 * We recall that x_t+1 = x_t + vx_t * dt, x_t being the position at time t, and vx_t being the velocity at time t.
 * 
 * @param dt 
 */
void KalmanFilter3D::updateF(const float& dt) {
  F_(0,3) = dt;
  F_(1,4) = dt;
  F_(2,5) = dt;
}