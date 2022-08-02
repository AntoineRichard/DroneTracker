#ifndef KalmanFilter_H
#define KalmanFilter_H

#include <vector>

// EIGEN
#include <eigen3/Eigen/Dense>
#include <math.h>

class KalmanFilter {
  private:
    float dt_;
    bool use_dim_;
    bool use_z_;

    Eigen::VectorXf X_;
    Eigen::VectorXf Z_;
    Eigen::MatrixXf P_;
    Eigen::MatrixXf F_;
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;
    Eigen::MatrixXf I8_;
    Eigen::MatrixXf H_;

    void initialize();
    void getMeasurement(const std::vector<float>&);

  public:
    KalmanFilter();
    KalmanFilter(float, bool, bool);
    void resetFilter(const std::vector<float>&);
    void update();
    void predict();
    void correct(const std::vector<float>&);
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
};

#endif