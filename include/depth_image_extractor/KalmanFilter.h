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
    bool use_vel_;

    Eigen::VectorXf X_;
    Eigen::VectorXf Z_;
    Eigen::MatrixXf P_;
    Eigen::MatrixXf F_;
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;
    Eigen::MatrixXf I8_;
    Eigen::MatrixXf H_;

    void initialize(const std::vector<float>&, const std::vector<float>&);
    void getMeasurement(const std::vector<float>&);

  public:
    KalmanFilter();
    KalmanFilter(const float&, const bool&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void resetFilter(const std::vector<float>&);
    void updateF(const float&);
    void buildR(const std::vector<float>&);
    void buildH();
    void predict();
    void predict(const float&);
    void correct(const std::vector<float>&);
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
};

#endif