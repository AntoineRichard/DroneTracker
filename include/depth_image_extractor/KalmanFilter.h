#ifndef KalmanFilter_H
#define KalmanFilter_H

#include <vector>

// EIGEN
#include <eigen3/Eigen/Dense>
#include <math.h>

#define I_PX 0
#define I_PY 1
#define I_PZ 2
#define I_VX 3
#define I_VY 4
#define I_VZ 5
#define I_H  6
#define I_W  7
#define I_CX 8
#define I_CY 9

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