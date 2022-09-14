#ifndef KalmanFilter_H
#define KalmanFilter_H

#include <vector>
#include <stdio.h>

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

class BaseKalmanFilter {
  protected:
    float dt_;
    bool use_dim_;
    bool use_vel_;

    Eigen::VectorXf X_;
    Eigen::VectorXf Z_;
    Eigen::MatrixXf P_;
    Eigen::MatrixXf F_;
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;
    Eigen::MatrixXf I_;
    Eigen::MatrixXf H_;

    virtual void initialize(const std::vector<float>&, const std::vector<float>&);
    virtual void getMeasurement(const std::vector<float>&);
    virtual void printF();
    virtual void printX();

  public:
    BaseKalmanFilter();
    explicit BaseKalmanFilter(const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    ~BaseKalmanFilter();

    virtual void resetFilter(const std::vector<float>&);
    virtual void updateF(const float&);
    virtual void buildR(const std::vector<float>&);
    virtual void buildH();
    void predict();
    void predict(const float&);
    void correct(const std::vector<float>&);
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
};

class KalmanFilter2D : public BaseKalmanFilter {
  protected:
    void initialize(const std::vector<float>&, const std::vector<float>&);
    void getMeasurement(const std::vector<float>&);
    void printF() override;
    void printX() override;

  public:
    KalmanFilter2D();
    KalmanFilter2D(const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    ~KalmanFilter2D();
    virtual void resetFilter(const std::vector<float>&) override;
    void updateF(const float&) override;
    void buildR(const std::vector<float>&) override;
    void buildH() override;
};

class KalmanFilter3D : public BaseKalmanFilter {
  protected:
    void initialize(const std::vector<float>&, const std::vector<float>&);
    void getMeasurement(const std::vector<float>&);
    void printF() override;
    void printX() override;

  public:
    KalmanFilter3D();
    KalmanFilter3D(const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void resetFilter(const std::vector<float>&) override;
    void updateF(const float&) override;
    void buildR(const std::vector<float>&) override;
    void buildH() override;
};

#endif