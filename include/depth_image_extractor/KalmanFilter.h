/**
 * @file KalmanFilter.h
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief Header of the Kalman filter classes.
 * @details This file implements a set of kalman filter for object tracking.
 */

#ifndef KalmanFilter_H
#define KalmanFilter_H

#include <vector>
#include <stdio.h>

// EIGEN
#include <eigen3/Eigen/Dense>
#include <math.h>

/**
 * @brief A basic Kalman filter object
 * @details This object implements the default prediction and correction function of a linear Kalman filter.
 * It also provides a set of helper function to debug or access its variables.
 */
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

/**
 * @brief A 2D (x,y) linear Kalman filter.
 * @details This class implements a filter to estimate the motion of objects in two dimensions.
 * The state is composed of 6 variables: x, y, vx, vy, w, h.
 * x is the position on the x axis, y is the position on the y axis, vx and vy, are the velocities on the x and y axis respectively, w is the object width, and h is the object height.
 * The measurement of this filter can be adjusted by the user. The position is always observed, but the velocity and the width/height can, or not, be observed.
 * By default we recommend observing the width/height, and not the velocity. The filter will derive the velocity on its own after its first correction.
 */
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

/**
 * @brief a 3D (x,y,z) linear Kalman filter.
 * @details This class implements a filter to estimate the motion of objects in three dimensions.
 * The state is composed of 8 variables: x, y, z, vx, vy, vz, w, h.
 * x, y, and z are the position on the x, y, and z axis respectively, vx, vy and vz are the velocities on the x, y and z axis respectively, w is the object width, and h is the object height.
 * The measurement of this filter can be adjusted by the user. The position is always observed, but the velocity and the width/height can, or not, be observed.
 * By default we recommend observing the width/height, and not the velocity. The filter will derive the velocity on its own after its first correction.
 */
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