#ifndef TRACKER_H
#define TRACKER_H

#include <string>
#include <vector>
#include <map>

#include <depth_image_extractor/KalmanFilter.h>
#include <depth_image_extractor/Hungarian.h>

#include <ros/ros.h>

class Object {
  private:
    KalmanFilter* KF_;
    unsigned int nb_skipped_frames_;
    unsigned int nb_frames_;
    unsigned int id_;
  public:
    Object();
    Object(const unsigned int&, const float&, const bool&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void setState(const std::vector<float>&);
    void newFrame();
    void predict();
    void predict(const float&);
    void correct(const std::vector<float>&);
    void getState(std::vector<float>&);
    void getUncertainty(std::vector<float>&);
    int getSkippedFrames();
};

class Tracker {
  private:
    // Tracker state
    std::map<unsigned int, Object*> Objects_;
    unsigned int track_id_count_;

    // Solver
    HungarianAlgorithm* HA_;

    // Tracker parameters
    unsigned int max_frames_to_skip_;
    float dist_threshold_;
    float center_threshold_;
    float area_threshold_; 
    float body_ratio_;

    // Kalman parameters
    std::vector<float> R_;
    std::vector<float> Q_;
    float dt_;
    bool use_dim_;
    bool use_z_;
    bool use_vel_;

    void incrementFrame();
    float centroidsError(const std::vector<float>&, const std::vector<float>&) const;
    float distanceError(const std::vector<float>&, const std::vector<float>&) const;
    float areaRatio(const std::vector<float>&, const std::vector<float>&) const;
    float bodyShapeError(const std::vector<float>&, const std::vector<float>&) const;
    bool isMatch(const std::vector<float>&, const std::vector<float>&) const;
    void computeCost(std::vector<std::vector<double>>&, const std::vector<std::vector<float>>&, std::map<int, int>&);
    void hungarianMatching(std::vector<std::vector<double>>&, std::vector<int>&);
  public:
    Tracker();
    Tracker(const int&, const float&, const float&, const float&, const float&, const float&, const bool&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void update(const float&, const std::vector<std::vector<float>>&);
    void getStates(std::map<int, std::vector<float>>&);
};

#endif