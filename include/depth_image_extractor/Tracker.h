#ifndef TRACKER_H
#define TRACKER_H

#include <string>
#include <vector>
#include <map>

#include <depth_image_extractor/KalmanFilter.h>
#include <depth_image_extractor/Hungarian.h>
#include <stdio.h>

class Object {
  protected:
    unsigned int nb_skipped_frames_;
    unsigned int nb_frames_;
    unsigned int id_;
    BaseKalmanFilter* KF_;
  public:
    Object();
    Object(const unsigned int&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    ~Object();
    virtual void setState(const std::vector<float>&);
    virtual void newFrame();
    virtual void predict();
    virtual void predict(const float&);
    virtual void correct(const std::vector<float>&);
    virtual void getState(std::vector<float>&);
    virtual void getUncertainty(std::vector<float>&);
    virtual int getSkippedFrames();
};

class Object2D : public Object {
  public:
    Object2D();
    Object2D(const Object2D &);
    Object2D(const unsigned int&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

class Object3D : public Object {
  public:
    Object3D();
    Object3D(const unsigned int&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

class BaseTracker {
  private:
    void incrementFrame();
    bool isMatch(const std::vector<float>&, const std::vector<float>&) const;
    void computeCost(std::vector<std::vector<double>>&, const std::vector<std::vector<float>>&, std::map<int, int>&);
    void hungarianMatching(std::vector<std::vector<double>>&, std::vector<int>&);
  protected:
    // Tracker state
    unsigned int track_id_count_;
    std::map<unsigned int, Object*> Objects_;

    // Tracker parameters
    unsigned int max_frames_to_skip_;
    float distance_threshold_;
    float center_threshold_;
    float area_threshold_; 
    float body_ratio_;
    
    // Kalman parameters
    std::vector<float> R_;
    std::vector<float> Q_;
    float dt_;
    bool use_dim_;
    bool use_vel_;

    // Solver
    HungarianAlgorithm* HA_;

    virtual float centroidsError(const std::vector<float>&, const std::vector<float>&) const;
    virtual float areaRatio(const std::vector<float>&, const std::vector<float>&) const;
    virtual void addNewObject();
  public:
    BaseTracker();
    BaseTracker(const int&, const float&, const float&, const float&, const float&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void update(const float&, const std::vector<std::vector<float>>&);
    void getStates(std::map<int, std::vector<float>>&);
};

class Tracker2D : public BaseTracker {
  protected:
    // Tracker state
    //std::map<unsigned int, Object2D*> Objects_;

    float centroidsError(const std::vector<float>&, const std::vector<float>&) const;
    float areaRatio(const std::vector<float>&, const std::vector<float>&) const;
    void addNewObject() override;
  public:
    Tracker2D();
    Tracker2D(const int&, const float&, const float&, const float&, const float&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

class Tracker3D : public BaseTracker {
  protected:
    // Tracker state
    //std::map<unsigned int, Object3D*> Objects_;

    float centroidsError(const std::vector<float>&, const std::vector<float>&) const;
    float areaRatio(const std::vector<float>&, const std::vector<float>&) const;
    void addNewObject() override;
  public:
    Tracker3D();
    Tracker3D(const int&, const float&, const float&, const float&, const float&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

#endif