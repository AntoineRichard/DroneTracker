/**
 * @file Tracker.h
 * @author antoine.richard@uni.lu
 * @version 0.1
 * @date 2022-09-21
 * 
 * @copyright University of Luxembourg | SnT | SpaceR 2022--2022
 * @brief The header of the tracker class.
 * @details This file implements simple algorithms to track objects.
 */

#ifndef TRACKER_H
#define TRACKER_H

#include <string>
#include <vector>
#include <map>
#include <chrono>



#include <detect_and_track/KalmanFilter.h>
#include <detect_and_track/Hungarian.h>
#include <stdio.h>

/**
 * @brief An object to be tracked.
 * @details A class that contains all the information required to track an object.
 * It has a built-in Kalman filter, to predict the position of the object,
 * and a set of method to evaluate if the object matches a given observation.
 * It also keeps track of some object specific information.
 * 
 */
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

/**
 * @brief An object to be tracked.
 * @details A class that contains all the information required to track an object.
 * It has a built-in Kalman filter, to predict the position of the object,
 * and a set of method to evaluate if the object matches a given observation.
 * It also keeps track of some object specific information.
 * 
 */
class Object2D : public Object {
  public:
    Object2D();
    Object2D(const Object2D &);
    Object2D(const unsigned int&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

/**
 * @brief An object to be tracked.
 * @details A class that contains all the information required to track an object.
 * It has a built-in Kalman filter, to predict the position of the object,
 * and a set of method to evaluate if the object matches a given observation.
 * It also keeps track of some object specific information.
 * 
 */
class Object3D : public Object {
  public:
    Object3D();
    Object3D(const unsigned int&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
};

/**
 * @brief An object tracker.
 * @details This class tracks multiple objects.
 * It uses their built-in kalman filter to estimate where the objects are going to be next.
 * Then it compares their estimated position to a set observation to find the best possible match.
 * This matching is done using the Hungarian Algorithm. Once the matching is done, the matches are analyzed and confirmed.
 * The tracked objects which have an associated measurement are then updated.
 * 
 */
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
    void removeOldTracks();
  public:
    BaseTracker();
    BaseTracker(const int&, const float&, const float&, const float&, const float&, const float&, const bool&, const bool&, const std::vector<float>&, const std::vector<float>&);
    void update(const float&, const std::vector<std::vector<float>>&);
    void getStates(std::map<unsigned int, std::vector<float>>&);
};

/**
 * @brief An object tracker.
 * @details This class tracks multiple objects.
 * It uses their built-in kalman filter to estimate where the objects are going to be next.
 * Then it compares their estimated position to a set observation to find the best possible match.
 * This matching is done using the Hungarian Algorithm. Once the matching is done, the matches are analyzed and confirmed.
 * The tracked objects which have an associated measurement are then updated.
 * 
 */
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

/**
 * @brief An object tracker.
 * @details This class tracks multiple objects.
 * It uses their built-in kalman filter to estimate where the objects are going to be next.
 * Then it compares their estimated position to a set observation to find the best possible match.
 * This matching is done using the Hungarian Algorithm. Once the matching is done, the matches are analyzed and confirmed.
 * The tracked objects which have an associated measurement are then updated.
 * 
 */
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