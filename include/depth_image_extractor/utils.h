#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

static const std::vector<cv::Scalar> ColorPalette{
      cv::Scalar(0, 64, 255),
      cv::Scalar(64, 255, 0),
      cv::Scalar(255, 0, 64),
      cv::Scalar(0, 255, 192),
      cv::Scalar(255, 192, 0),
      cv::Scalar(192, 0, 255),
      cv::Scalar(0, 192, 255),
      cv::Scalar(192, 255, 0),
      cv::Scalar(255, 0, 192),
      cv::Scalar(0, 255, 64),
      cv::Scalar(255, 64, 0),
      cv::Scalar(64, 0, 255),
      cv::Scalar(0, 128, 255),
      cv::Scalar(128, 255, 0),
      cv::Scalar(255, 0, 128),
      cv::Scalar(0, 255, 128),
      cv::Scalar(255, 128, 0),
      cv::Scalar(128, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 0, 255)
  };

class csvWriter {
    private:
        std::ofstream ofs_;
        std::string separator_;
        std::string endline_;
        std::string filename_;
        unsigned int max_buffer_size_;
        std::vector<std::vector<float>> buffer_;

        void makeHeader(std::vector<std::string> header) {
            openFile();
            for (unsigned int row=0; row < header.size(); row ++) {
                ofs_ << header[row] << separator_;
            }
            ofs_ << endline_;
            closeFile();
        }

        bool createFile() {
            ofs_.open(filename_, std::ofstream::out | std::ofstream::trunc); 
            closeFile();
        }

        void writeToFile() {
            for (unsigned int line=0; line < buffer_.size(); line ++) {
                for (unsigned int row=0; row < buffer_[line].size(); row ++) {
                    ofs_ << std::to_string(buffer_[line][row]) << separator_;
                }
                ofs_ << endline_;
            }
            buffer_.clear();
        }

        void openFile() {
            ofs_.open(filename_, std::ofstream::out | std::ofstream::app);
        }
        void closeFile() {
            ofs_.close();
        }
    public:
        csvWriter(std::string&  filename, std::string& separator, std::string& endline, std::vector<std::string>& header, unsigned int& max_buffer_size) : ofs_() {
            ofs_.exceptions(std::ios::failbit | std::ios::badbit);
            separator_ = separator;
            endline_ = endline;
            filename_ = filename;
            createFile();
            makeHeader(header);
        }

        ~csvWriter() {
            flush();
        }

        void flush() {
            openFile();
            writeToFile();
            closeFile();
        }

        bool addToBuffer(std::vector<float> data) {
            buffer_.push_back(data);
            if (buffer_.size() >= max_buffer_size_) {
                flush();
            }
        }
};

class BoundingBox {
    public:
        BoundingBox (float* data): 
            class_id_((int) (data[6]>data[5])),
            confidence_(data[4]),
            x_(data[0]),
            y_(data[1]),
            w_(data[2]),
            h_(data[3]),
            x_min_(data[0] - data[2]/2),
            x_max_(data[0] + data[2]/2),
            y_min_(data[1] - data[3]/2),
            y_max_(data[1] + data[3]/2),
            area_(data[2] * data[3])
            {}

        static bool sortComparisonFunction(const BoundingBox& bbox_0, const BoundingBox& bbox_1) {
            return bbox_0.confidence_ > bbox_1.confidence_;
        }

        float calculateIOU (const BoundingBox& bbox) {
            const float x_min_new = std::max(x_min_, bbox.x_min_);
            const float x_max_new = std::min(x_max_, bbox.x_max_);
            const float w_new = x_max_new - x_min_new;
            if (w_new <= 0.0f) {
                return 0.0f;
            }

            const float y_min_new = std::max(y_min_, bbox.y_min_);
            const float y_max_new = std::min(y_max_, bbox.y_max_);
            const float h_new = y_max_new - y_min_new;
            if (h_new <= 0.0f) {
                return 0.0f;
            }

            return w_new * h_new / (area_ + bbox.area_ - w_new * h_new);
        } 

        void compareWith(BoundingBox& bbox, const float thred_IOU) {
            if (bbox.valid_ == false || class_id_ != bbox.class_id_) {
                return;
            }

            if (calculateIOU(bbox) >= thred_IOU) {
                // ROS_INFO(
                //     "bbox0: tx = %.4f, ty = %.4f, tw = %.4f, th = %.4f", 
                //     x_, y_, w_, h_
                // );
                // ROS_INFO(
                //     "bbox1: tx = %.4f, ty = %.4f, tw = %.4f, th = %.4f", 
                //     bbox.x_, bbox.y_, bbox.w_, bbox.h_
                // );
                // ROS_INFO("IOU = %.4f\n", calculateIOU(bbox));
                bbox.valid_ = false;
            }
        }

        int class_id_;
        float confidence_;
        float x_; // center x
        float y_; // center y
        float w_; // width
        float h_; // height
        float x_min_;
        float x_max_;
        float y_min_;
        float y_max_;
        float area_;
        
        bool valid_ = true;
};
