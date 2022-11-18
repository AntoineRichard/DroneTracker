#include <detect_and_track/utils.h>

csvWriter::csvWriter(std::string&  filename, std::string& separator, std::string& endline, std::vector<std::string>& header, unsigned int& max_buffer_size) : ofs_() {
    ofs_.exceptions(std::ios::failbit | std::ios::badbit);
    separator_ = separator;
    endline_ = endline;
    filename_ = filename;
    createFile();
    makeHeader(header);
}

csvWriter::~csvWriter() {
    flush();
}

void csvWriter::makeHeader(std::vector<std::string> header) {
    openFile();
    for (unsigned int row=0; row < header.size(); row ++) {
        ofs_ << header[row] << separator_;
    }
    ofs_ << endline_;
    closeFile();
}

void csvWriter::createFile() {
    ofs_.open(filename_, std::ofstream::out | std::ofstream::trunc); 
    closeFile();
}

void csvWriter::writeToFile() {
    for (unsigned int line=0; line < buffer_.size(); line ++) {
        for (unsigned int row=0; row < buffer_[line].size(); row ++) {
            ofs_ << std::to_string(buffer_[line][row]) << separator_;
        }
        ofs_ << endline_;
    }
    buffer_.clear();
}

void csvWriter::openFile() {
    ofs_.open(filename_, std::ofstream::out | std::ofstream::app);
}

void csvWriter::closeFile() {
    ofs_.close();
}

void csvWriter::flush() {
    openFile();
    writeToFile();
    closeFile();
}

void csvWriter::addToBuffer(std::vector<float> data) {
    buffer_.push_back(data);
    if (buffer_.size() >= max_buffer_size_) {
        flush();
    }
}

/*RotatedBounding::RotatedBoundingBox(float* data, int& class_id) {
    class_id_ = class_id;
    confidence_ = data[6]; 
    x_ = data[0];
    y_ = data[1];
    w_ = data[2];
    h_ = data[3];
    cos_ = data[4]
    sin_ = data[5]
    theta_ = std::atan(sin_/cos_);
    area_ = data[2] * data[3];

    float bx0, by0, bx1, bx2;
    bx0 = - w_ / 2;
    by0 = - h_ / 2;
    bx1 = w_ / 2;
    by1 = h_ / 2;

    float x1l, y1l, x2l, y2l, x3l, y3l, x4l, y4l;
    x1l = bx0 * cos_ - by0 * sin_; 
    y1l = bx0 * sin_ + by0 * cos_; 
    x2l = bx1 * cos_ - by0 * sin_; 
    y2l = bx1 * sin_ + by0 * cos_; 
    x3l = bx1 * cos_ - by1 * sin_; 
    y3l = bx1 * sin_ + by1 * cos_; 
    x4l = bx0 * cos_ - by1 * sin_; 
    y4l = bx0 * sin_ + by1 * cos_;

    x1_ = x_ + x1l;
    y1_ = y_ + y1l;
    x2_ = x_ + x2l;
    y2_ = y_ + y2l;
    x3_ = x_ + x3l;
    y3_ = y_ + y3l;
    x4_ = x_ + x4l;
    y4_ = y_ + y4l;

    x_max_ = std::max(x2, x3);
    x_min_ = std::min(x1,x4);
    y_max_ = std::max(y3,y4);
    y_min_ = std::max(y1,y2);
    }

static bool sortComparisonFunction(const RotatedBoundingBox& bbox_0, const RotatedBoundingBox& bbox_1) {
    return bbox_0.confidence_ > bbox_1.confidence_;
}

float calculateIOU(const RotatedBoundingBox& bbox) {
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

    const float cos = std::cos(-theta);
    const float sin = std::sin(-theta);
    const float dx = bbox.x_ - x_;
    const float dy = bbox.y_ - y_;

    // Rotate the bbox by theta
    x1l = bbox.x1l_ * cos - bbox.y1l_ * sin + dx;
    y1l = bbox.x1l_ * sin + bbox.y1l_ * cos + dy;
    x2l = bbox.x2l_ * cos - bbox.y2l_ * sin + dx;
    y2l = bbox.x2l_ * sin + bbox.y2l_ * cos + dy;
    x3l = bbox.x3l_ * cos - bbox.y3l_ * sin + dx;
    y3l = bbox.x3l_ * sin + bbox.y3l_ * cos + dy;
    x4l = bbox.x4l_ * cos - bbox.y4l_ * sin + dx;
    y4l = bbox.x4l_ * sin + bbox.y4l_ * cos + dy;

    return w_new * h_new / (area_ + bbox.area_ - w_new * h_new);
} 

void compareWith(RotatedBoundingBox& bbox, const float thred_IOU) {
    if (bbox.valid_ == false || class_id_ != bbox.class_id_) {
        return;
    }

    if (calculateIOU(bbox) >= thred_IOU) {
        bbox.valid_ = false;
    }
}
*/
BoundingBox::BoundingBox(float* data, int& class_id) {
    class_id_ =  class_id,
    confidence_ = data[4];
    x_ = data[0];
    y_ = data[1];
    w_ = data[2];
    h_ = data[3];
    x_min_ = data[0] - data[2]/2;
    x_max_ = data[0] + data[2]/2;
    y_min_ = data[1] - data[3]/2;
    y_max_ = data[1] + data[3]/2;
    area_ = data[2] * data[3];
}

BoundingBox::BoundingBox(const float& xmin, const float& ymin, const float& width, const float& height, const float& conf, const int& class_id) {
    class_id_ =  class_id,
    confidence_ = conf;
    x_ = xmin + width/2;
    y_ = ymin + height/2;
    w_ = width;
    h_ = height;
    x_min_ = xmin;
    x_max_ = xmin+width;
    y_min_ = ymin;
    y_max_ = ymin+height;
    area_ = height*width;
}


float BoundingBox::calculateIOU(const BoundingBox& bbox) {
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

void BoundingBox::compareWith(BoundingBox& bbox, const float thred_IOU) {
    if (bbox.valid_ == false || class_id_ != bbox.class_id_) {
        return;
    }

    if (calculateIOU(bbox) >= thred_IOU) {
        bbox.valid_ = false;
    }
}