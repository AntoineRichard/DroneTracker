#include <detect_and_track/utils.h>

/**
 * @brief Construct a new csv Writer::csv Writer object
 * 
 * @param filename 
 * @param separator 
 * @param endline 
 * @param header 
 * @param max_buffer_size 
 */
csvWriter::csvWriter(std::string&  filename, std::string& separator, std::string& endline, std::vector<std::string>& header, unsigned int& max_buffer_size) : ofs_() {
    ofs_.exceptions(std::ios::failbit | std::ios::badbit);
    separator_ = separator;
    endline_ = endline;
    filename_ = filename;
    createFile();
    makeHeader(header);
}

/**
 * @brief Construct a new csv Writer::csv Writer object
 * 
 */
csvWriter::csvWriter() : ofs_() {
}

/**
 * @brief Destroy the csv Writer::csv Writer object
 * 
 */
csvWriter::~csvWriter() {
    flush();
}

/**
 * @brief 
 * 
 * @param header 
 */
void csvWriter::makeHeader(std::vector<std::string> header) {
    openFile();
    for (unsigned int row=0; row < header.size(); row ++) {
        ofs_ << header[row] << separator_;
    }
    ofs_ << endline_;
    closeFile();
}

/**
 * @brief 
 * 
 */
void csvWriter::createFile() {
    ofs_.open(filename_, std::ofstream::out | std::ofstream::trunc); 
    closeFile();
}

/**
 * @brief 
 * 
 */
void csvWriter::writeToFile() {
    for (unsigned int line=0; line < buffer_.size(); line ++) {
        for (unsigned int row=0; row < buffer_[line].size(); row ++) {
            ofs_ << std::to_string(buffer_[line][row]) << separator_;
        }
        ofs_ << endline_;
    }
    buffer_.clear();
}

/**
 * @brief 
 * 
 */
void csvWriter::openFile() {
    ofs_.open(filename_, std::ofstream::out | std::ofstream::app);
}

/**
 * @brief 
 * 
 */
void csvWriter::closeFile() {
    ofs_.close();
}

/**
 * @brief 
 * 
 */
void csvWriter::flush() {
    openFile();
    writeToFile();
    closeFile();
}

/**
 * @brief 
 * 
 * @param data 
 */
void csvWriter::addToBuffer(std::vector<float> data) {
    buffer_.push_back(data);
    if (buffer_.size() >= max_buffer_size_) {
        flush();
    }
}

/**
 * @brief Construct a new Bounding Box:: Bounding Box object
 * 
 */
BoundingBox::BoundingBox() {}

/**
 * @brief Construct a new Bounding Box:: Bounding Box object
 * 
 * @param data 
 * @param class_id 
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

/**
 * @brief Construct a new Bounding Box:: Bounding Box object
 * 
 * @param xmin 
 * @param ymin 
 * @param width 
 * @param height 
 * @param conf 
 * @param class_id 
 */
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

/**
 * @brief 
 * 
 * @param bbox 
 * @return float 
 */
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

/**
 * @brief 
 * 
 * @param bbox 
 * @param thred_IOU 
 */
void BoundingBox::compareWith(BoundingBox& bbox, const float thred_IOU) {
    if (bbox.valid_ == false || class_id_ != bbox.class_id_) {
        return;
    }

    if (calculateIOU(bbox) >= thred_IOU) {
        bbox.valid_ = false;
    }
}

/**
 * @brief 
 * 
 * @param state 
 */
void BoundingBox::cast2state(std::vector<float>& state) {
    state.resize(6);
    state[0] = x_;
    state[1] = y_;
    state[2] = 0;
    state[3] = 0;
    state[4] = w_;
    state[5] = h_;
}

/**
 * @brief Construct a new Bounding Box 3 D:: Bounding Box 3 D object
 * 
 */
BoundingBox3D::BoundingBox3D() : BoundingBox::BoundingBox() {}

/**
 * @brief Construct a new Bounding Box 3 D:: Bounding Box 3 D object
 * 
 * @param data 
 * @param class_id 
 */
BoundingBox3D::BoundingBox3D(float* data, int& class_id) {
    class_id_ =  class_id,
    confidence_ = data[6];
    w_ = data[3];
    d_ = data[4];
    h_ = data[5];
    x_min_ = data[0] - data[3]/2;
    x_max_ = data[0] + data[3]/2;
    y_min_ = data[1] - data[4]/2;
    y_max_ = data[1] + data[4]/2;
    z_min_ = data[2] - data[5]/2;
    z_max_ = data[2] + data[5]/2;
    volume_ = data[3] * data[4] * data[5];
}

/**
 * @brief Construct a new Bounding Box 3 D:: Bounding Box 3 D object
 * 
 * @param x 
 * @param y 
 * @param z 
 * @param width 
 * @param depth 
 * @param height 
 * @param conf 
 * @param class_id 
 */
BoundingBox3D::BoundingBox3D(const float& x, const float& y, const float& z, const float& width, const float& depth, const float& height, const float& conf, const int& class_id) {
    class_id_ =  class_id,
    confidence_ = conf;
    w_ = width;
    d_ = depth;
    h_ = height;
    x_ = x;
    y_ = y;
    z_ = z;
    x_min_ = x - width/2;
    x_max_ = x + width/2;
    y_min_ = y - depth/2;
    y_max_ = y + depth/2;
    z_min_ = z - height/2;
    z_max_ = z + height/2;
    volume_ = height*width*depth;
}

/**
 * @brief 
 * 
 * @param bbox 
 * @return float 
 */
float BoundingBox3D::calculateIOU(const BoundingBox3D& bbox) {
    const float x_min_new = std::max(x_min_, bbox.x_min_);
    const float x_max_new = std::min(x_max_, bbox.x_max_);
    const float w_new = x_max_new - x_min_new;
    if (w_new <= 0.0f) {
        return 0.0f;
    }

    const float y_min_new = std::max(y_min_, bbox.y_min_);
    const float y_max_new = std::min(y_max_, bbox.y_max_);
    const float d_new = y_max_new - y_min_new;
    if (d_new <= 0.0f) {
        return 0.0f;
    }
    
    const float z_min_new = std::max(z_min_, bbox.z_min_);
    const float z_max_new = std::min(z_max_, bbox.z_max_);
    const float h_new = z_max_new - z_min_new;
    if (h_new <= 0.0f) {
        return 0.0f;
    }

  return w_new * h_new * d_new/ (volume_ + bbox.volume_ - w_new * h_new * d_new);
} 

/**
 * @brief 
 * 
 * @param state 
 */
void BoundingBox3D::cast2state(std::vector<float>& state) {
    state.resize(9);
    state[0] = x_;
    state[1] = y_;
    state[2] = z_;
    state[3] = 0;
    state[4] = 0;
    state[5] = 0;
    state[6] = w_;
    state[7] = d_;
    state[8] = h_;
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