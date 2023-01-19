# Object detection and tracking
![OS](https://img.shields.io/badge/OS-Ubuntu_20.04-orange.svg) ![ROS2_FOXY](https://img.shields.io/badge/ROS2-Foxy-brightgreen.svg) ![VERSION](https://img.shields.io/badge/Detect&Track-0.9-lightgrey.svg)

This repository enables the detection and tracking of objects on machines equipped with Nvidia GPUs.
The code is written in C++ 17, enabling running networks such as Yolov5 without python3 installed.
We provide means to track the object in the image plane (u,v coordinate tracking), rotated bounding-boxes (u,v,theta), as well as 3D tracking (x,y,z) if an RGB-D camera is used.
More on that later.

# Requirements and installation
To run this code, the following dependencies need to be installed:
- cmake 3.0.2 (or higher)
- Support for C++ 17 or higher
- OpenCV4.X (tested with version 4.2.0)
- CUDA 11.X (tested with version 11.3)
- CuDNN 8.X (tested with version 8.4.1.5 (for cuda 11.6))
- TensorRT 8.X (tested with version 8.4.1.5)

For the compilation to work, the CMakeList must be modified. The include and link directories of TensorRT must be ajusted to match your own installation of TensorRT.

This is what it looks like in our case (l33-34):
```
include_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/include)
link_directories(/home/antoine/Downloads/TensorRT-8.4.1.5/lib)
```

Once all this is done, the code should compile and be ready to use!

# Transforming the models in TensorRT engine
Before we can run our first neural network, we must convert it to a format TensorRT can read!
To do so, we first need to convert our model to an ONNX model, and then convert it to a TensorRT engine.

## Converting to ONNX
As mentioned earlier, the first step consists in converting the model to ONNX.
ONNX is a open format designed to ease the deployement of neural-networks.
If you are using YoLOv5 from ultralytics, you can use the following command:
```
python3 export.py --weights PATH/2/PT_FILE.pt --img 640 --batch 1 --include onnx --simplify --opset 11
```
If you are using another model, we recommend following these tutorials:
- Pytorch: https://pytorch.org/docs/stable/onnx.html
- Tensorflow: https://onnxruntime.ai/docs/tutorials/tf-get-started.html

Once the model is converted, we encourage you to check the generated model using the neutron app:
- https://netron.app/

This application will enable you to visualize your network.
Using it, you can also check the number of classes the network is detecting.
To do so, scroll down till you reach the last layer of the network.
On Yolov5 it looks like that:

TODO Insert image

We can see there that the shape of the final layer is `1x25200x6`. This means that we are detecting a single class. 
If you were detecting 2 classes it would read `1x25200x7`, 3 classes `1x252600x8`, etc...
This comes from the structure of the output layer. For regular, non-rotated bounding-boxes it looks like that:
```
min_x, min_y, width, height, confidence, class_1, class_2,...,class_N
```
This depends on your network and is only intended to help you find the number of classes your network can classify.
If your network detects rotated bounding boxes or polygons this method not work.


## Converting to TensorRT
Now that we have our ONNX model we can turn it into a TensorRT engine, which is required to run this code.
Before building the engine, make sure that you are on the target deployment platform.
This means that if you plan on deploying this model on a Jetson Xavier, you have to build it on the Xavier itself, and not your server or desktop/laptop.
Some of the optimization done through the building process are unique to the target GPU.
To compile the model use the following command:
```
trtexec --onnx=PATH/2/ONNX_MODEL.onnx --workspace=4096 --saveEngine=PATH/2/TENSORRT_MODEL.engine --verbose
```
If it doesn't work, it may be because the path to tensorRT is not exported properly.
To export it you can use the following command (default install):
```
export PATH=$PATH:/usr/src/tensorrt/bin
```
In our case we used:
```
export PATH=$PATH:/home/antoine/Downloads/TensorRT-8.4.1.5/bin
```
The workspace argument is the maximum amount of VRAM allocated to the neural-net (in Mb).
You can change this to suite your deployment need, 4Gb is propably too much for a Jetson Nano, 2Gb seems more reasonable.
To increase the performance, one can also use the `--fp16` flag, which will force the model to run on float16 operations instead of float32.
This should result in higher frames per second, but will also impact the network accuracy.

# How to use this code with ROS
Now that we have our model, we are finally ready to get detecting!
In ROS we provide premade profiles that worked we tested on the detection and tracking of drones and rocks.
We provide four launch-files:
- `detect.launch`, only detects the objects and outputs the bounding boxes as ROS messages.
- `detect_and_locate.launch`, detects and estimates the 3D position of the objects in the camera local frame. They are then outputed as ROS messages.
- `detect_and_track2D.launch`, detects and tracks objects, the tracks and detections are outputed as ROS messages.
- `detect_track_2D_and_locate.launch`, detects, locates and tracks the objects. They are then outputed as ROS messages.

To use any of these files use the usual launch command. However, before you do, we would encourage you to go through the next section, how to configure the different components.

## Editing the config files
In the following we outline the different parameters and what they are used for.

### The object detector
The object detector has the following parameters, they can be changed in `config/object_detection.yaml`:
- `path_to_engine`, `string`, the absolute path to the TensorRT engine.
- `class_map`, `list<string>`, the list of classes the network can detect.
- `num_classes`, `int`, the number of classes the network can detect.
- `buffer_size`, `int`, the number of inputs and outputs of the network.
- `image_width`, `int`, the width of the image to be processed.
- `image_height`, `int`, the height of the image to be processed.
- `nms_threshold`, `float`, the threshold used in non-maximum supression, a filtering step used to remove duplicate detections.
- `conf_threshold`, `float`, the minimum amount of confidence the network must have to consider the detected bounding box as a valid detection.
- `max_output_bbox_count`, `int`, the maximum number of boundingboxes the network can output.

### The pose estimation
The object detector has the following parameters, they can be changed in `config/pose_estimator.yaml`:
- `lens_distortion_model`, `string`, the type of lens distortion model, either `pin_hole` or `plumb_blob`.
- `K`, `std::vector<float>`, the parameters of the plumb blob distortion model. The list in the yaml must have a size of 5 and is organized as follows: `[k1,k2,k3,k4,k5]`.
- `camera_parameters`, `std::vector<float>`, the parameters of the pin-hole model. The list in the yaml must have a size of 4 and is organized as follows: `[fx,fy,cx,cy]`.  
- `position_mode`, `std::string`, the way the 3D position of the object is evaluated. For now we provide two modes, `min_distance`, and `center`. We plan on extending these to `mask`, and `box_approximation` when we'll add support for instance segmentation.
- `rejection_threshold`, `float`, a parameter used in the min_distance mode, it sets the amount of closest points that are considered as outliers. These values must be between [0,1].
- `keep_threshold`, `float`, a parameter used in min_distance mode, the amount of closest points that are being averaged to determine the distance to the detected object.

### The tracker
The tracker has the following parameters, they can be changed in the `config/tracker.yaml`:
- `max_frames_to_skip`, `int`, the maximum number of frames that can be skipped in a row before a trace is deleted.
- `dist_threshold`, `float`, the upperbound distance to consider the association between two possible match. If their distance is higher, then it is considered infinite, i.e. it will prevent them being matched by the Hungarian Algorithm.
- `center_threshold`, `float`, the maximum distance between the center of a matched detection and trace to be considered a real match.
- `area_threshold`, `float`, the maximum ratio of size between the area of a matched detection and trace to be considered a real match.
- `body_ratio`, `float`, the minimum ratio of size between the area of a matched detection and trace to be considered a real match.
- `dt`, `float`, the default dt inbetween two frames. In most cases it will be useless, as the tracker will rely on ROS to get the time between two observation. The option is integrated to make the code as modular as possible and easy to edit.
- `use_dim`, `bool`, specifies if dimmension of the bounding boxes should be used in the observation phase of the tracking (usually set to true).
- `use_vel`, `bool`, specifies if the velocity of the bounding boxes should be used in the observation phase of the tracking (usually set to false).
- `Q`, `list<float>`, the process noise that will be used by the Kalman Filter. The size, or order of the variables depend on the Kalman filter you are using. For 2D tracking the dimmension is 6, (x,y,vx,vy,h,w). Note that the whole of it must be provided regardless of the `use_dim` or `use_vel` flag.
- `R`, `list<float>`, the observation noise that will be used by the Kalman Filter. The size, or order of the variables depend on the Kalman filter you are using. For 2D tracking the dimmension is 6, (x,y,vx,vy,h,w). Note that the whole of it must be provided regardless of the `use_dim` or `use_vel` flag.
- `min_box_width`, `int`, the minimum width of the bounding boxes that can be tracked.
- `max_box_width`, `int`, the maximum width of the bounding boxes that can be tracked.
- `min_box_height`, `int`, the minimum height of the bounding boxes that can be tracked.
- `max_box_height`, `int`, the maximum height of the bounding boxes that can be tracked.

# How to use this code in standalone mode
TODO

# How to modify this code
## Code Structure
The code is articulated around 4 main classes:
- The `ObjectDetector` class, applies a forward-pass of the network on a single image.
- The `PoseEstimation` class, when using 3D data, this class estimates the position of the object in the camera reference frame.
- The `Tracker` class, it tracks a wide variery of objects, 2D, 2D with rotation, and 3D. It comes with two companion classes:
    - The `KalmanFilter` class, a filter that is used to propagate the detections.
    - The `Object class`, a helper class, to store data relevant to the tracking.
- The `Detection` class, it uses the previously introduced classes to detect, locate and track the objects. 
