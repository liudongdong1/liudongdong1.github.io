# MediaPipe_Usage


## 1. Installation

```shell
$ git clone https://github.com/google/mediapipe.git

# Change directory into MediaPipe root directory
$ cd mediapipe

#install Bazel
#link https://blog.csdn.net/liudongdong19

#install opencv and ffmpeg
sudo apt-get install libopencv-core-dev libopencv-highgui-dev \libopencv-calib3d-dev libopencv-features2d-dev \libopencv-imgproc-dev libopencv-video-dev

# Requires a GPU with EGL driver support.
# Can use mesa GPU libraries for desktop, (or Nvidia/AMD equivalent).
sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev

# To compile with GPU support, replace
--define MEDIAPIPE_DISABLE_GPU=1
# with
--copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11
          
$ export GLOG_logtostderr=1

# if you are running on Linux desktop with CPU only
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hello_world:hello_world

# If you are running on Linux desktop with GPU support enabled (via mesa drivers)
$ bazel run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
    mediapipe/examples/desktop/hello_world:hello_world
```

## 2.  Example 

### 2.1. CPU

```shell
#build
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu

#run
GLOG_logtostderr=1 
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file = mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

#multi-hand tracking
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu

#run
GLOG_logtostderr=1 
bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu --calculator_graph_config_file = mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

```

### 2.2. GPU

```shell
#build
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
  mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu

#run
GLOG_logtostderr=1 
bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
  --calculator_graph_config_file = mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
```

## 3. HandRelative

### 3.1.  Mobile 

> Graph: [`mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt)
>
> Android target:  [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu:handtrackinggpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handtrackinggpu/BUILD)
>
> iOS target: [`mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/handtrackinggpu/BUILD)

> Graph: [`mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt)
>
> Android target: [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1Wk6V9EVaz1ks_MInPqqVGvvJD01SGXDc) [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/multihandtrackinggpu:multihandtrackinggpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/multihandtrackinggpu/BUILD)
>
> iOS target: [`mediapipe/examples/ios/multihandtrackinggpu:MultiHandTrackingGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/multihandtrackinggpu/BUILD)

> Graph: [`mediapipe/graphs/hand_tracking/hand_detection_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_detection_mobile.pbtxt)
>
> Android target: [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1qUlTtH7Ydg-wl_H6VVL8vueu2UCTu37E) [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu:handdetectiongpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/handdetectiongpu/BUILD)
>
> iOS target: [`mediapipe/examples/ios/handdetectiongpu:HandDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/handdetectiongpu/BUILD)

### 3.2. Desktop

> Running on CPU
>
> - Graph: [`mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt)
> - Target: [`mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/hand_tracking/BUILD)
>
> Running on GPU
>
> - Graph: [`mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt)
> - Target: [`mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/hand_tracking/BUILD)

> Running on CPU
>
> - Graph: [`mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live)
> - Target: [`mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/multi_hand_tracking/BUILD)
>
> Running on GPU
>
> - Graph: [`mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt)
> - Target: [`mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/multi_hand_tracking/BUILD)

## 4. [Visualizer tool](https://viz.mediapipe.dev/)

> used to visualize machine learning pipline;

[remain some framework concepts](https://google.github.io/mediapipe/framework_concepts/graphs.html)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mediapipe_usage/  

