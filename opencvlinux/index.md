# OpencvLinux


> 在源代码编译mediapipe的时候，在编译安装opencv3.4.12时，遇到一些问题，记录下来。

### 1. opencv3.4.12 Install

```shell
# To have a full installation:
# $ cd <mediapipe root dir>
# $ sh ./setup_opencv.sh
#
# To only modify the mediapipe config for opencv:
# $ cd <mediapipe root dir>
# $ sh ./setup_opencv.sh config_only

set -e
if [ "$1" ] && [ "$1" != "config_only" ]
  then
    echo "Unknown input argument. Do you mean \"config_only\"?"
    exit 0
fi

opencv_build_file="$( cd "$(dirname "$0")" ; pwd -P )"/third_party/opencv_linux.BUILD
echo $opencv_build_file

workspace_file="$( cd "$(dirname "$0")" ; pwd -P )"/WORKSPACE
echo $workspace_file
if [ -z "$1" ]
  then
    echo "Installing OpenCV from source"
    sudo apt update && sudo apt install build-essential git
    sudo apt install cmake ffmpeg libavformat-dev libdc1394-22-dev libgtk2.0-dev \
                     libjpeg-dev libpng-dev libswscale-dev libtbb2 libtbb-dev \
                     libtiff-dev
    #rm -rf /tmp/build_opencv
    #mkdir /tmp/build_opencv
    cd /tmp/build_opencv
    #git clone https://github.com/opencv/opencv_contrib.git
    #git clone https://github.com/opencv/opencv.git
    mkdir opencv-3.4.12/release
    #cd opencv_contrib
   # git checkout 3.4
    #cd ../opencv
    #git checkout 3.4
    cd opencv-3.4.12/release
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_ts=OFF \
          -DOPENCV_EXTRA_MODULES_PATH=/tmp/build_opencv/opencv_contrib-3.4.12/modules \
          -DBUILD_opencv_aruco=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF \
          -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn=OFF \
          -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=OFF \
          -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF \
          -DBUILD_opencv_js=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_phase_unwrapping=OFF \
          -DBUILD_opencv_plot=OFF -DBUILD_opencv_quality=OFF -DBUILD_opencv_reg=OFF \
          -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_shape=OFF \
          -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_surface_matching=OFF \
          -DBUILD_opencv_world=OFF -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF
    make -j 16
    sudo make install
    rm -rf /tmp/build_opencv
    echo "OpenCV has been built. You can find the header files and libraries in /usr/local/include/opencv2/ and /usr/local/lib"

    # https://github.com/cggos/dip_cvqt/issues/1#issuecomment-284103343
    sudo touch /etc/ld.so.conf.d/mp_opencv.conf
    sudo bash -c  "echo /usr/local/lib >> /etc/ld.so.conf.d/mp_opencv.conf"
    sudo ldconfig -v
fi
```

> `~/opencv_contrib/modules/xfeatures2d/src/boostdesc.cpp:673:20: fatal error: boostdesc_bgm.i: No such file or directory`
>
> 解决方法：
>
> 查看 build 文件夹下的日志文件 CMakeDownloadLog.txt，在日志文件CMakeDownloadLog.txt中搜索 boostdesc_bgm.i 关键词。日志文件里就有它的下载地址，到指定位置下载即可。https://github.com/opencv/opencv_contrib/issues/1301，点开上面这个网址往下拉，有人提供了缺失的各个文件的链接，点击保存. 或者直接在这个网页里搜索 BenbenIO 这个用户的回答。
>
> 或者到本文提供的下载镜像去下载：[boostdesc_bgm.i,vgg_generated_48.i等.rar](https://files.cnblogs.com/files/arxive/boostdesc_bgm.i%2Cvgg_generated_48.i等.rar)
>
> 下载后，直接拷贝源码并生存同名文件，放在 **opencv_contrib/modules/xfeatures2d/src/** 路径下即可。

> `对于opencv2/xfeatures2d/cuda.hpp: No such file or directory 类问题的解决方法`  改为绝对目录
>
> ```shell
> 
> /usr/local/arm/opencv-3.4.0/opencv_contrib-3.4.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp:42:10: 
> fatal error: /opencv2/xfeatures2d.hpp: No such file or directory
>  #include "/opencv2/xfeatures2d.hpp"
>           ^~~~~~~~~~~~~~~~~~~~~~~~~~
> compilation terminated.
> ```
>
> ```shell
> 40 #ifndef __OPENCV_XFEATURES2D_HPP__
> 41 #define __OPENCV_XFEATURES2D_HPP__
>  
> 42#include"/usr/local/arm/opencv3.4.0/opencv_contrib3.4.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp"
> ```

### 2. 测试

安装测试代码；

```c++
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
    cout << "Hello OpenCV " << CV_VERSION << endl;
    Mat myMat = imread("timg.jpeg", 1);
    namedWindow("Opencv Image", WINDOW_AUTOSIZE);
    imshow("Opencv Image", myMat);
    waitKey(5000);
    return 0;
}
```

- 方式一： g++ test.cpp `pkg-config --libs --cflags opencv` -o opencv
- 方式二： cmakelist.txt

```shell
cmake_minimum_required(VERSION 2.8)
project(opencv)
find_package(OpenCV REQUIRED)
add_executable(opencv opencv.cpp)
target_link_libraries(opencv ${OpenCV_LIBS})
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/opencvlinux/  

