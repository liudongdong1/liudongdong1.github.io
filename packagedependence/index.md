# PackageDependence


> pkg-config是一个linux下的命令，用于获得某一个库/模块的所有编译相关的信息。`如果你写了一个库，不管是静态的还是动态的，要提供给第三方使用，那除了给人家库/头文件，最好也写一个pc文件，这样别人使用就方便很多，不用自己再手动写依赖了你哪些库，只需要敲一个”pkg-config [YOUR_LIB] –libs –cflags”`。  vcpkg windows 对应工具

### 1. pkg-config 命令

> 第一种：取系统的/usr/lib下的所有*.pc文件。
> 第二种：PKG_CONFIG_PATH环境变量所指向的路径下的所有*.pc文件。

```shell
# Package Information for pkg-config
prefix=/usr
exec_prefix=${prefix}
libdir=${prefix}/lib/x86_64-linux-gnu
includedir_old=${prefix}/include/opencv
includedir_new=${prefix}/include

Name: OpenCV
Description: Open Source Computer Vision Library
Version: 2.4.8
Libs: -L${libdir} ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_calib3d.so -lopencv_calib3d ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_contrib.so -lopencv_contrib ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_core.so -lopencv_core ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_features2d.so -lopencv_features2d ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_flann.so -lopencv_flann ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_gpu.so -lopencv_gpu ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_highgui.so -lopencv_highgui ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_imgproc.so -lopencv_imgproc ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_legacy.so -lopencv_legacy ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_ml.so -lopencv_ml ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_objdetect.so -lopencv_objdetect ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_ocl.so -lopencv_ocl ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_photo.so -lopencv_photo ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_stitching.so -lopencv_stitching ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_superres.so -lopencv_superres ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_ts.so -lopencv_ts ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_video.so -lopencv_video ${exec_prefix}/lib/x86_64-linux-gnu/libopencv_videostab.so -lopencv_videostab
Cflags: -I${includedir_old} -I${includedir_new}
```

> **pc文件的所有参数：**
>
> `Name:`该模块的名字，比如你的pc名字是xxxx.pc，那么名字最好也是xxxx。
> **Description:** 模块的简单描述。上文pkg-config –list-all命令出来的结果，每个名字后面就是description。
> **URL:** 用户可以通过该URL获得更多信息，或者下载信息。也是辅助的，可要可不要。
> `Version:` 版本号。
> **Requires:** 该模块有木有依赖于其他模块。一般没有。
> **Requires.private:** 该模块有木有依赖于其他模块，并且还不需要第三方知道的。一般也没有。
> **Conflicts:** 有没有和别的模块冲突。常用于版本冲突。比如，Conflicts: bar < 1.2.3，表示和bar模块的1.2.3以下的版本有冲突。
> `Cflags:` 这个就很重要了。pkg-config的参数–cflags就指向这里。主要用于写本模块的头文件的路径。
> `Libs:` 也很重要，pkg-config的参数–libs就指向这里。主要用于写本模块的库/依赖库的路径。
> **Libs.private:** 本模块依赖的库，但不需要第三方知道。

### 2. 常用命令

```shell
pkg-config [NAME] –cflags  # 查看头文件信息
pkg-config [NAME] –libs #查看库信息。
pkg-config –list-all  #查看pkg-config的所有模块信息。
```

###  3. pc文件

```shell
# 动态链接库
prefix=/usr/local
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib

Name: foo
Description: The foo library
Version: 1.0.0
Cflags: -I${includedir}/foo
Libs: -L${libdir} -lfoo
```

```shell
#静态库链接动态库时，如何使用该静态库，如果我有个静态库libXXX.a，它依赖了很多其他动态库libAA.so，libBB.so，那么第三方程序DD.c要使用libXXX.a时，编译时还得链接libAA.so，libBB.so。
#如何让第三方程序，可以不用操心我这个libXXX.a到底依赖了什么？很简答，就是我的libXXX.a写一个pc文件。
prefix=/home/chenxf/ffmpeg_build
exec_prefix=${prefix}
libdir=${prefix}/lib
includedir=${prefix}/include

Name: libavcodec
Description: FFmpeg codec library
Version: 57.38.100
Requires: libswresample >= 2.0.101, libavutil >= 55.22.101
Requires.private:
Conflicts:
Libs: -L${libdir}  -lavcodec -lXv -lX11 -lXext -lxcb -lXau -lXdmcp -lxcb-shm -lxcb -lXau -lXdmcp -lxcb-xfixes -lxcb-render -lxcb-shape -lxcb -lXau -lXdmcp -lxcb-shape -lxcb -lXau -lXdmcp -lX11 -lasound -L/home/chenxf/ffmpeg_build/lib -lx265 -lstdc++ -lm -lrt -ldl -L/home/chenxf/ffmpeg_build/lib -lx264 -lpthread -lm -ldl -lfreetype -lz -lpng12 -lass -lfontconfig -lenca -lm -lfribidi -lexpat -lfreetype -lz -lpng12 -lm -llzma -lz -pthread
Libs.private:
Cflags: -I${includedir}

#添加环境变量PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/chenxf/ffmpeg_build/lib/pkgconfig
#export PKG_CONFIG_PATH
#gcc test.c -o test ‘pkg-config libavcodec libavformat libavutil --cflags --libs’
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/packagedependence/  

