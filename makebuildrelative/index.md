# MakeBuildRelative


> **预处理：**g++ -E main.cpp -o main.ii，-E表示只进行预处理。预处理主要是处理各种宏展开；添加行号和文件标识符，为编译器产生调试信息提供便利；删除注释；保留编译器用到的编译器指令等。
>
> **编译：**g++ -S main.ii –o main.s，-S表示只编译。编译是在预处理文件基础上经过一系列词法分析、语法分析及优化后生成汇编代码。
>
> **汇编：**g++ -c main.s –o main.o。汇编是将汇编代码转化为机器可以执行的指令。
>
> **链接：**g++ main.o。链接生成可执行程序，之所以需要链接是因为我们代码不可能像main.cpp这么简单，现代软件动则成百上千万行，如果写在一个main.cpp既不利于分工合作，也无法维护，因此通常是由一堆cpp文件组成，编译器分别编译每个cpp，这些cpp里会引用别的模块中的函数或全局变量，在编译单个cpp的时候是没法知道它们的准确地址，因此在编译结束后，需要链接器将各种还没有准确地址的符号（函数、变量等）设置为正确的值，这样组装在一起就可以形成一个完整的可执行程序。

## 1. CMakeLists.txt

```shell
# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 2.8 )
# 添加c++11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )
# 声明一个cmake工程
project( 工程名 )
MESSAGE(STATUS "Project: SERVER")               #打印相关消息消息
# 找到后面需要库和头文件的包
find_package（包的名称及最低版本）
# 例如find_package(OpenCV 2.4.3 REQUIRED)
# 头文件
include_directories("路径")
# 例如
#include_directories(
# ${PROJECT_SOURCE_DIR}
# ${PROJECT_SOURCE_DIR}/include
# ${EIGEN3_INCLUDE_DIR}
)
# 设置路径（下面生成共享库的路径）
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# 即生成的共享库在工程文件夹下的lib文件夹中
# 创建共享库（把工程内的cpp文件都创建成共享库文件，方便通过头文件来调用）
add_library(${PROJECT_NAME} SHARED
src/cpp文件名
……
）
# 这时候只需要cpp，不需要有主函数
# ${PROJECT_NAME}是生成的库名 表示生成的共享库文件就叫做 lib工程名.so
# 也可以专门写cmakelists来编译一个没有主函数的程序来生成共享库，供其它程序使用
 
# 链接库
# 把刚刚生成的${PROJECT_NAME}库和所需的其它库链接起来
target_link_libraries(${PROJECT_NAME}
/usr/lib/i386-linux-gnu/libboost_system.so
)
# 编译主函数，生成可执行文件
# 先设置路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# 可执行文件生成
add_executable(要生成的可执行文件名 从工程目录下写起的主函数文件名)
# 这个可执行文件所需的库（一般就是刚刚生成的工程的库咯）
target_link_libraries(可执行文件名 ${PROJECT_NAME})
```

> cmake中一些预定义变量
>
> - `PROJECT_SOURCE_DIR 工程的根目录`
> - `PROJECT_BINARY_DIR` 运行`cmake命令的目录`,通常是`${PROJECT_SOURCE_DIR}/build`
> - CMAKE_INCLUDE_PATH 环境变量,非cmake变量
> - CMAKE_LIBRARY_PATH 环境变量
> - `CMAKE_CURRENT_SOURCE_DIR` `当前处理的CMakeLists.txt所在的路径`
> - `CMAKE_CURRENT_BINARY_DIR target`编译目录
>   使用ADD_SURDIRECTORY(src bin)可以更改此变量的值
>   SET(EXECUTABLE_OUTPUT_PATH <新路径>)并不会对此变量有影响,只是改变了最终目标文件的存储路径
> - CMAKE_CURRENT_LIST_FILE 输出调用这个变量的CMakeLists.txt的完整路径
> - CMAKE_CURRENT_LIST_LINE 输出这个变量所在的行
> - CMAKE_MODULE_PATH 定义自己的cmake模块所在的路径
>   SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake),然后可以用INCLUDE命令来调用自己的模块
> - EXECUTABLE_OUTPUT_PATH 重新定义目标二进制可执行文件的存放位置
> - LIBRARY_OUTPUT_PATH 重新定义目标链接库文件的存放位置
> - PROJECT_NAME 返回通过PROJECT指令定义的项目名称
> - CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS 用来控制IF ELSE语句的书写方式

### 2. Make install 背后

> `configure --prefix=/...`：指定安装路径
> 不指定prefix，则`可执行文件默认放在/usr /local/bin`，`库文件默认放在/usr/local/lib`，`配置文件默认放在/usr/local/etc`。其它的资源文件放在/usr /local/share。你要卸载这个程序，要么在`原来的make目录下用一次make uninstall`（前提是make文件指定过uninstall）,要么`去上述目录里面把相关的文件一个个手工删掉。`
> `指定prefix，直接删掉一个文件夹就够了。`

### 3. opencv CMakeLists.txt 案例

```shell
# cmake needs this line
cmake_minimum_required(VERSION 3.1) 
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Define project name
project(facedetect_project)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
#set(${OpenCV_DIR} )
set(OpenCV_VERSION 4.1)
set(OpenCV_LIBS lib)
set(OpenCV_INCLUDE_DIRS include)
set(LINK_DIR lib)
INCLUDE_DIRECTORIES(
    include
    )
link_directories(${LINK_DIR})
 
set(PROJECT_NAME
    opencv_core
    opencv_features2d
    opencv_highgui
    opencv_objdetect
    # opencv_imgcodecs
    opencv_imgproc
    # opencv_photo
    opencv_videoio
    opencv_video
)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 
# Declare the executable target built from your sources
add_executable(facedetect face_detect.cpp)
#add_executable(opencv_example main.cpp)
# Link your application with OpenCV libraries
target_link_libraries(facedetect ${PROJECT_NAME})
```

### 4. Makefile example

```shell
TOPDIR  := .
CROSS_COMPILE=arm-linux-
AS      =$(CROSS_COMPILE)as
LD      =$(CROSS_COMPILE)ld
CC      =$(CROSS_COMPILE)gcc
CPP     =$(CC) -E
AR      =$(CROSS_COMPILE)ar
NM      =$(CROSS_COMPILE)nm
STRIP   =$(CROSS_COMPILE)strip
OBJCOPY =$(CROSS_COMPILE)objcopy
OBJDUMP =$(CROSS_COMPILE)objdump
EXTRA_LIBS += -lpthread
EXEC= test_led
OBJS= keyboard.o get_key.o test_led.o
all: $(EXEC)
$(EXEC): $(OBJS)
	$(CC)  -o $@ $(OBJS)  $(EXTRA_LIBS)
install:
	$(EXP_INSTALL) $(EXEC) $(INSTALL_DIR)
clean:
	-rm -f $(EXEC) *.elf *.gdb *.o
clean:
	rm -f *.o *~ core .depend
```

INCLUDE_DIRECTORIES("/tmp/build_opencv/opencv_contrib-3.4.12/modules/xfeatures2d/include")

/tmp/build_opencv/opencv_contrib-3.4.12/modules/xfeatures2d/include/opencv2/xfeatures2d/

### Cmake 

- 中find_package() 工作原理：https://www.jianshu.com/p/46e9b8a6cb6a

```shell
cmake-gui #图像化cmake
cmake --version
apt-get remove cmake
cd /usr/local/src
wget https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3.tar.gz
tar -xvzf cmake-3.15.3.tar.gz
cd cmake-3.15.3
./bootstrap
make -j4
make install
##python 使用C++11 框架 pylind11
```

### gcc cpp g++ 区别

```shell
gcc和g++的主要区别
# 1. 对于 *.c和*.cpp文件，gcc分别当做c和cpp文件编译（c和cpp的语法强度是不一样的）
# 2. 对于 *.c和*.cpp文件，g++则统一当做cpp文件编译
# 3. 使用g++编译文件时，g++会自动链接标准库STL，而gcc不会自动链接STL
# 4. gcc在编译C文件时，可使用的预定义宏是比较少的
# 5. gcc在编译cpp文件时/g++在编译c文件和cpp文件时（这时候gcc和g++调用的都是cpp文件的编译器），会加入一些额外的宏，这些宏如下：
# 6. 在用gcc编译c++文件时，为了能够使用STL，需要加参数 –lstdc++ ，但这并不代表 gcc –lstdc++ 和 g++等价，它们的区别不仅仅是这个
#gcc 版本
gcc -version
sudo apt-get install gcc-5 g++-5
```


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/makebuildrelative/  

