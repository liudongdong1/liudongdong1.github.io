# Android_NDK开发教程


### 0. NDK 环境安装

- https://blog.csdn.net/hello_1995/article/details/108858909
- Tools->SDK Manager->Android SDK->SDK Tools，勾选 NDK 和 CMake
- 选中 Show Package Details 复选框，选中 NDK (Side by side) 复选框及其下方与您要安装的 NDK 版本对应的复选框。Android Studio 会将所有选中版本的 NDK 安装到 android-sdk/ndk/ 目录中

### 1. 编写JNI流程-- 静态注册方法

> 静态注册 JNI 方法的弊端非常明显，就是方法名会变得很长，而且当需要更改类名、包名或者方法时，需要按照之前方法重新生成头文件，灵活性不高.

（1）使用 **native** 关键字定义Java方法（即需要调用的 native 方法）

```java
package com.example.ndk;

public class NativeTest {
    public native void init();

    public native void init(int age);

    public native boolean init(String name);

    public native void update();
}
```

（2）使用 **javac** 编译上述 Java 源文件 （即 .java 文件）最终得到 .class文件

```shell
javac NativeTest.java
```

（3）通过 **javah** 命令编译 .class 文件,最终导出 JNI 的头文件（.h文件）

```java
javah com.example.ndk.NativeTest
```

```c++
#include <jni.h>
/* Header for class com_example_ndk_NativeTest */

#ifndef _Included_com_example_ndk_NativeTest
#define _Included_com_example_ndk_NativeTest
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_example_ndk_NativeTest
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_ndk_NativeTest_init__
  (JNIEnv *, jobject);

/*
 * Class:     com_example_ndk_NativeTest
 * Method:    init
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_example_ndk_NativeTest_init__I
  (JNIEnv *, jobject, jint);

/*
 * Class:     com_example_ndk_NativeTest
 * Method:    init
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_example_ndk_NativeTest_init__Ljava_lang_String_2
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_example_ndk_NativeTest
 * Method:    update
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_ndk_NativeTest_update
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
```

（4）使用 C/C++实现在 Java 中声明的 native 方法
（5）编译 .so 库文件， 使用CMake或者NDK-build进行编译
（6）通过 Java 代码加载动态库，然后调用 native 方法

### 2. 编写JNI流程-- 动态注册方法

> - 通过 vm（ Java 虚拟机）参数获取 JNIEnv 变量
>
> - 通过 FindClass 方法找到对应的 Java 类
> - 通过 RegisterNatives 方法，传入 JNINativeMethod 数组，注册 native 函数

- 准备JNINativeMethod 数组

```c
typedef struct {
	// Java层native方法名称
    const char* name;
	// 方法签名
    const char* signature;
	// native层方法指针
    void*       fnPtr;
} JNINativeMethod;

static JNINativeMethod methods[] = {
        {"init", "()V", (void *)c_init1},
        {"init", "(I)V", (void *)c_init2},
        {"init", "(Ljava/lang/String;)Z", (void *)c_init3},
        {"update", "()V", (void *)c_update},
};
```

- 重写 JNI_OnLoad 方法

```c
JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv *env = NULL;
    jint result = -1;
 
    // 获取JNI env变量
    if (vm->GetEnv((void**) &env, JNI_VERSION_1_6) != JNI_OK) {
        // 失败返回-1
        return result;
    }
 
    // 获取native方法所在类
    const char* className = "com/example/ndk/NativeTest";
    jclass clazz = env->FindClass(className);
    if (clazz == NULL) {
        return result;
    }
 
    // 动态注册native方法
    if (env->RegisterNatives(clazz, methods, sizeof(methods) / sizeof(methods[0])) < 0) {
        return result;
    }
 
    // 返回成功
    result = JNI_VERSION_1_6;
    return result;
}

extern "C" JNIEXPORT void JNICALL
c_init1(JNIEnv *env, jobject thiz) {
    // TODO: implement
}
extern "C" JNIEXPORT void JNICALL
c_init2(JNIEnv *env, jobject thiz, jint age) {
    // TODO: implement
}
extern "C" JNIEXPORT jboolean JNICALL
c_init3(JNIEnv *env, jobject thiz, jstring name) {
    // TODO: implement
}
extern "C" JNIEXPORT void JNICALL
c_update(JNIEnv *env, jobject thiz) {
    // TODO: implement
}
```



### 3. CMakeList.txt 语法

```shell
#1. 指定 cmake 的最小版本
cmake_minimum_required(VERSION 3.4.1)

#2. 设置项目名称
project(demo)

#3. 设置编译类型
add_executable(demo test.cpp) # 生成可执行文件
add_library(common STATIC test.cpp) # 生成静态库
add_library(common SHARED test.cpp) # 生成动态库或共享库

#4. 明确指定包含哪些源文件
add_library(demo test.cpp test1.cpp test2.cpp)

#5. 自定义搜索规则并加载文件
file(GLOB SRC_LIST "*.cpp" "protocol/*.cpp")
add_library(demo ${SRC_LIST}) //加载当前目录下所有的 cpp 文件
## 或者
file(GLOB SRC_LIST "*.cpp")
file(GLOB SRC_PROTOCOL_LIST "protocol/*.cpp")
add_library(demo ${SRC_LIST} ${SRC_PROTOCOL_LIST})
## 或者
aux_source_directory(. SRC_LIST)//搜索当前目录下的所有.cpp文件
aux_source_directory(protocol SRC_PROTOCOL_LIST) 
add_library(demo ${SRC_LIST} ${SRC_PROTOCOL_LIST})

#6. 用来显式地定义变量，多个变量用空格或分号隔开，引用的时候使用${SRC_LIST}
set(SRC_LIST main.cpp test.cpp)

#7. 在列表末尾添加新的对象
list(APPEND SRC_LIST new_test.cpp)

#8. 向终端输出用户定义的信息，包含了三种类型：SEND_ERROR，产生错误，生成过程被跳过；
# STATUS，输出前缀为—-的信息；FATAL_ERROR，立即终止所有 CMake 过程
message([SEND_ERROR | STATUS | FATAL_ERROR] “message to display” … )

#9. 查找指定库文件
find_library(
              log-lib //为 log 定义一个变量名称
              log ) //ndk 下的 log 库

#10. 设置包含的目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#11. 设置链接库搜索目录
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

#12. 设置 target 需要链接的库
target_link_libraries( # 目标库
                       demo
 
                       # 目标库需要链接的库
                       # log-lib 是上面 find_library 指定的变量名
                       ${log-lib} )
                       
#13. 指定链接动态库或者静态库
target_link_libraries(demo libtest.a) # 链接libtest.a
target_link_libraries(demo libtest.so) # 链接libtest.so

#14. 根据全路径链接动态静态库
target_link_libraries(demo ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtest.a)
target_link_libraries(demo ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtest.so)

#15. 指定链接多个库
target_link_libraries(demo
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtest.a
    test.a
    boost_thread
    pthread)
```

| 预定义变量               | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| PROJECT_SOURCE_DIR       | 工程的根目录                                                 |
| PROJECT_BINARY_DIR       | 运行 cmake 命令的目录，通常是 ${PROJECT_SOURCE_DIR}/build    |
| PROJECT_NAME             | 返回通过 project 命令定义的项目名称                          |
| CMAKE_CURRENT_SOURCE_DIR | 当前处理的 CMakeLists.txt 所在的路径                         |
| CMAKE_CURRENT_BINARY_DIR | target 编译目录                                              |
| CMAKE_CURRENT_LIST_DIR   | CMakeLists.txt 的完整路径                                    |
| CMAKE_CURRENT_LIST_LINE  | 当前所在的行                                                 |
| CMAKE_MODULE_PATH        | 定义自己的 cmake 模块所在的路径，SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)，然后可以用INCLUDE命令来调用自己的模块 |
| EXECUTABLE_OUTPUT_PATH   | 重新定义目标二进制可执行文件的存放位置                       |
| LIBRARY_OUTPUT_PATH      | 重新定义目标链接库文件的存放位置                             |


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/android_ndk%E5%BC%80%E5%8F%91%E6%95%99%E7%A8%8B/  

