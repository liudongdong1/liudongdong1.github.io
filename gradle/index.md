# Gradle


> 常见的 Java 构建工具包括 Ant、Gant 和 Maven 等，Gradle 结合了以上工具的优点，基于约定大于配置，通用灵活，是 Android 的官方构建工具。

### 1.  搭建环境

> 确保系统已安装 JDK 1.7及以上，此处将介绍 Gradle 在 Windows 平台下的手动安装。在 https://gradle.org/releases/ 下载最新的 release 包并解压至相应文件夹，然后在系统环境变量添加`GRADLE_HOME`，作者的变量值为`D:\gradle-5.4.1`，最后再将`%GRADLE_HOME%\bin`添加进`Path`变量中即可。在命令行中键入`gradle -v`验证环境搭建结果。 `IDEA`等工具中集成了gradle工具和maven工具，不需要额外下载

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619183659054.png)

### 2. Project

> 每个 Gradle 项目都由一个或多个 Project 构成，`每个 Project 又都由 Task 构成`。一个 `build.gradle`文件便是对一个 Project 对象的配置。在 Android 项目中，根目录会存在一个`build.gradle`文件，每个模块下也会有一个该文件。
>
> Gradle 包装器。为了应对团队开发中 Gradle 环境和版本的差异会对编译结果带来的不确定性，使用 Gradle Wrapper，它是一个脚本，可以指定构建版本、快速运行项目，从而达到标准化、提到开发效率。Android Studio 新建项目时自带 Gradle Wrapper，因此 Android 开发者很少单独下载安装 Gradle

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619184207482.png)

`Gradle 内置 Wrapper Task`，执行 Wrapper Task 就可以在项目目录中生成对应的目录文件。在项目根目录执行`gradle wrapper` 命令即可。之后根目录的文件结构如下

复制

```xml
├── gradle
│   └── wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradlew
└── gradlew.bat
```

- gradle-wrapper.jar ：`包含 Gradle 运行时的逻辑代码`。
- gradle-wrapper.properties ：负责`配置包装器运行时行为的属性文件`
- gradlew：Linux 平台下，用于执行` Gralde 命令的包装器脚本`。
- gradlew.bat：Windows 平台下，用于`执行 Gradle 命令的包装器脚本`。

查看`.properties`文件

```xml
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-5.4.1-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```

- distributionBase：`Gradle 解包后存储的主目录`。
- distributionPath：distributionBase `指定目录的子目录`。`distributionBase+distributionPath 为 Gradle 解包后的存放位置`。
- distributionUrl：`Gradle 发行版压缩包的下载地址`。
- zipStoreBase：Gradle `压缩包存储主目录`。
- zipStorePath：zipStoreBase 指定目录的子目录。zipStoreBase+zipStorePath 为 Gradle 压缩包的存放位置。

### 3. 插件

> - 脚本插件：额外的构建脚本，类似于一个 `build.gradle`
> - 对象插件：又叫二进制插件，是实现了 `Plugin 接口的类`

- build.gradle早期

```json
//构建脚本（给脚本用的脚本）声明gradle脚本本身需要使用的资源
buildscript {
    //存储一个属于gradle的变量，整个工程都能用，可通过gradle.ext.springBootVersion使用
    ext {
        springBootVersion = '2.1.2.RELEASE'
    }
    /*配置仓库地址，从而找到外部依赖
    按照你在文件中(build.gradle)仓库的顺序寻找所需依赖(如jar文件)，
    如果在某个仓库中找到了，那么将不再其它仓库中寻找
    */
    repositories {
        //mavenLocal()本地库，local repository(${user.home}/.m2/repository)
        mavenCentral()//maven的中央仓库
        //阿里云Maven远程仓库
        maven { url "http://maven.aliyun.com/nexus/content/groups/public/" }
    }
    /*配置springboot插件加载
    */
    dependencies {
        // classpath 声明说明了在执行其余的脚本时，ClassLoader 可以使用这些依赖项
        classpath("org.springframework.boot:spring-boot-gradle-plugin:${springBootVersion}")
    }
}
//使用以下插件
apply plugin: 'java'
apply plugin: 'org.springframework.boot'
apply plugin: 'io.spring.dependency-management'

group = 'com.example'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '1.8'//jvm版本要求
// 定义仓库
repositories {
    maven{url 'http://maven.aliyun.com/nexus/content/groups/public/'}
    maven{url 'https://mvnrepository.com/'}
    mavenCentral()
}
// 定义依赖:声明项目中需要哪些依赖
dependencies {
    compile  'org.springframework.boot:spring-boot-starter'
    compile('org.springframework.boot:spring-boot-starter-web')//引入web模块，springmvc
    compile  'org.springframework.boot:spring-boot-starter-test'
}
```

- build.gradle 新版本

```json
//使用以下插件
plugins {
    id 'org.springframework.boot' version '2.3.0.RELEASE'
    id 'io.spring.dependency-management' version '1.0.9.RELEASE'
    id 'java'
}
 
group = 'com.example'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '1.8' //jvm版本要求
 
// 定义仓库
repositories {
    maven{url 'http://maven.aliyun.com/nexus/content/groups/public/'}
    maven{url 'https://mvnrepository.com/'}
    mavenCentral()
}
//存储一个属于gradle的变量，整个工程都能用，可通过gradle.ext.springBootVersion使用
ext {
    set('springCloudVersion', "Hoxton.SR4")
}
 
// 定义依赖:声明项目中需要哪些依赖
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.cloud:spring-cloud-starter-openfeign'
    testImplementation('org.springframework.boot:spring-boot-starter-test') {
        exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
    }
}
 
dependencyManagement {
    imports {
        mavenBom "org.springframework.cloud:spring-cloud-dependencies:${springCloudVersion}"
    }
}
 
test {
    useJUnitPlatform()
}
```

- Settings.gradle

```json
//settings.gradles是模块Module配置文件，大多数setting.gradle的作用是为了配置子模块，
//根目录下的setting.gradle脚本文件是针对module的全局配置
 
//settings.gradle用于创建多Project的Gradle项目。Project在IDEA里对应Module模块。
 
//例如配置module名rootProject.name = 'project-root',为指定父模块的名称，
//include 'hello' 为指定包含哪些子模块
------------------------
//平台根
rootProject.name = 'project-root'
//包含子系统以及模块
include ':project-core'
//Hello系统模块的加载
include ':project-hello'
//World系统模块的加载
include ':project-world'
```

### 4. Android Gradle

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619194112903.png)

> Android Studio 中的每个项目包含一个或多个含有源代码文件和资源文件的模块，这些模块可以独立构建测试，模块类型包含以下几种
>
> - Android `应用程序模块`：可能依赖于库模块，构建`系统会将其生成一个 apk 文件`
> - Android `库模块`：包含可重用的特定于 Android 的代码和资源，构建系统将其生成一个 aar 文件
> - App` 引擎模`块：包含应用程序引擎继承的代码和资源
> - Java `库模块`：包含可重用的代码，构建系统将其生成一个 jar 文件

- setting.gradle

> settings.gradle 是负责配置项目的脚本
> 对应 [Settings](https://github.com/gradle/gradle/blob/v4.1.0/subprojects/core/src/main/java/org/gradle/api/initialization/Settings.java) 类，gradle 构建过程中，会根据 settings.gradle 生成 Settings 的对象
> 对应的可调用的方法在[文档](https://docs.gradle.org/current/dsl/org.gradle.api.initialization.Settings.html)里可以查找
> 其中几个主要的方法有:
>
> - include(projectPaths)
> - includeFlat(projectNames)
> - project(projectDir)

- 项目build.gradle

```json
buildscript {
    ext.kotlin_version = '1.3.10'
    repositories {
        mavenCentral()
        google()
        jcenter()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:3.4.0'
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
        classpath "org.jetbrains.kotlin:kotlin-android-extensions:$kotlin_version"
    }
}

allprojects {
    repositories {
        google()
        jcenter()
        mavenCentral()
        maven { url 'https://jitpack.io' }
        maven { url "https://maven.google.com" }
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}
```

| 属性                      | 描述                                                         |
| ------------------------- | ------------------------------------------------------------ |
| applicationId             | 指定App的包名                                                |
| minSdkVersion             | App最低支持的SDK版本                                         |
| targetSdkVersion          | 基于哪个SDK版本开发                                          |
| versionCode               | App内部的版本号，用于控制App升级                             |
| versionName               | App版本名称，也就是发布的版本号                              |
| testApplicationId         | 配置测试App的包名                                            |
| testInstrumentationRunner | 配置单元测试使用的Runner，默认为android.test.InstrumentationTestRunner |
| proguardFile              | ProGuard混淆所使用的ProGuard配置文件                         |
| proguardFiles             | 同时配置多个ProGuard配置文件                                 |
| signingConfig             | 配置默认的签名信息                                           |

| 属性                | 描述                                       |
| ------------------- | ------------------------------------------ |
| applicationIdSuffix | 配置applicationId的后缀                    |
| debuggable          | 表示是否支持断点调试                       |
| jniDebuggable       | 表示是否可以调试NDK代码                    |
| buildConfigField    | 配置不同的开发环境，比如测试环境和正式环境 |
| shrinkResources     | 是否自动清理未使用的资源，默认值为false    |
| zipAlignEnabled     | 是否开启开启zipalign优化，提高apk运行效率  |
| proguardFile        | ProGuard混淆所使用的ProGuard配置文件       |
| proguardFiles       | 同事配置多个ProGuard配置文件               |
| signingConfig       | 配置默认的签名信息                         |
| multiDexEnabled     | 是否启用自动拆分多个Dex的功能              |

- 模块build.gradle

```json
apply plugin: 'com.android.application' // 引入 android gradle 插件

android { // 配置 android gradle plugin 需要的内容
    compileSdkVersion 26
    defaultConfig { // 版本，applicationId 等配置
        applicationId "com.zy.easygradle"
        minSdkVersion 19
        targetSdkVersion 26
        versionCode 1
        versionName "1.0"
    }
    buildTypes { 
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions { // 指定 java 版本
        sourceCompatibility 1.8
        targetCompatibility 1.8
    }

    // flavor 相关配置
    flavorDimensions "size", "color"
    productFlavors {
        big {
            dimension "size"
        }
        small {
            dimension "size"
        }
        blue {
            dimension "color"
        }
        red {
            dimension "color"
        }
    }
}

// 项目需要的依赖
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar']) // jar 包依赖
    implementation 'com.android.support:appcompat-v7:26.1.0' // 远程仓库依赖
    implementation 'com.android.support.constraint:constraint-layout:1.1.3'
    implementation project(':module1') // 项目依赖
}
```

> `application`，说明当前模块为一个应用程序模块，Gradle 的 Android 插件分为以下几种
>
> - 应用程序插件：插件 id 为`com.android.application`，构建生成 apk
> - 库插件：插件 id 为`com.android.library`，构建生成 aar
> - 测试插件：插件 id 为`com.android.test`，用于测试其他模块
> - feature 插件：插件 id 为`com.android.feature`，用于创建 Android Instant App
> - instant App 插件：插件 id 为`com.android.instantapp`，是 Android Instant App 的入口

| 新配置         | 弃用配置 | 行为                                                         | 作用                                                         |
| -------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| implementation | compile  | 依赖项在编译时对模块可用，并且仅在运行时对模块的消费者可用。 对于大型多项目构建，使用 implementation 而不是 api/compile 可以显著缩短构建时间，因为它可以减少构建系统需要重新编译的项目量。 大多数应用和测试模块都应使用此配置。 | implementation 只会暴露给直接依赖的模块，使用此配置，在模块修改以后，只会重新编译直接依赖的模块，间接依赖的模块不需要改动 |
| api            | compile  | 依赖项在编译时对模块可用，并且在编译时和运行时还对模块的消费者可用。 此配置的行为类似于 compile（现在已弃用），一般情况下，您应当仅在库模块中使用它。 应用模块应使用 implementation，除非您想要将其 API 公开给单独的测试模块。 | api 会暴露给间接依赖的模块，使用此配置，在模块修改以后，模块的直接依赖和间接依赖的模块都需要重新编译 |
| compileOnly    | provided | 依赖项仅在编译时对模块可用，并且在编译或运行时对其消费者不可用。 此配置的行为类似于 provided（现在已弃用）。 | 只在编译期间依赖模块，打包以后运行时不会依赖，可以用来解决一些库冲突的问题 |
| runtimeOnly    | apk      | 依赖项仅在运行时对模块及其消费者可用。 此配置的行为类似于 apk（现在已弃用）。 | 只在运行时依赖模块，编译时不依赖                             |

> - implementation可以让module在编译时隐藏自己使用的依赖，但是在运行时这个依赖对所有模块是可见的。而api与compile一样，无法隐藏自己使用的依赖。
> - 如果使用api，一个module发生变化，这条依赖链上所有的module都需要重新编译，而使用implemention，只有直接依赖这个module需要重新编译。

### 5. gradle 生命周期及回调

- 初始化阶段主要做的事情是`有哪些项目需要被构建`，然后为对应的项目创建 Project 对象
- 配置阶段主要做的事情是对上一步创建的项目进行配置，这时候会执行 build.gradle 脚本，并且会生成要执行的 task
- 执行阶段主要做的事情就是执行 task，进行主要的构建工作

```java
gradle.addBuildListener(new BuildListener() {
    @Override
    void buildStarted(Gradle gradle) {
        println('构建开始')
        // 这个回调一般不会调用，因为我们注册的时机太晚，注册的时候构建已经开始了，是 gradle 内部使用的
    }

    @Override
    void settingsEvaluated(Settings settings) {
        println('settings 文件解析完成')
    }

    @Override
    void projectsLoaded(Gradle gradle) {
        println('项目加载完成')
        gradle.rootProject.subprojects.each { pro ->
            pro.beforeEvaluate {
                println("${pro.name} 项目配置之前调用")
            }
            pro.afterEvaluate{
                println("${pro.name} 项目配置之后调用")
            }
        }
    }

    @Override
    void projectsEvaluated(Gradle gradle) {
        println('项目解析完成')
    }

    @Override
    void buildFinished(BuildResult result) {
        println('构建完成')
    }
})

gradle.taskGraph.whenReady {
    println("task 图构建完成")
}
gradle.taskGraph.beforeTask {
    println("每个 task 执行前会调这个接口")
}
gradle.taskGraph.afterTask {
    println("每个 task 执行完成会调这个接口")
}
```

### 6. [gradle plugin](https://www.cnblogs.com/mingfeng002/p/11751119.html)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619194703629.png)

### 7. Problem Record

> 重启出现如下问题   Gradle sync failed: Connection timed out: connect, Consult IDE log for more details (Help | Show Log) (53s 730ms)

```java
distributionUrl=https\://services.gradle.org/distributions/gradle-4.0-milestone-1-all.zip
```

通过网址：http://services.gradle.org/distributions/下载,下载之后把下载的文件直接复制到C:\Users\Administrator\.gradle\wrapper\dists\gradle-x.x-all\中时间最近的目录下，单击Android Studio工具栏“Sync Project Gradle Files”或者重启Android Studio，问题就可以解决了。 `或者修改gradle版本也可以`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210619195415495.png)

### Resource

- https://www.cnblogs.com/mingfeng002/p/11751119.html
- https://docs.gradle.org/current/userguide/build_environment.html
- [Building Android apps](https://docs.gradle.org/current/samples/sample_building_android_apps.html)
- [Building Java applications](https://docs.gradle.org/current/samples/sample_building_java_applications.html)
- [Building Java libraries](https://docs.gradle.org/current/samples/sample_building_java_libraries.html)
- [Building Groovy applications](https://docs.gradle.org/current/samples/sample_building_groovy_applications.html)
- [Building Groovy libraries](https://docs.gradle.org/current/samples/sample_building_groovy_libraries.html)
- [Building Scala applications](https://docs.gradle.org/current/samples/sample_building_scala_applications.html)
- [Building Scala libraries](https://docs.gradle.org/current/samples/sample_building_scala_libraries.html)
- [Building Kotlin JVM applications](https://docs.gradle.org/current/samples/sample_building_kotlin_applications.html)
- [Building Kotlin JVM libraries](https://docs.gradle.org/current/samples/sample_building_kotlin_libraries.html)
- [Building C++ applications](https://docs.gradle.org/current/samples/sample_building_cpp_applications.html)
- [Building C++ libraries](https://docs.gradle.org/current/samples/sample_building_cpp_libraries.html)
- [Building Swift applications](https://docs.gradle.org/current/samples/sample_building_swift_applications.html)
- [Building Swift libraries](https://docs.gradle.org/current/samples/sample_building_swift_libraries.html)
- [Creating build scans](https://scans.gradle.com/)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/gradle/  

