# AndroidStudio_工具


### 1. SDK 工具

- **apkanalyzer**：用于在构建过程完成后深入分析我们的 APK 组成。
- **avdmanager**：可让我们从命令行创建和管理 Android `虚拟设备` (AVD)。
- **sdkmanager**：可让我们`查看、安装、更新和卸载 Android SDK `的软件包。
- **jobb**：可以让我们构建不透明二进制 Blob (OBB) 格式的已加密和未加密 APK 扩展文件。

#### .1. apkanalyzer

```shell
apkanalyzer [global-options] subject verb [options] apk-file [apk-file2]
```

- **查看 APK 文件属性**

| 命令选项          | 说明                                                         |
| :---------------- | :----------------------------------------------------------- |
| apk summary       | 输出应用 ID、版本代码和版本名称。                            |
| apk file-size     | 输出 APK 的总文件大小。                                      |
| apk download-size | 输出 APK 的下载大小估计值。                                  |
| apk features      | 输出 APK 用来触发 Play 商店过滤的功能。                      |
| apk compare       | 比较 apk-file 和 apk-file2 的大小。 --different-only：输出存在差异的目录和文件。 --files-only：不输出目录条目。 --patch-size：显示逐个文件的补丁程序大小估计值，而不是原始差异。 |

- **查看 APK 文件系统 **

| 命令选项         | 说明                                                       |
| :--------------- | :--------------------------------------------------------- |
| files list       | 列出 APK 中的所有文件。                                    |
| files cat --file | 输出文件内容。必须使用 --file path 选项指定 APK 内的路径。 |

- **查看清单中的信息**

| 命令选项                | 说明                       |
| :---------------------- | :------------------------- |
| manifest print          | 以 XML 格式输出 APK 清单。 |
| manifest application-id | 输出应用 ID 值。           |
| manifest version-name   | 输出版本名称值。           |
| manifest version-code   | 输出版本代码值。           |
| manifest min-sdk        | 输出最低 SDK 版本。        |
| manifest target-sdk     | 输出目标 SDK 版本。        |
| manifest permissions    | 输出权限列表。             |
| manifest debuggable     | 输出应用是否可调试。       |

- **访问 DEX 文件信息**

| 命令选项         | 说明                                                         |
| :--------------- | :----------------------------------------------------------- |
| dex list         | 输出 APK 中的 DEX 文件列表。                                 |
| dex references   | 输出指定 DEX 文件中的方法引用数。                            |
| dex packages     | 输出 DEX 中的类树。在输出中，P、C、M 和 F 分别表示软件包、类、方法和字段。–defined-only：在输出中仅包含 APK 中定义的类。 –files：指定要包含的 DEX 文件名。默认：所有 DEX 文件。 –proguard-folder file：指定用于搜索映射的 Proguard 输出文件夹。 –proguard-mappings file：指定 Proguard 映射文件。 –proguard-seeds file：指定 Proguard 种子文件。 –proguard-usages file：指定 Proguard 用法文件。 |
| dex code --class | 以 smali 格式输出类或方法的字节码。输出中必须包含类名，并且要输出完全限定类名以进行反编译。 |

- **看存储在 APK 的资源**

| 命令选项                               | 说明                                                   |
| :------------------------------------- | :----------------------------------------------------- |
| resources packages                     | 输出资源表中定义的软件包列表。                         |
| resources configs --type               | 输出指定 type 的配置列表。type 是资源类型，如 string。 |
| resources value --config --name --type | 输出由 config、name 和 type 指定的资源的值。           |
| resources names --config --type        | 输出属于某种配置和类型的资源名称列表。                 |
| resources xml --file                   | 以简单易懂的形式输出 XML 二进制文件。                  |

#### .2. avdmanager

- 从命令行创建和管理 Android 虚拟设备 (AVD)。借助 AVD，我们可以定义要在 Android 模拟器中模拟的 Android 手机、Wear OS 手表或 Android TV 设备的特性。

```
avdmanager [global options] command [command options]
```

| 命令选项                                      | 说明                                                         |
| :-------------------------------------------- | :----------------------------------------------------------- |
| create avd -n name -k “sdk_id” [-c] [-f] [-p] | 创建一个新的 AVD。必须为该 AVD 提供一个名称，并使用加引号的 sdk_id 指定要用于该 AVD 的 SDK 软件包的 ID。 -c：此 AVD 的 SD 卡映像的路径，或要为此 AVD 创建的新 SD 卡映像的大小。 -f：强制创建 AVD。 -p：将从中创建此 AVD 的文件的目录所在位置的路径。 |
| delete avd -n                                 | 删除一个 AVD。必须使用 name 指定该 AVD。                     |
| move avd -n name [-p] [-r]                    | 移动和/或重命名一个 AVD。必须使用 name 指定该 AVD。 -p：用于接收此 AVD 的文件的目录所在位置的绝对路径。 -r：AVD 的新名称。 |
| list                                          | 列出所有可用的目标、设备定义或 AVD。                         |

#### .3. adkmanager

- 查看、安装、更新和卸载 Android SDK 的软件包

```
sdkmanger --list [options]
sdkmanger packages [options]
sdkmanager --update [options]
```

| 命令选项          | 说明                                                         |
| :---------------- | :----------------------------------------------------------- |
| –sdk_root         | 使用指定的 SDK 路径而不是包含此工具的 SDK。                  |
| –channel          | 包含从 channel_0 到 channel_id 所有渠道中的软件包。 可用的渠道包括：0（稳定版）、1（测试版）、2（开发版）和 3（Canary 版）。 |
| –include_obsolete | 在列出或更新软件包时纳入那些已过时的软件包。                 |
| –no_https         | 强制所有连接使用 HTTP 而不是 HTTPS。                         |
| –verbose          | 详细输出模式。该模式会输出错误、警告和参考性消息。           |
| –proxy            | 通过给定类型的代理建立连接：用 http 指定一个高层级协议（如 HTTP 或 FTP）的代理，或用 socks 指定一个 SOCKS（V4 或 V5）代理。 |
| –proxy_host       | 要使用的代理的 IP 或 DNS 地址。                              |
| –proxy_port       | 要连接到的代理端口号。                                       |

#### .4. jobb

- 构建不透明二进制 Blob (OBB) 格式的已加密和未加密 APK 扩展文件。OBB 文件用于为 Android 应用提供额外文件资源（例如图形、音频和视频），这些文件资源与应用的 APK 文件是分开的。

```shell
jobb [-d <directory>][-o <filename>][-pn <package>][-pv <version>] \
     [-k <key>][-ov][-dump <filename>][-v][-about]
```

| 命令选项 | 说明                                                         |
| :------- | :----------------------------------------------------------- |
| -d       | 设置创建 OBB 文件时所用的输入目录，或提取 (-dump) 现有文件时所用的输出目录。创建 OBB 文件时，指定目录及其所有子目录的内容都将包含在 OBB 文件系统中。 |
| -o       | 指定 OBB 文件的文件名。创建 OBB 和提取（转储）其内容时，必须提供此参数。 |
| -pn      | 指定装载 OBB 文件的应用的软件包名称，该名称对应于应用清单中指定的 package 值。创建 OBB 文件时，必须提供此参数。 |
| -pv      | 设置可装载 OBB 文件的应用的最低版本，这对应于应用清单中的 android:versionCode 值。创建 OBB 文件时，必须提供此参数。 |
| -k       | 指定用于加密新 OBB 文件或解密现有的已加密 OBB 文件的密码。   |
| -ov      | 创建叠加在现有 OBB 文件结构上的 OBB 文件。该选项可让我们将新文件包的内容装载到先前的文件包所在的位置，旨在用于创建之前生成的 OBB 文件的补丁版本。 |
| -dump    | 提取指定 OBB 文件的内容。                                    |
| -v       | 设置该工具的详细输出。                                       |
| -about   | 显示 jobb 工具的版本和帮助信息。                             |

### 2. sdk 构建工具

- **aapt2**：解析 Android 资源，为其编制索引，然后将其编译为针对 Android 平台优化的二进制格式，最后将编译后的资源打包到单个输出中。
- **apksigner**：为 APK 签名，并检查签名能否在给定 APK 支持的所有平台版本上成功通过验证。
- **zipalign**：确保所有未压缩数据的开头均相对于文件开头部分执行特定的对齐，从而优化 APK 文件。

#### .1. aapt2

- AAPT2 会解析资源、为资源编制索引，并将资源编译为针对 Android 平台进行过优化的二进制格式。
- AAPT2 支持通过启用增量编译实现更快的资源编译。包括编译和链接

```shell
aapt2 compile path-to-input-files [options] -o output-directory/

aapt2 compile project_root/module_root/src/main/res/values-en/
strings.xml -o compiled/
```

| 命令选项         | 说明                                              |
| :--------------- | :------------------------------------------------ |
| -o               | 指定已编译资源的输出路径。                        |
| –dir             | 指定要在其中搜索资源的目录。                      |
| –pseudo-localize | 生成默认字符串的伪本地化版本，如 en-XA 和 en-XB。 |
| –no-crunch       | 停用 PNG 处理。                                   |
| –legacy          | 将使用早期版本的 AAPT 时允许的错误视为警告。      |
| -v               | 启用详细日志记录。                                |

```shell
aapt2 link -o output.apk
 -I android_sdk/platforms/android_version/android.jar
    compiled/res/values_values.arsc.flat
    compiled/res/drawable_Image.flat --manifest /path/to/AndroidManifest.xml -v
```

| 命令选项                               | 说明                                                         |
| :------------------------------------- | :----------------------------------------------------------- |
| -o                                     | 指定链接的资源 APK 的输出路径。                              |
| –manifest                              | 指定要构建的 Android 清单文件的路径。                        |
| -I                                     | 提供平台的 android.jar 或其他 APK（如 framework-res.apk）的路径。 |
| -A                                     | 指定要包含在 APK 中的资产目录。                              |
| -R                                     | 传递要链接的单个 .flat 文件，使用 overlay 语义。             |
| –package-id                            | 指定要用于应用的软件包 ID。                                  |
| –allow-reserved-package-id             | 允许使用保留的软件包 ID。                                    |
| –java                                  | 指定要在其中生成 R.java 的目录。                             |
| –proguard                              | 为 ProGuard 规则生成输出文件。                               |
| –proguard-conditional-keep-rules       | 为主 dex 的 ProGuard 规则生成输出文件。                      |
| –no-auto-version                       | 停用自动样式和布局 SDK 版本控制。                            |
| –no-version-vectors                    | 停用矢量可绘制对象的自动版本控制。                           |
| –no-version-transitions                | 停用转换资源的自动版本控制。                                 |
| –no-resource-deduping                  | 禁止在兼容配置中自动删除具有相同值的重复资源。               |
| –enable-sparse-encoding                | 允许使用二进制搜索树对稀疏条目进行编码。                     |
| -z                                     | 要求对标记为“建议”的字符串进行本地化。                       |
| -c                                     | 提供以英文逗号分隔的配置列表。                               |
| –preferred-density                     | 允许 AAPT2 选择最相符的密度并删除其他所有密度。              |
| –output-to-dir                         | 将 APK 内容输出到 -o 指定的目录中。                          |
| –min-sdk-version                       | 设置要用于 AndroidManifest.xml 的默认最低 SDK 版本。         |
| –target-sdk-version                    | 设置要用于 AndroidManifest.xml 的默认目标 SDK 版本。         |
| –version-code                          | 指定没有版本代码时要注入 AndroidManifest.xml 中的版本代码。  |
| –compile-sdk-version-name              | 指定没有版本名称时要注入 AndroidManifest.xml 中的版本名称。  |
| –proto-format                          | 以 Protobuf 格式生成已编译的资源。                           |
| –non-final-ids                         | 使用非最终资源 ID 生成 R.java。                              |
| –emit-ids                              | 在给定的路径上生成一个文件，该文件包含资源类型的名称及其 ID 映射的列表。 |
| –stable-ids                            | 使用通过 --emit-ids 生成的文件，该文件包含资源类型的名称以及为其分配的 ID 的列表。 |
| –custom-package                        | 指定要在其下生成 R.java 的自定义 Java 软件包。               |
| –extra-packages                        | 生成相同的 R.java 文件，但软件包名称不同。                   |
| –add-javadoc-annotation                | 向已生成的所有 Java 类添加 JavaDoc 注释。                    |
| –output-text-symbols                   | 生成包含指定文件中 R 类的资源符号的文本文件。                |
| –auto-add-overlay                      | 允许在叠加层中添加新资源。                                   |
| –rename-manifest-package               | 重命名 AndroidManifest.xml 中的软件包。                      |
| –rename-instrumentation-target-package | 更改插桩的目标软件包的名称。                                 |
| -0                                     | 指定不想压缩的文件的扩展名。                                 |
| –split                                 | 根据一组配置拆分资源，以生成另一个版本的 APK。               |
| -v                                     | 可提高输出的详细程度。                                       |

#### .2. apksigner

- 在使用 apksigner 工具为 APK 签名时，必须提供签名者的私钥和证书。
  - 使用 **–ks** 选项指定密钥库文件。
  - 使用 **–key** 和 **–cert** 选项分别指定私钥文件和证书文件。私钥文件必须使用 PKCS #8 格式，证书文件必须使用 X.509 格式。

```shell
apksigner sign --ks release.jks app.apk

apksigner sign --key release.pk8 --cert release.x509.pem app.apk

#使用多个签名
apksigner sign --ks first-release-key.jks --next-signer --ks second-release-key.jks app.apk

apksigner verify app.apk
```

#### .3. zipalign

- zipalign 是一种归档对齐工具，可对 Android 应用 (APK) 文件提供重要的优化。其目的是要确保所有未压缩数据的开头均相对于文件开头部分执行特定的对齐。

```shell
zipalign [-f] [-v] <alignment> infile.apk outfile.apk
```

**alignment** 是一个整数，用于定义字节对齐边界。此值必须始终为 4（可提供 32 位对齐），否则实际将不会执行任何操作。

标记：

- **-f**：覆盖现有的 outfile.zip
- **-v**：详细输出
- **-p**：outfile.zip 应对 infile.zip 中的所有共享对象文件使用相同的页面对齐方式
- **-c**：确认给定文件的对齐方式

### 3. sdk 平台工具

- **adb**：Android 调试桥 (adb) 是一种多功能的工具，您可以用它来管理模拟器实例或 Android 设备的状态。还可以使用它在设备上安装 APK。
- **logcat**：此工具可通过 adb 调用，用于查看应用和系统日志。
- **fastboot**：将平台或其他系统映像刷写到设备上。

#### .1. adb

```shell
adb connect device_ip  #连接到设备
adb disconnect device_ip  #断开连接到设备
adb devices -l  #查询设备
adb install path—to-apk  #安装应用
adb push local remote  #将文件复制到设备
adb pull remote local  #将设备复制文件
adb shell shell—command  #发送shell命令
adb kill-server  #停止adb服务器

adb shell pm list packages #--显示系统应用包名
adb shell pm list packages #-3--显示第三方应用包名
#查询手机cpu和内存信息
adb shell cat /proc/cpuinfo
adb shell cat /proc/meminfo
adb shell ps
adb shell kill pid
```

#### .2. logcat

- 用于转储系统消息日志，包括设备抛出错误时的堆栈轨迹，以及从我们的应用中使用 Log 类写入的消息。

| 命令选项 | 说明                                                         |
| :------- | :----------------------------------------------------------- |
| -b       | 加载可供查看的备用日志缓冲区，例如 events 或 radio。         |
| -c       | 清除（清空）所选的缓冲区并退出。                             |
| –regex   | 只输出日志消息与正则表达式匹配的行。                         |
| -m       | 输出特定行后退出。                                           |
| –print   | 与 --regex 和 --max-count 配对，使内容绕过正则表达式过滤器。 |
| -d       | 将日志转储到屏幕并退出。                                     |
| -f       | 将日志消息输出写入 。                                        |
| -g       | 输出指定日志缓冲区的大小并退出。                             |
| -n       | 设置轮替日志的数量上限。                                     |
| -r       | 每输出特定字节时轮替日志文件。                               |
| -s       | 相当于过滤器表达式 ‘*:S’。                                   |
| -v       | 设置日志消息的输出格式。                                     |
| -D       | 输出各个日志缓冲区之间的分隔线。                             |
| –pid     | 仅输出来自给定 PID 的日志。                                  |

```shell
adb logcat -v time *:W |grep pid > /data/data/log.txt
adb pull /data/data/log.txt d:/log/
adb shell "logcat -v time *:W |grep pid " > d:/log/log.txt
adb logcat -c
logcat -v time -n 10 -r 102400 -f /sdcard/logcat.txt（一直发送）
logcat -v time -n 10 -r 102400 -f /sdcard/logcat.txt &（发送一次）

adb shell monkey -p com.xyy.vwill -s 100 10000--momkey测试  得到1个小时,设置次数可能在百万以上
```

#### .3. fastboot

- 引导加载模式下的刷写工具。   --todo？ 学习一下如何刷机

```
adb reboot bootloader   #使设备进入faskboot模式
fastboot flasing unlock  #解锁引导加载程序
fastboot flashing lock #锁定引导加载程序
fastboot flashall -w  #刷写全部镜像
```



### 4. 模拟器工具avd

- **emulator**：一种基于 QEMU 的设备模拟工具，可用于在实际的 Android 运行时环境中调试和测试应用。可以直接在cmd 窗口启动模拟器，不需要通过android 工具
- **mksdcard**： 可帮助我们创建可与模拟器一起使用的磁盘映像，以模拟存在外部存储卡（例如 SD 卡）的情形。

#### .1. emulator

```shell
emulator -list-avds  # 查看avd 名称列表
emulator -help
emulator -avd avd_name  # 启动模拟器
emulator -help-netspeed  #
emulator -help-environment  #列出模拟器环境变量
```

| 命令选项                   | 说明                                                         |
| :------------------------- | :----------------------------------------------------------- |
| -no-snapshot-load          | 执行冷启动，并在退出时保存模拟器状态。                       |
| -no-snapshot-save          | 执行快速启动，但在退出时不保存模拟器状态。                   |
| -no-snapshot               | 彻底停用快速启动功能。                                       |
| -camera-back -camera-front | 设置后置或前置相机的模拟模式。 emulated：模拟器在软件中模拟相机。 webcamn：模拟器使用连接到开发计算机的摄像头，由数字指定，例如 webcam0。 none：在虚拟设备中停用相机。 |
| -webcam-list               | 列出开发计算机上可用于模拟的摄像头。                         |
| -memory                    | 指定物理 RAM 大小，范围为从 128 MB 到 4096 MB。              |
| -sdcard                    | 指定 SD 卡分区映像文件的文件名和路径。                       |
| -wipe-data                 | 删除用户数据并从初始数据文件中复制数据。                     |
| -debug                     | 启用或停用一个或多个标记的调试消息显示。                     |
| -logcat                    | 启用一个或多个标记的 logcat 消息显示，并将其写入终端窗口。   |
| -show-kernel               | 在终端窗口中显示内核调试消息。                               |
| -verbose                   | 将模拟器初始化消息输出到终端窗口。                           |
| -dns-server                | 使用指定的 DNS 服务器。                                      |
| -http-proxy                | 通过指定的 HTTP/HTTPS 代理进行所有 TCP 连接。                |
| -netdelay                  | 模拟设置网络延迟                                             |
| -netfast                   | 停用网络节流功能。                                           |
| -netspeed                  | 设置网络速度模拟。                                           |
| -port                      | 设置用于控制台和 adb 的 TCP 端口号。                         |
| -tcpdump                   | 捕获网络数据包并将其存储在文件中。                           |
| -accel                     | 配置模拟器虚拟机加速。                                       |
| -accel-check               | 检查是否已安装模拟器虚拟机加速所需的管理程序（HAXM 或 KVM）。 |
| -engine                    | 指定模拟器引擎： auto：自动选择引擎（默认值）。 classic：使用较旧的 QEMU 1 引擎。 qemu2：使用较新的 QEMU 2 引擎。 |
| -gpu                       | 选择 GPU 模拟模式。                                          |
| -version                   | 显示模拟器版本号。                                           |
| -no-boot-anim              | 在模拟器启动期间停用启动动画以加快启动速度。                 |
| -screen                    | 设置模拟触摸屏模式。 touch：模拟触摸屏（默认值）。 multi-touch：模拟多点触控屏幕。 no-touch：停用触摸屏和多点触控屏幕模拟。 |

| 命令选项           | 说明                                       |
| :----------------- | :----------------------------------------- |
| -bootchart         | 启用 bootchart，设有超时（以秒为单位）。   |
| -cache             | 指定缓存分区映像文件。                     |
| -cache-size        | 设置缓存分区大小（以 MB 为单位）。         |
| -data              | 设置用户数据分区映像文件。                 |
| -datadir           | 使用绝对路径指定数据目录。                 |
| -force-32bit       | 在 64 位平台上使用 32 位模拟器。           |
| -help-disk-images  | 获取有关磁盘映像的帮助。                   |
| -help-char-devices | 获取有关字符 device 规范的帮助。           |
| -help-sdk-images   | 获取与应用开发者相关的磁盘映像的帮助。     |
| -help-build-images | 获取与平台开发者相关的磁盘映像的帮助。     |
| -initdata          | 指定数据分区的初始版本。                   |
| -kernel            | 使用特定的模拟内核。                       |
| -noaudio           | 停用对此虚拟设备的音频支持。               |
| -nocache           | 启动没有缓存分区的模拟器。                 |
| -no-snapshot       | 禁止自动加载和保存操作。                   |
| -no-snapshot-load  | 阻止模拟器从快照存储加载 AVD 状态。        |
| -no-snapshot-save  | 阻止模拟器在退出时将 AVD 状态保存到快照。  |
| -no-window         | 停用模拟器上的图形窗口显示。               |
| -partition-size    | 指定系统数据分区大小（以 MB 为单位）。     |
| -prop              | 在启动时在模拟器中设置 Android 系统属性。  |
| -ramdisk           | 指定 ramdisk 启动映像。                    |
| -shell             | 在当前终端上创建根 shell 控制台。          |
| -sysdir            | 使用绝对路径指定系统目录。                 |
| -system            | 指定初始系统文件。                         |
| -writable-system   | 使用此选项在模拟会话期间创建可写系统映像。 |

#### .2. mksdcard

- 使用 mksdcard 工具创建 FAT32 磁盘映像，然后将该映像加载到运行不同 Android 虚拟设备 (AVD) 的模拟器中，以模拟多个设备中存在相同 SD 卡的情形。

```shell
mksdcard -l label size file

mksdcard -l mysdcard 1024M mysdcardfile.img
emulator avd adv_name -sdcard mysdcardfile.img
```

| 命令选项 | 说明                                                         |
| :------- | :----------------------------------------------------------- |
| -l       | 指定要创建的磁盘映像的卷标。                                 |
| size     | 一个整数，用于指定要创建的磁盘映像的大小。                   |
| file     | 指定要创建的磁盘映像的路径/文件名。此路径相对于当前的工作目录指定。 |

### 5. android 分析工具

- **dumpsys**：一种在 Android 设备上运行的工具，可提供有关系统服务的信息。
- **dmtracedump**：一种用于从跟踪日志文件生成图形化的调用堆栈图的工具。
- **systrace**：借助该工具收集和检查设备上在系统一级运行的所有进程的时间信息。

#### .1. dumpsys

- dumpsys 是一种在 Android 设备上运行的工具，可提供有关系统服务的信息。我们可以使用 Android 调试桥 (ADB) 从命令行调用 dumpsys，获取在连接的设备上运行的所有系统服务的诊断输出。

```shell
adb shell dumpsys [-t timeout] [--help | -l | --skip services | service [arguments] | -c | -h]
```

| 命令选项 | 说明                                                       |
| :------- | :--------------------------------------------------------- |
| -t       | 指定超时期限（秒）。                                       |
| –help    | 输出 dumpsys 工具的帮助文本。                              |
| -l       | 输出可与 dumpsys 配合使用的系统服务的完整列表。            |
| –skip    | 指定不希望包含在输出中的服务。                             |
| service  | 指定希望输出的服务。                                       |
| -c       | 指定某些服务时，附加此选项能以计算机可读的格式输出数据。   |
| -h       | 对于某些服务，附加此选项可查看该服务的帮助文本和其他选项。 |

#### .2. dmtracedump

- dmtracedump 是一种用于从跟踪日志文件生成图形化的调用堆栈图的工具。

```java
SimpleDateFormat dateFormat =new SimpleDateFormat("dd_MM_yyyy_hh_mm_ss", Locale.getDefault());
String logDate = dateFormat.format(new Date());
Debug.startMethodTracing(logDate);
params[i].run();// 追踪此方法
Debug.stopMethodTracing();

//sd卡(内置/外置)Android/data/com.android.mms/files 下有相应的trace文件
sudo apt-get install graphviz

./dmtracedump -g aabc.png 23_02_2019_04_36_49.trace  // 

```



#### .3. systrace

- systrace 命令会调用 Systrace 工具，我们可以借助该工具收集和检查设备上在系统一级运行的所有进程的时间信息。

```shell
python systrace.py [options] [categories]
```

| 命令选项   | 说明                                                         |
| :--------- | :----------------------------------------------------------- |
| -o         | 将 HTML 跟踪报告写入指定的文件。                             |
| –time      | 设置跟踪设备活动时间。                                       |
| –buf-size  | 设置跟踪缓冲区大小。                                         |
| –ktrace    | 跟踪逗号分隔列表中指定的特定内核函数的活动。                 |
| –app       | 启用对应用的跟踪，指定为包含进程名称的逗号分隔列表。         |
| –from-file | 根据文件创建交互式 HTML 报告，而不是运行实时跟踪。           |
| –serial    | 在已连接的特定设备上进行跟踪。                               |
| categories | 包含指定的系统进程的跟踪信息，如 gfx 表示用于渲染图形的系统进程。 |

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/androidstudio_%E5%B7%A5%E5%85%B7/  

