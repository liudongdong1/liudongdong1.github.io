# EnvInstall


首先必须安装Java  JDK  （Java JDK下载、安装和环境变量配置，传送阵：[点击开始传送](http://blog.csdn.net/siwuxie095/article/details/53386227)）

去Android Studio的中文社区（官网）下载最新的Android Studio，传送阵：[点击开始传送](http://www.android-studio.org/)

![](https://img-blog.csdn.net/20161202101347447)下载完毕后，开始安装：

（1）首先是欢迎界面：

![](https://img-blog.csdn.net/20161202101551969)

（2）选择需要安装的组件，Android Studio主程序默认已勾选，

Android SDK这里也要勾选（假如你已经单独装了SDK，就不需要了）

Android Virtual Device安卓虚拟设备，就是在电脑上虚拟出安卓手机的环境，让你可以直接在电脑上运行开发出的APP

这里没有勾选，也建议不要勾选，测试APP的话，直接在真机（一部安卓手机）上测试更好，因为官方出的这个安卓虚拟设备，在电脑上运行很慢，即便你没有一部安卓手机，也可以选择其他的安卓模拟器，运行速度都比这个快

![](https://img-blog.csdn.net/20161202101720078)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（3）选择  I Agree  

![](https://img-blog.csdn.net/20161202102714412)

（4）这是Android Studio  和  SDK  的默认安装路径

![img](https://img-blog.csdn.net/20161202102916453)

（5）这里分别改成：D:\Android\Android Studio 和 D:\Android\SDK

![](https://img-blog.csdn.net/20161202103026304)

（6）直接点击 Install 进行安装 （这里没有勾选 Do not create shortcuts,  这是问你是否要桌面快捷方式）

![](https://img-blog.csdn.net/20161202103157223)

（7）安装完成，直接 Next

![](https://img-blog.csdn.net/20161202103422807)

（8）既然安装完成就直接启动Android Studio吧

![](https://img-blog.csdn.net/20161202103843950)

（9）我之前并没有使用过Android Studio，所以选择这一项

![](https://img-blog.csdn.net/20161202104000169)

（10）进入此界面，开始载入Android Studio主程序

![](https://img-blog.csdn.net/20161202104023298)

（11）出现了Unable to access Android SDK add-on list

   这里选择了Setup Proxy  如果选择Cancel  见（14）

![](https://img-blog.csdn.net/20161202104116393)

（12）具体配置见图，Host name:  mirrors.neusoft.edu.cn

![](https://img-blog.csdn.net/20161202104244014)

（13）关于mirrors.neusoft.edu.cn，实际上是大连东软信息学院的一个开源镜像网站

用过Eclipse的应该知道，Eclipse的下载页面的镜像网站之一就是大连东软信息学院

![](https://img-blog.csdn.net/20161202104614015)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)![](https://img-blog.csdn.net/20161202104636703)

Eclipse下载页面：

![](https://img-blog.csdn.net/20161202105008795)

（14）如果你选择了Cancel

![](https://img-blog.csdn.net/20161202105225033)

需要到Android Studio的bin目录下找到 idea.properties 这个文件

这里的路径是：D:\Android\Android Studio\bin

使用记事本或其他编辑器，打开这个文件，更改 disable.android.first.run  的值等于true,

即disable.android.first.run=true  如果没有则直接添加

![](https://img-blog.csdn.net/20161202105746051)

（15）直接 Next

![](https://img-blog.csdn.net/20161202110329490)

（16）这里选择 Custom  自定义

![img](https://img-blog.csdn.net/20161202110412381)

（17）这里选择Darcula主题，护眼

![](https://img-blog.csdn.net/20161202110502054)

（18）更改Android SDK的路径为上面已经设置过的路径：D:\Android\SDK

![](https://img-blog.csdn.net/20161202110557538)

![](https://img-blog.csdn.net/20161202110747362)

![](https://img-blog.csdn.net/20161202110816113)

（19）点击 Finish

![](https://img-blog.csdn.net/20161202110855395)

（20）点击 Finish

![](https://img-blog.csdn.net/20161202110944381)

（21）安装完成，安装目录一览：

![](https://img-blog.csdn.net/20161202111145432)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

![](https://img-blog.csdn.net/20161202111232667)

![](https://img-blog.csdn.net/20161202111210713)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

5、下面开始配置Android Studio相关:

（1）点击Configure

![](https://img-blog.csdn.net/20161202111047869)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（2）选择 Settings

![](https://img-blog.csdn.net/20161202111443263)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（3）配置界面一览：

![](https://img-blog.csdn.net/20161202111521419)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（4）先修改一下字体大小，因为默认字体实在太小了

点击Save As，然后才能开始修改字体大小 Size， 改成16就好了 （至于字体样式看个人习惯，我这里直接默认）

![](https://img-blog.csdn.net/20161202111601405)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（5）选择显示行号

![](https://img-blog.csdn.net/20161202111852830)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

6、开始第一个Hello World吧

（1）我的默认存放路径改成了：E:\AndroidStudioProjects

![](https://img-blog.csdn.net/20161202112128208)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（2）Phone and Tablet  手机和平板电脑   Wear  可穿戴式设备   TV  就是电视了  看你开发什么上面的APP，Minimum SDK最好选择 Android 5.0  这是一个截止目前（2016/12/2）承上启下的SDK，开发出的APP会更好的兼容不同版本的Android系统

![](https://img-blog.csdn.net/20161202112414930)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（3）选择一个Empty Activity  空活动

![](https://img-blog.csdn.net/20161202112951283)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（4）直接默认吧

![](https://img-blog.csdn.net/20161202113056878)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（5）勾选如图

![](https://img-blog.csdn.net/20161202113134972)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（6）随便看看吧

![](https://img-blog.csdn.net/20161202113212160)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

（7）完毕

![](https://img-blog.csdn.net/20161202113248457)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

7、SDK的环境变量配置：

第一步：添加 ANDROID_HOME

新建系统变量 ANDROID_HOME

变量名：ANDROID_HOME  变量值：D:\Android\SDK

![](https://img-blog.csdn.net/20161202113607276)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

第二步：添加Path变量

此变量已存在，直接编辑即可

变量值：%ANDROID_HOME%\tools;%ANDROID_HOME%\platform-tools

（注意：win10下要分行编辑，且末尾没有分号）

![](https://img-blog.csdn.net/20161202113845982)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

8、添加完成，确认保存。

最后验证一下：打开cmd命令行窗口：分别输入   adb     android  

两个命令进行验证，都没有出错，则配置成功。



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/envinstall/  

