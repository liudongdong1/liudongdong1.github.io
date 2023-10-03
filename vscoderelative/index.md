# vscodeRelative


### 1. TODO Tree

> 某块代码需要修改，或者某块代码需要以后进一步完善，如果能够给它做一个标记，那么后续定位到对应位置是一件非常轻松高效的事情。

### 2. vscode-icons

> 不仅能够给文件夹、文件添加上舒适的图标，而且可以自动检测项目，根据项目不同功能配上不同图标，例如，git、Markdown、配置项、工具类等等。

### 3. **Better Comments**

> 可以根据告警、查询、TODO、高亮等标记对注释进行不同的展示。此外，还可以对注释掉的代码进行样式设置。

### 4. **Bracket Pair Colorizer**

> 可以给`()`、`[]`、`{}`这些常用括号显示不同颜色，当点击对应括号时能够用线段直接链接到一起，让层次结构一目了然。

### 5. **Better Align**

> 主要用于代码的**上下对齐**。
>
> 它能够用冒号（：）、赋值（=，+=，-=，*=，/=）和箭头（=>）对齐代码。**Ctrl+Shift+p输入“Align”确认即可。**

### 6. settings Sync

> 不同的电脑上都会使用VSCode, VSCode中安装有许多插件， settings Sync让这些插件在不同的电脑之间同步。
>
> F1输入命令sync
>
> 选择Update/Upload Settings，如果是新创建的话全产生gist id，如果是想下载的话，选择Downlad Settings.  
>
>  **Shift + Alt + U**          或者 Shift+Alt+D  #upload or download settings

### 7. **Remote - SSH**

> 通过修改settings中的设置，实现ssh远程无密钥登录。

```xml
Host iot
    HostName 192.168.2.3
    User iot
    Port 22
    IdentityFile C:/Users/dell/.ssh/id_rsa
```

> 这个是之前的版本，先点击+， 然后输入ssh连接命令，会自动生成config文件，然后左边会自动跟新相应的hostname，否则不跟新； 公钥写入服务器后，需要重新启动ssh服务

```
Host remotetank
    HostName 172.26.200.203
    User iot
    Port 65522
#    IdentityFile C:/Users/liudongdong/.ssh/id_rsa  这个行不需要设置
```

> easyconnect 软件当打开vpn的时候，没有生成虚拟ip，只能通过IE代理，不能建立ssh虚拟隧道连接

### 8. **[Remote - Containers](https://link.zhihu.com/?target=https%3A//marketplace.visualstudio.com/items%3FitemName%3Dms-vscode-remote.remote-containers)**  docker

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220511201615991.png)

- register: 用于连接dockerhub
- images:  支持 run, run interative, inspect, pull, push, tag, copy full tag, remove 操作
- containters: 
  - view logs: 查看log信息
  - attach shell: 进入容器 /bin/bash
  - attach visual studio code: 类似vscode 打开，编辑代码
  - inspect：检查
  - stop， restart

### 9. Beautify

> 使用：打开要格式化的文件 —> F1 —> Beautify file —> 选择你要格式化的代码类型
>
> 格式化对齐快捷键：
> Windows： Ctrl + K + F
> Windows：Shift + Alt + F
> Mac： Shift + Option + F
> Ubuntu： Ctrl + Shift + I

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610221552174.png)

### 10. vscode-pdf

### 11. [al-code-outline](https://github.com/anzwdev/al-code-outline)

> 类似函数大纲

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610221905656.png)

### 12. jupyter

```json
 "jupyter.experiments.optOutFrom": [ "NativeNotebookEditor"],   //jupyter 有json和notebook俩种查看方式；
```

- Ctrl-Enter : 运行本单元
- Alt-Enter : 运行本单元，在其下插入新单元

### 13. leetcode

```xml
    "leetcode.endpoint": "leetcode-cn",
    "leetcode.defaultLanguage": "java",
    "leetcode.workspaceFolder": "E:\\找工作\\leetcode\\leetcode\\"
```



### 快捷键

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210121233910973.png)

- 要操作光标所在`文件`中的所有代码块：
  - 折叠所有 `Ctrl+K+0`
  - 展开所有 `Ctrl+K+J`
- 仅仅操作光标所处`代码块`内的代码：
  - 折叠 `Ctrl+Shift+[`
  - 展开 `Ctrl+Shift+]`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210610220957222.png)

### 14. cpp 环境配置

- 安装 C/C++ 扩展插件工具
- [下载 MinGW](https://sourceforge.net/projects/mingw-w64/files/)，进入网站后不要点击 "Download Lasted Version"，往下滑，找到最新版的 **"x86_64-posix-seh"**
-  配置环境变量： C:\Program Files\mingw64\bin
- 按下 win + R，输入 cmd，回车键之后输入 g++;
- 进入调试界面添加配置环境，选择 C++(GDB/LLDB)，再选择 g++.exe，之后会自动生成 launch.json 配置文件,编辑 launch.json 配置文件，主要修改 "externalConsole": true, 返回.cpp 文件，按 F5 进行调试


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/vscoderelative/  

