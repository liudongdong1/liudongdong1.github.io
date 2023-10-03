# bat


### 1. cmd关闭占用进程

> IDEA中启动Tomcat报错，Error running Tomcat7.0.52: Address localhost:1099 is already in use 或者是 java.rmi.server.ExportException: Port already in use: 1099 ，表示1099端口被其他进程占用了。
> **解决方法：**
>
> 1. win+R，运行，输入cmd，进入命令提示符
> 2. 输入**netstat -aon | findstr 1099**，找到占用1099端口的进程ID：**PID**
> 3. 输入**taskkill -f -pid PID**     或者 **tasklist | findstr PID**查看进程名，然后任务管理器->显示所有用户的进程->结束进程。  taskkill /im "pycharm.exe"
> 4. 重启Tomcat

### 2. bat 命令

> 1、REM 和 ::             给程序加上注释
> 2、ECHO 和 @           echo会显示运行的内容，加@则不会在运行框中显示运行内容（会继续运行，只是不会显示）。
> 3、PAUSE               暂停
> 4、ERRORLEVEL         命令运行结束，单独一行输入echo %errorlevel%会显示运行是否成功（成功0，失败1）
> 5、TITLE                设置cmd窗口的标题，格式为title name#
> 6、COLOR               改变窗口的颜色，格式为color 02
> 7、mode 配置系统设备      配置系统设备，比如mode con cols=100 lines=40，意思为设置窗口的宽和高
> 8、GOTO 和 :             跳转，用：XX构筑一个标记，用goto XX跳转到XX标记处
> 9、FIND                 在文件中搜索字符串
> 10、`START   批处理调用外部程序的命令（不理会外部运行状况，等到外部命令运行后才能继续运行），格式为start xxx（路径名）`
> 11、assoc 和 ftype         文件关联（目前没发现有什么用）
> 12、pushd 和 popd         切换当前目录（用于不确定文件夹的情况，dos编程常用）
> 13、`CALL                在批处理的过程中调用另一个批处理，当另一个批处理执行完了后回调自身`
> 14、shift                 更改批处理文件中可替换参数的位置
> 15、IF                   判断，回头详细研究
> 16、setlocal 与            变量延迟
> 17、ATTRIB              显示或更改文件属性

### 3. 自动化运行ipython

```cmd
REM ip.bat
call D:\Commonsoftware\Anaconda\Scripts\activate.bat
call activate myenv
call jupyter qtconsole
```

### 4. 自动化运行 Jupyter Notebook

```cmd
call D:\Commonsoftware\Anaconda\Scripts\activate.bat 
call activate myenv
start chrome http://localhost:8888/tree
call jupyter notebook
pause
```

### 5. 自动化运行模型

```cmd
@echo off
setlocal enabledelayedexpansion
set cyc=100
set ti=3000
call D:\anaconda3\Scripts\activate.bat
call activate myenv
python model_preprocess.py
for /l %% i in (1,1,!cyc!) do (
   start cmd /c "cd ./AppPath && model.exe"
   choice /t !ti! /d y /n >nul
   taskkill /f /im model.exe
)
python model_postprocess.py
endlocal
taskkill /f /im cmd.exe
```

### 6. 照片批量命名

```cmd
@echo off
set a=-45
setlocal EnableDelayedExpansion
for %%n in (*.jpg) do (
set /A a+=45°
ren “%%n” “!a!.jpg”
)
```

### 7. 批量提取文件名

```cmd
DIR *.* /B >施工图列表.TXT
```

### 8. start 延时启动

```cmd
 #start + 空格 + 引号 +空格+ 程序目录
```

![image-20210309000356985](https://gitee.com/github-25970295/blogimgv2022/raw/master/raw/master/img/image-20210309000356985.png)

- [APK 签名](https://github1s.com/dailey007/WindowsBatUtils/blob/HEAD/ShowApkMainActivity/README.md)

- [windows 通过计划任务定时执行bat文件](https://blog.csdn.net/jlq_diligence/article/details/89459471)
- dir_count

```cmd
@echo off
setlocal   #中间的程序对于系统变量的改变只在程序内起作用，不会影响整个系统级别。
TITLE dir_count
set reg_dir=%1
cd %reg_dir%
attrib.exe /s ./* | find /v "File not found - " | find /c /v ""
endlocal
exit /b
```

- job_relative

```cmd
@echo off
TITLE is_running
@REM echo Checking %1
tasklist /FI "IMAGENAME eq %1" 2>NUL | find /I /N "%1">NUL
if "%ERRORLEVEL%"=="0" ( echo YES ) else ( echo NO )

@REM tasklist /FI "windowtitle eq %1" 2>NUL | find /I /N "%1">NUL
@REM if "%ERRORLEVEL%"=="0" ( echo YES ) else ( echo NO )
exit /b
```

- sleep_ping

```cmd
@echo off
setlocal
TITLE sleep_ping
set /a ns_p1=%1+1
@REM echo %time%
ping 127.0.0.1 -n %ns_p1% > nul
@REM echo %time%
endlocal
exit /b
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/bat/  

