# BatCMD


#### 1. 命令介绍

- ##### echo， dir,  cd,  mkdir,  rd

- rd /s/q d:temp:  \#删除 d:temp 文件夹及其子文件夹和文件，/q安静模式

- del /q/a/f/s d:temp*.*:  #删除 d:temp 及子文件夹里面的所有文件，包括隐藏、只读、系统文件，不包括子目录

- cls: 清空屏幕

- type: #显示文件内容，  type 1.txt

- copy c:test.txt d:test.bak  #复制 c:test.txt 文件到 d: ，并重命名为 test.bak

- pause:  暂停命令

- find "abc" c:test.txt   #在 c:test.txt 文件里查找含 abc 字符串的行

- tree: #显示目录结构

- &：  #顺序执行多条命令，而不管命令是否执行成功

- &&: #顺序执行多条命令，当碰到执行出错的命令后将不执行后面的命令

- || : #顺序执行多条命令，当碰到执行正确的命令后将不执行后面的命令

- | : 管道命令

- date <temp.txt;   copy c:test.txt f: >nul

- start: 批处理中调用外部程序的命令，否则等外部程序完成后才继续执行剩下的指令

- call  :#批处理中调用另外一个批处理的命令，否则剩下的批处理指令将不会被执行

- **exit** : #退出CMD.EXE程序或当前批处理脚本

- @:  表示不显示@后面的命令

- rem:  注释命令，在C语言中相当与/*--------*/,它并不会被执行，只是起一个注释的作用，便于别人阅读和你自己日后修改。

#### 2. 代码片段

- 条件结构

```cmd
#if-else
IF EXIST filename. (
	del filename.
) ELSE (
	echo filename. missing.
)
#for  打印C盘根目录下的目录名
@echo off
for /d %%i in (c:/*) do (
  echo %%i
)
pause
```

- 根据输入选项操作

```cmd
@echo off
set /p var="Please input the number(1,2,3):"
if %var% == 1 (
  echo "the number equal to 1"
) else if %var% == 2 (
  echo "the number equal to 2"
) else if %var% == 3 (
  echo "the number equal to 3"
) else (
  echo "input wrong number,exit program."
)
pause
```

- 文件和目录相关操作

```cmd
@echo off
rem "About operate directory&file bat script"
title Test bat
set CURRENTDIR=D:\worktset TEMPDIR=%CURRENTDIR%\temp
set TEMPFILE=%TEMPDIR%\temp.txt
if not exist %TEMPDIR% (
  echo "Create temp directory"
  mkdir %TEMPDIR%
) else (
  echo The directory of %TEMPDIR% existed,recreate directory
)
if not exist %TEMPFILE% (
  echo Create temp file
  type nul > %TEMPFILE%
) else (
  echo 
  echo "=========%DATE% %TIME%================" >> %TEMPFILE%
)
echo Happy New Year! >> %TEMPFILE%
echo Congratulate to everyone >> %TEMPFILE%
rem copy file and directory
set TEMPDIR2=%CURRENTDIR%\temp2
md %TEMPDIR2%
xcopy /s /y %TEMPDIR% %TEMPDIR2%
type %TEMPDIR2%\temp.txt
pause
```

- 将指定目录下文件输出移动到指定目录下：

```cmd
@echo off
rem 如果路径中包含空格，变量值需带双引号rem WORK_DIR表示要操作的文件夹，DEST_DIR表示文件要保存的目标文件夹
SET WORK_DIR="c:\Program Files"
SET DEST_DIR="D:\temp"
if not exist %DEST_DIR% (
  mkdir %DEST_DIR%
)
for /f "delims=" %%i in ('dir /b /s /o:n /a:a %WORK_DIR%') do (
  echo %%i
  copy "%%i" %DEST_DIR%
) 
pause
```

- 使用anaconda环境，python

```cmd
@echo off
call C:\Users\dell\Anaconda3\Scripts\activate base
C:
cd C:\project\Face_recognition\FaceServer
start python recognize_faces_in_pictures.py
start python visualise.py 
cd C:\project\computervision\backgroundServ
start python picturecharInfer.py
exit
```

#### 3. 学习链接

- https://www.cnblogs.com/linyfeng/p/8072002.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/batcmd/  

