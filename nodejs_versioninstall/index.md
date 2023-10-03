# nodejs_versionInstall


> Node.js是基于Chrome JavaScript运行时建立的一个平台，实际上它是对Google Chrome V8引擎进行了封装，它主要用于创建快速的、可扩展的网络应用。Node.js采用**事件驱动**和**非阻塞I/O模型**，使其变得轻量和高效，非常适合构建运行在分布式设备的数据密集型的实时应用。运行于浏览器的JavaScript，浏览器就是JavaScript代码的解析器，而**Node.js则是服务器端JavaScript的代码解析器**，存在于服务器端的JavaScript代码由Node.js来解析和运行。
>
> `npm，nodejs，cnpm 关系`：npm 是 Node.js 的包管理工具，所谓包管理工具可以理解为大佬们将一些常用的功能写成包并发布到 npm 市场上，然后别人通过 npm 直接安装即可使用（类似手机应用 app）。而因为 npm 在国内有一定限制，所以就需要用淘宝的镜像 cnpm，从而提高我们 npm 的下载安装速度（类似手机网络和 WIFI 下载手机应用 app）

### 0. Nodejs 介绍

![Node-README-01.png](https://gitee.com/github-25970295/picture2023/raw/master/6629b0d130ca43de9dc6fdbc7eeb4960tplv-k3u1fbpfcp-zoom-in-crop-mark4536000.png)

### 1. Node 版本控制工具

#### 1. nvm 工具

```shell
nvm ls-remote --lts  #查看所有可以安装的LTS版本
nvm use 14.0.0
npm install -g yarn
nvm use 12.0.1
npm install -g yarn
nvm uninstall 12.0.1
nvm list [available] #显示已安装的列表。可选参数available，显示可安装的所有版本。list可简化为ls。
NVM_NODEJS_ORG_MIRROR=https://npm.taobao.org/mirrors/node nvm install 6.10.2 #使用淘宝源安装

nvm alias default v6.6.0  #设定默认的node版本 nvm alias default <版本号>
nvm current #查看当前版本显示

#删除时权限问题
sudo chown -R $(whoami) "$NVM_DIR/versions/node/v6.6.0"
sudo chmod -R u+w "$NVM_DIR/versions/node/v6.6.0"      
```

- 问题记录：报错exit status 1 或 exit status 145

> 目录得全英文，不能有空格或者中文字符，需要管理员权限.
>
> 之前下载的nodejs得删除掉。

#### 2. n 工具

##### .1. 安装

```shell
# 使用 npm / yarn
npm i -g n
yarn global add n

# 使用 brew
brew install n
```

##### .2. 版本查看

```shell
# 查看 n 版本
n --version/-V

# 查看 node 本地当前使用版本
node --version/-v

# 查看 node 远程版本
n lsr/ls-remote [--all] // 默认20个，--all展示所有

# 查看 n 管理的 node 版本
n [ls/list/--all]

#查看 Node.js 版本安装路径
n which/bin <version>
```

##### .3. node 安装

```shell
# 安装指定版本
n [install/i] <version>
# 安装稳定版本
n lts/stable
# 安装最新版本
n latest/current
# 安装文件中对应 node 版本 [.n-node-version, .node-version, .nvmrc, or package.json]
n auto
# 安装 package.json 对应 node 版本
n engine
# 通过发布流的代码名 例如[ boron, carbon]
n boron/carbon


# 删除当前版本
n uninstall
# 删除指定版本
n rm/- <version>
# 删除除当前版本之外的所有版本
n prune


# 使用指定 node 版本
n run/use/as <version> [args...]
# 先下载节点和npm，使用修改过的PATH执行命令
n exec <vers> <cmd> [args...]
```

### 3. NPM & CNPM命令

- 淘宝CNPM 使用：https://npmmirror.com/
- npm 命令：https://bbs.huaweicloud.com/blogs/343692
- [npm常用命令与操作篇](https://segmentfault.com/a/1190000022205162)：https://segmentfault.com/a/1190000022205162

```shell
# 查看 npm 的版本 
$ npm -v  //6.4.0 << 安装成功会返回版本号

# 查看各个命令的简单用法
$ npm -l 
 
# 查看 npm 命令列表
$ npm help

# 查看 npm 的配置
$ npm config list -l
#创建模块
$ npm init

$ npm search <搜索词> [-g]
 
 #当前项目安装的所有模块
$npm list

#列出全局安装的模块 带上[--depth 0] 不深入到包的支点 更简洁
$ npm list -g --depth 0


12.可以目录的形式来展现当前安装的所有node包
npm list parseable=true

13.查看包的依赖关系
npm view moudleName dependencies

14.查看包的源文件地址
npm view moduleName repository.url

15.查看包所依赖的Node的版本
npm view moduleName engines

9.查看node模块的package.json文件夹
npm view moduleNames

# 读取package.json里面的配置单安装  
$ npm install 
//可简写成 npm i

# 默认安装指定模块的最新(@latest)版本
$ npm install [<@scope>/]<name> 
//eg:npm install gulp

# 安装指定模块的指定版本
$ npm install [<@scope>/]<name>@<version>
//eg: npm install gulp@3.9.1

# 安装指定指定版本范围内的模块
$ npm install [<@scope>/]<name>@<version range>
//eg: npm install vue@">=1.0.28 < 2.0.0"

# 安装指定模块的指定标签 默认值为(@latest)
$ npm install [<@scope>/]<name>@<tag>
//eg:npm install sax@0.1.1

# 通过Github代码库地址安装
$ npm install <tarball url>
//eg:npm install git://github.com/package/path.git

#卸载当前项目或全局模块 
$ npm uninstall <name> [-g] 

#升级当前项目或全局的指定模块
$ npm update <name> [-g] 
//eg: npm update express 
      npm update express -g
      
$ npm run的参数。
# 如果不加任何参数，直接运行，会列出package.json里面所有可以执行的脚本命令
---package.json文件---
"scripts": {
  "test": "mocha test/"
}

-------终端-------
$ npm run test -- anothertest.js
# 等同于直接执行
$ mocha test/ anothertest.js
```



```shell
npm install -g cnpm --registry = https://registry.npmmirror.com

cnpm install vue-router@4
```



### 4.Resource

- https://www.jianshu.com/p/0cfeed299f2a


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/nodejs_versioninstall/  

