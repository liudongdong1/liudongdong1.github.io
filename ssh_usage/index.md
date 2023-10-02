# SSH_usage


>SSH（Secure Shell 的缩写）是一种网络协议，用于`加密两台计算机之间的通信`，并且支持各种身份验证机制。还能对操作者进行`认证（authentication）和授权（authorization）`。明文的网络协议可以套用在它里面，从而实现加密。
>
>SSH 软件分成两个部分：向服务器发出请求的部分，称为`客户端（client）`，OpenSSH 的实现为 ssh；接收客户端发出的请求的部分，称为`服务器（server）`，OpenSSH 的实现为 sshd。

#### 1. Client

##### .1. 链接登录

>OpenSSH 的客户端是二进制程序 ssh。它在 Linux/Unix 系统的位置是`/usr/local/bin/ssh`，Windows 系统的位置是`/Program Files/OpenSSH/bin/ssh.exe`

```shell
#查看版本
ssh -V
#远程登陆
ssh user@hostname  #或者： ssh -l username host
ssh -p 8821 foo.com

#查看链接服务器指纹
ssh-keygen -l -f /etc/ssh/ssh_host_ecdsa_key.pub
#删除 服务器指纹
ssh-keygen -R hostname
```

>所谓“服务器指纹”，指的是 SSH 服务器公钥的哈希值。每台 SSH 服务器都有唯一一对密钥，用于跟客户端通信，其中公钥的哈希值就可以用来识别服务器。`ssh 会将本机连接过的所有服务器公钥的指纹`，都储存在本机的`~/.ssh/known_hosts`文件中。每次连接服务器时，通过该文件判断是否为陌生主机（陌生公钥）

##### .2. 远程命令

```shell
# 登录成功后，会立刻执行command 命令
ssh username@hostname command
ssh foo@server.example.com cat /etc/hosts

#互动式 shell
ssh -t server.example.com emacs
```

##### .3. 配置文件

###### 1. 文件位置

SSH 客户端的`全局配置文件`是`/etc/ssh/ssh_config`，`用户个人`的配置文件在`~/.ssh/config`，优先级高于全局配置文件。除了配置文件，`~/.ssh`目录还有一些用户个人的密钥文件和其他文件。

- `~/.ssh/id_ecdsa`：用户的 ECDSA 私钥。
- `~/.ssh/id_ecdsa.pub`：用户的 ECDSA 公钥。
- `~/.ssh/id_rsa`：用于 SSH 协议版本2 的 RSA 私钥。
- `~/.ssh/id_rsa.pub`：用于SSH 协议版本2 的 RSA 公钥。
- `~/.ssh/identity`：用于 SSH 协议版本1 的 RSA 私钥。
- `~/.ssh/identity.pub`：用于 SSH 协议版本1 的 RSA 公钥。
- `~/.ssh/known_hosts`：包含 SSH 服务器的公钥指纹。

###### 2. 主机设置

>用户个人的配置文件`~/.ssh/config`，可以按照不同服务器，列出各自的连接参数，从而不必每一次登录都输入重复的参数。

```yaml
Host *
     Port 2222

Host remoteserver
     HostName remote.example.com
     User neo
     Port 2112
```

>`Host *`表示对所有主机生效，后面的`Port 2222`表示所有主机的默认连接端口都是2222，这样就不用在登录时特别指定端口了。这里的缩进并不是必需的，只是为了视觉上，易于识别针对不同主机的设置。

###### 3. 配置命名

- `AddressFamily inet`：表示只使用 IPv4 协议。如果设为`inet6`，表示只使用 IPv6 协议。
- `BindAddress 192.168.10.235`：指定本机的 IP 地址（如果本机有多个 IP 地址）。
- `CheckHostIP yes`：检查 SSH 服务器的 IP 地址是否跟公钥数据库吻合。
- `Ciphers blowfish,3des`：指定加密算法。
- `Compression yes`：是否压缩传输信号。
- `ConnectionAttempts 10`：客户端进行连接时，最大的尝试次数。
- `ConnectTimeout 60`：客户端进行连接时，服务器在指定秒数内没有回复，则中断连接尝试。
- `DynamicForward 1080`：指定动态转发端口。
- `GlobalKnownHostsFile /users/smith/.ssh/my_global_hosts_file`：指定全局的公钥数据库文件的位置。
- `Host server.example.com`：指定连接的域名或 IP 地址，也可以是别名，支持通配符。`Host`命令后面的所有配置，都是针对该主机的，直到下一个`Host`命令为止。
- `HostKeyAlgorithms ssh-dss,ssh-rsa`：指定密钥算法，优先级从高到低排列。
- `HostName myserver.example.com`：在`Host`命令使用别名的情况下，`HostName`指定域名或 IP 地址。
- `IdentityFile keyfile`：指定私钥文件。
- `LocalForward 2001 localhost:143`：指定本地端口转发。
- `LogLevel QUIET`：指定日志详细程度。如果设为`QUIET`，将不输出大部分的警告和提示。
- `MACs hmac-sha1,hmac-md5`：指定数据校验算法。
- `NumberOfPasswordPrompts 2`：密码登录时，用户输错密码的最大尝试次数。
- `PasswordAuthentication no`：指定是否支持密码登录。不过，这里只是客户端禁止，真正的禁止需要在 SSH 服务器设置。
- `Port 2035`：指定客户端连接的 SSH 服务器端口。
- `PreferredAuthentications publickey,hostbased,password`：指定各种登录方法的优先级。
- `Protocol 2`：支持的 SSH 协议版本，多个版本之间使用逗号分隔。
- `PubKeyAuthentication yes`：是否支持密钥登录。这里只是客户端设置，还需要在 SSH 服务器进行相应设置。
- `RemoteForward 2001 server:143`：指定远程端口转发。
- `SendEnv COLOR`：SSH 客户端向服务器发送的环境变量名，多个环境变量之间使用空格分隔。环境变量的值从客户端当前环境中拷贝。
- `ServerAliveCountMax 3`：如果没有收到服务器的回应，客户端连续发送多少次`keepalive`信号，才断开连接。该项默认值为3。
- `ServerAliveInterval 300`：客户端建立连接后，如果在给定秒数内，没有收到服务器发来的消息，客户端向服务器发送`keepalive`消息。如果不希望客户端发送，这一项设为`0`。
- `StrictHostKeyChecking yes`：`yes`表示严格检查，服务器公钥为未知或发生变化，则拒绝连接。`no`表示如果服务器公钥未知，则加入客户端公钥数据库，如果公钥发生变化，不改变客户端公钥数据库，输出一条警告，依然允许连接继续进行。`ask`（默认值）表示询问用户是否继续进行。
- `TCPKeepAlive yes`：客户端是否定期向服务器发送`keepalive`信息。
- `User userName`：指定远程登录的账户名。
- `UserKnownHostsFile /users/smith/.ssh/my_local_hosts_file`：指定当前用户的`known_hosts`文件（服务器公钥指纹列表）的位置。
- `VerifyHostKeyDNS yes`：是否通过检查 SSH 服务器的 DNS 记录，确认公钥指纹是否与`known_hosts`文件保存的一致。

#### 2. 密钥登录

预备步骤，客户端通过`ssh-keygen`生成自己的公钥和私钥。OpenSSH 提供了一个工具程序`ssh-keygen`命令，用来生成密钥。

- 第一步，手动将客户端的`公钥放入远程服务器的指定位置`。

>用户公钥保存在服务器的`~/.ssh/authorized_keys`文件。你要以哪个用户的身份登录到服务器，密钥就必须保存在该用户主目录的`~/.ssh/authorized_keys`文件。只要把公钥添加到这个文件之中，就相当于公钥上传到服务器了。每个公钥占据一行。如果该文件不存在，可以手动创建。

- 第二步，`客户端`向服务器`发起 SSH 登录的请求`。

- 第三步，`服务器`收到用户 SSH 登录的请求，`发送一些随机数据给用户，要求用户证明自己的身份`。

- 第四步，`客户端`收到服务器发来的数据，`使用私钥对数据进行签名，然后再发还给服务器`。

- 第五步，`服务器`收到客户端发来的加密签名后，`使用对应的公钥解密`，然后跟原始数据比较。如果一致，就允许用户登录。

```shell
ssh-keygen
chmod 600 ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa.pub

cat ~/.ssh/id_rsa.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

#ssh-copy-id： 自动上传公钥
ssh-copy-id -i key_file user@host  #`-i`参数用来指定公钥文件，`user`是所要登录的账户名，`host`是服务器地址。如果`authorized_keys`文件已经存在，使用`ssh-copy-id`命令之前，务必保证`authorized_keys`文件的末尾是换行符（假设该文件已经存在）。
```

>`ssh-agent`命令就是为了解决这个问题而设计的，它让用户在整个 Bash 对话（session）之中，只在第一次使用 SSH 命令时输入密码，然后将私钥保存在内存中，后面都不需要再输入私钥的密码了。
>
>- 为了安全性，启用`密钥登录`之后，最好`关闭服务器的密码登录`。 sshd 的配置文件`/etc/ssh/sshd_config` PasswordAuthentication no

```shell
ssh-agent bash
eval `ssh-agent`
```

#### 3. 证书登录

>证书登录的主要优点有两个：（1）用户和服务器不用交换公钥，这更容易管理，也具有更好的可扩展性。（2）证书可以设置到期时间，而公钥没有到期时间。针对不同的情况，可以设置有效期很短的证书，进一步提高安全性。
>
>- （1）用户和服务器都将自己的公钥，发给 CA；（2）CA 使用服务器公钥，生成服务器证书，发给服务器；（3）CA 使用用户的公钥，生成用户证书，发给用户。

- 第一步，用户登录服务器时，SSH 自动将用户证书发给服务器。

- 第二步，服务器检查用户证书是否有效，以及是否由可信的 CA 颁发。证实以后，就可以信任用户。

- 第三步，SSH 自动将服务器证书发给用户。

- 第四步，用户检查服务器证书是否有效，以及是否由信任的 CA 颁发。证实以后，就可以信任服务器。

- 第五步，双方建立连接，服务器允许用户登录。

```bash
# 生成 CA 签发用户证书的密钥
$ ssh-keygen -t rsa -b 4096 -f ~/.ssh/user_ca -C user_ca
```

- `-t rsa`：指定密钥算法 RSA。
- `-b 4096`：指定密钥的位数是4096位。安全性要求不高的场合，这个值可以小一点，但是不应小于1024。
- `-f ~/.ssh/user_ca`：指定生成密钥的位置和文件名。
- `-C user_ca`：指定密钥的识别字符串，相当于注释，可以随意设置。

#### 4. Server

```shell
# 启动
$ sudo systemctl start sshd.service
# 停止
$ sudo systemctl stop sshd.service
# 重启
$ sudo systemctl restart sshd.service
# 让 sshd 在计算机下次启动时自动运行。
$ sudo systemctl enable sshd.service
$ sudo systemctl restart sshd.service  #配置文件修改以后，并不会自动生效，必须重新启动 sshd。
$ sshd -f /usr/local/ssh/my_config #sshd 启动时会自动读取默认的配置文件。如果希望使用其他的配置文件，可以用 sshd 命令的`-f`参数指定。
```

##### .1. 配置文件

sshd 的配置文件在`/etc/ssh`目录，主配置文件是`sshd_config`，此外还有一些安装时生成的密钥。

- `/etc/ssh/sshd_config`：配置文件
- `/etc/ssh/ssh_host_ecdsa_key`：ECDSA 私钥。
- `/etc/ssh/ssh_host_ecdsa_key.pub`：ECDSA 公钥。
- `/etc/ssh/ssh_host_key`：用于 SSH 1 协议版本的 RSA 私钥。
- `/etc/ssh/ssh_host_key.pub`：用于 SSH 1 协议版本的 RSA 公钥。
- `/etc/ssh/ssh_host_rsa_key`：用于 SSH 2 协议版本的 RSA 私钥。
- `/etc/ssh/ssh_host_rsa_key.pub`：用于 SSH 2 协议版本的 RSA 公钥。
- `/etc/pam.d/sshd`：PAM 配置文件。

`修改配置文件以后，可以使用下面的命令验证，配置文件是否有语法错误。`

```bash
$ sshd -t
```

`新的配置文件生效，必须重启 sshd。`

```bash
$ sudo systemctl restart sshd
```

##### .2. 配置命令

- **AcceptEnv**： `AcceptEnv`指定允许接受客户端通过`SendEnv`命令发来的哪些环境变量，即允许客户端设置服务器的环境变量清单，变量名之间使用空格分隔（`AcceptEnv PATH TERM`）。
- **AllowGroups**： `AllowGroups`指定允许登录的用户组（`AllowGroups groupName`，多个组之间用空格分隔。如果不使用该项，则允许所有用户组登录。
- **AllowUsers**：`AllowUsers`指定允许登录的用户，用户名之间使用空格分隔（`AllowUsers user1 user2`），也可以使用多行`AllowUsers`命令指定，用户名支持使用通配符。如果不使用该项，则允许所有用户登录。该项也可以使用`用户名@域名`的格式（比如`AllowUsers jones@example.com`）。
- **AllowTcpForwarding**：`AllowTcpForwarding`指定是否允许端口转发，默认值为`yes`（`AllowTcpForwarding yes`），`local`表示只允许本地端口转发，`remote`表示只允许远程端口转发。
- **AuthorizedKeysFile**：`AuthorizedKeysFile`指定储存用户公钥的目录，默认是用户主目录的`ssh/authorized_keys`目录（`AuthorizedKeysFile .ssh/authorized_keys`）。
- **Banner**：`Banner`指定用户登录后，sshd 向其展示的信息文件（`Banner /usr/local/etc/warning.txt`），默认不展示任何内容。
- **ChallengeResponseAuthentication**：`ChallengeResponseAuthentication`指定是否使用“键盘交互”身份验证方案，默认值为`yes`（`ChallengeResponseAuthentication yes`）。从理论上讲，“键盘交互”身份验证方案可以向用户询问多重问题，但是实践中，通常仅询问用户密码。如果要完全禁用基于密码的身份验证，请将`PasswordAuthentication`和`ChallengeResponseAuthentication`都设置为`no`。
- **Ciphers**：`Ciphers`指定 sshd 可以接受的加密算法（`Ciphers 3des-cbc`），多个算法之间使用逗号分隔。
- **ClientAliveCountMax**：`ClientAliveCountMax`指定建立连接后，客户端失去响应时，服务器尝试连接的次数（`ClientAliveCountMax 8`）。
- **ClientAliveInterval**：`ClientAliveInterval`指定允许客户端发呆的时间，单位为秒（`ClientAliveInterval 180`）。如果这段时间里面，客户端没有发送任何信号，SSH 连接将关闭。
- **Compression**：`Compression`指定客户端与服务器之间的数据传输是否压缩。默认值为`yes`（`Compression yes`）
- **DenyGroups**：`DenyGroups`指定不允许登录的用户组（`DenyGroups groupName`）。
- **DenyUsers**：`DenyUsers`指定不允许登录的用户（`DenyUsers user1`），用户名之间使用空格分隔，也可以使用多行`DenyUsers`命令指定。
- **FascistLogging**：SSH 1 版本专用，指定日志输出全部 Debug 信息（`FascistLogging yes`）。
- **HostKey**：`HostKey`指定 sshd 服务器的密钥，详见前文。
- **KeyRegenerationInterval**：`KeyRegenerationInterval`指定 SSH 1 版本的密钥重新生成时间间隔，单位为秒，默认是3600秒（`KeyRegenerationInterval 3600`）。
- **ListenAddress**：`ListenAddress`指定 sshd 监听的本机 IP 地址，即 sshd 启用的 IP 地址，默认是 0.0.0.0（`ListenAddress 0.0.0.0`）表示在本机所有网络接口启用。可以改成只在某个网络接口启用（比如`ListenAddress 192.168.10.23`），也可以指定某个域名启用（比如`ListenAddress server.example.com`）。如果要监听多个指定的 IP 地址，可以使用多行`ListenAddress`命令。

```bash
ListenAddress 172.16.1.1
ListenAddress 192.168.0.1
```

- **LoginGraceTime**：`LoginGraceTime`指定允许客户端登录时发呆的最长时间，比如用户迟迟不输入密码，连接就会自动断开，单位为秒（`LoginGraceTime 60`）。如果设为`0`，就表示没有限制。
- **LogLevel**：`LogLevel`指定日志的详细程度，可能的值依次为`QUIET`、`FATAL`、`ERROR`、`INFO`、`VERBOSE`、`DEBUG`、`DEBUG1`、`DEBUG2`、`DEBUG3`，默认为`INFO`（`LogLevel INFO`）。
- **MACs**：`MACs`指定sshd 可以接受的数据校验算法（`MACs hmac-sha1`），多个算法之间使用逗号分隔。
- **MaxAuthTries**：`MaxAuthTries`指定允许 SSH 登录的最大尝试次数（`MaxAuthTries 3`），如果密码输入错误达到指定次数，SSH 连接将关闭。
- **MaxStartups**：`MaxStartups`指定允许同时并发的 SSH 连接数量（MaxStartups）。如果设为`0`，就表示没有限制。这个属性也可以设为`A:B:C`的形式，比如`MaxStartups 10:50:20`，表示如果达到10个并发连接，后面的连接将有50%的概率被拒绝；如果达到20个并发连接，则后面的连接将100%被拒绝。
- **PasswordAuthentication**：`PasswordAuthentication`指定是否允许密码登录，默认值为`yes`（`PasswordAuthentication yes`），建议改成`no`（禁止密码登录，只允许密钥登录）。
- **PermitEmptyPasswords**：`PermitEmptyPasswords`指定是否允许空密码登录，即用户的密码是否可以为空，默认为`yes`（`PermitEmptyPasswords yes`），建议改成`no`（禁止无密码登录）。
- **PermitRootLogin**：`PermitRootLogin`指定是否允许根用户登录，默认为`yes`（`PermitRootLogin yes`），建议改成`no`（禁止根用户登录）。还有一种写法是写成`prohibit-password`，表示 root 用户不能用密码登录，但是可以用密钥登录。

```bash
PermitRootLogin prohibit-password
```

- **PermitUserEnvironment**：`PermitUserEnvironment`指定是否允许 sshd 加载客户端的`~/.ssh/environment`文件和`~/.ssh/authorized_keys`文件里面的`environment= options`环境变量设置。默认值为`no`（`PermitUserEnvironment no`）。
- **Port**：`Port`指定 sshd 监听的端口，即客户端连接的端口，默认是22（`Port 22`）。出于安全考虑，可以改掉这个端口（比如`Port 8822`）。配置文件可以使用多个`Port`命令，同时监听多个端口。

```bash
Port 22
Port 80
Port 443
Port 8080
```

- **PrintMotd**： `PrintMotd`指定用户登录后，是否向其展示系统的 motd（Message of the day）的信息文件`/etc/motd`。该文件用于通知所有用户一些重要事项，比如系统维护时间、安全问题等等。默认值为`yes`（`PrintMotd yes`），由于 Shell 一般会展示这个信息文件，所以这里可以改为`no`。
- **PrintLastLog**：`PrintLastLog`指定是否打印上一次用户登录时间，默认值为`yes`（`PrintLastLog yes`）。
- **Protocol**：`Protocol`指定 sshd 使用的协议。`Protocol 1`表示使用 SSH 1 协议，建议改成`Protocol 2`（使用 SSH 2 协议）。`Protocol 2,1`表示同时支持两个版本的协议。
- **PubKeyAuthentication**：`PubKeyAuthentication`指定是否允许公钥登录，默认值为`yes`（`PubKeyAuthentication yes`）。
- **QuietMode**：SSH 1 版本专用，指定日志只输出致命的错误信息（`QuietMode yes`）。
- **RSAAuthentication**：`RSAAuthentication`指定允许 RSA 认证，默认值为`yes`（`RSAAuthentication yes`）。
- **ServerKeyBits**：`ServerKeyBits`指定 SSH 1 版本的密钥重新生成时的位数，默认是768（`ServerKeyBits 768`）。
- **StrictModes**：`StrictModes`指定 sshd 是否检查用户的一些重要文件和目录的权限。默认为`yes`（`StrictModes yes`），即对于用户的 SSH 配置文件、密钥文件和所在目录，SSH 要求拥有者必须是根用户或用户本人，用户组和其他人的写权限必须关闭。
- **SyslogFacility**：`SyslogFacility`指定 Syslog 如何处理 sshd 的日志，默认是 Auth（`SyslogFacility AUTH`）。
- **TCPKeepAlive**：`TCPKeepAlive`指定打开 sshd 跟客户端 TCP 连接的 keepalive 参数（`TCPKeepAlive yes`）。
- **UseDNS**：`UseDNS`指定用户 SSH 登录一个域名时，服务器是否使用 DNS，确认该域名对应的 IP 地址包含本机（`UseDNS yes`）。打开该选项意义不大，而且如果 DNS 更新不及时，还有可能误判，建议关闭。
- **UseLogin**：`UseLogin`指定用户认证内部是否使用`/usr/bin/login`替代 SSH 工具，默认为`no`（`UseLogin no`）。
- **UserPrivilegeSeparation**：`UserPrivilegeSeparation`指定用户认证通过以后，使用另一个子线程处理用户权限相关的操作，这样有利于提高安全性。默认值为`yes`（`UsePrivilegeSeparation yes`）。
- **VerboseMode**：SSH 2 版本专用，指定日志输出详细的 Debug 信息（`VerboseMode yes`）。
- **X11Forwarding**：`X11Forwarding`指定是否打开 X window 的转发，默认值为 no（`X11Forwarding no`）。

#### 5. 端口转发

>端口转发有两个主要作用：端口转发有三种使用方法：`动态转发`，`本地转发`，`远程转发`。
>
>（1）将不加密的数据放在 SSH 安全连接里面传输，使得原本不安全的网络服务增加了安全性，比如通过端口转发访问 Telnet、FTP 等明文服务，数据传输就都会加密。
>
>（2）作为数据通信的加密跳板，绕过网络防火墙。

##### .1. 动态转发

>动态转发指的是，`本机与 SSH 服务器之间创建了一个加密连接`，然后本机内部针对某个端口的通信，都通过这个加密连接转发。它的一个使用场景就是，访问所有外部网站，都通过 SSH 转发。
>
>- 动态转发需要`把本地端口绑定到 SSH 服务器`。至于` SSH 服务器要去访问哪一个网站，完全是动态的，取决于原始通信`

```shell
ssh -D local-port tunnel-host -N

#使用案例
ssh -D 2121 tunnel-host -N
curl -x socks5://localhost:2121 http://www.example.com  #curl 的`-x`参数指定代理服务器，即通过 SOCKS5 协议的本地`2121`端口，访问`http://www.example.com`。
```

>`-D`表示动态转发，`local-port`是本地端口，`tunnel-host`是 SSH 服务器，`-N`表示这个 SSH 连接只进行端口转发，不登录远程 Shell，不能执行远程命令，只能充当隧道。

##### .2. 本地转发

>SSH 服务器作为中介的跳板机，建立本地计算机与特定目标网站之间的加密连接。本地转发是在本地计算机的 SSH 客户端建立的转发规则。
>
>- `指定一个本地端口（local-port）`，所有发向那个端口的请求，都会转发到 SSH 跳板机（tunnel-host），然后 SSH 跳板机作为中介，将收到的请求发到目标服务器（target-host）的目标端口（target-port）。
>- 本地端口转发`采用 HTTP 协议`，不用转成 SOCKS5 协议。

```shell
ssh -L local-port:target-host:target-port tunnel-host 
#`-L`参数表示本地转发，`local-port`是本地端口，`target-host`是你想要访问的目标服务器，`target-port`是目标服务器的端口，`tunnel-host`是 SSH 跳板机

#使用案例
#现在有一台 SSH 跳板机`tunnel-host`，我们想要通过这台机器，在本地`2121`端口与目标网站`www.example.com`的80端口之间建立 SSH 隧道
ssh -L 2121:www.example.com:80 tunnel-host -N
#访问本机的`2121`端口，就是访问`www.example.com`的80端口
curl http://localhost:2121
```

##### .3. 远程转发

>远程转发则是通过远程计算机访问本地计算机

```shell
#`-R`参数表示远程端口转发，`remote-port`是远程计算机的端口，`target-host`和`target-port`是目标服务器及其端口，`remotehost`是远程计算机。
ssh -R remote-port:target-host:target-port -N remotehost
```

- 内网某台服务器`localhost`在 80 端口开了一个服务，可以通过远程转发将这个 80 端口，映射到具有公网 IP 地址的`my.public.server`服务器的 8080 端口，使得访问`my.public.server:8080`这个地址，就可以访问到那台内网服务器的 80 端口

```shell
ssh -R 8080:localhost:80 -N my.public.server
#在内网`localhost`服务器上执行，建立从`localhost`到`my.public.server`的 SSH 隧道。运行以后，用户访问`my.public.server:8080`，就会自动映射到`localhost:80`。
```

- 本地计算机`local`在外网，SSH 跳板机和目标服务器`my.private.server`都在内网，必须通过 SSH 跳板机才能访问目标服务器。但是，本地计算机`local`无法访问内网之中的 SSH 跳板机，而 SSH 跳板机可以访问本机计算机。

```shell
ssh -R 2121:my.private.server:80 -N local
```

如果经常执行远程端口转发，可以将设置写入 SSH 客户端的用户个人配置文件（`~/.ssh/config`）。

```bash
Host remote-forward
  HostName test.example.com
  RemoteForward remote-port target-host:target-port
```

完成上面的设置后，执行下面的命令就会建立远程转发。

```bash
$ ssh -N remote-forward

# 等同于
$ ssh -R remote-port:target-host:target-port -N test.example.com
```

#### 6. SCP 命令

>`scp`是 secure copy 的缩写，相当于`cp`命令 + SSH。它的底层是 SSH 协议，默认端口是22，相当于先使用`ssh`命令登录远程主机，然后再执行拷贝操作。使用`scp`传输数据时，文件和密码都是加密的，不会泄漏敏感信息。
>
>`scp`主要用于以下三种复制操作。
>
>- 本地复制到远程。
>- 远程复制到本地。
>- 两个远程系统之间的复制。

```bash
#`source`是文件当前的位置，`destination`是文件所要复制到的位置
scp source destination
scp user@host:foo.txt bar.txt
```

**（1）本地文件复制到远程**

- 复制`本机文件`到远程系统的用法如下。


```bash
# 语法
$ scp SourceFile user@host:directory/TargetFile

# 示例
$ scp file.txt remote_username@10.10.0.2:/remote/directory
```

- 下面是复制`整个目录`的例子。


```bash
# 将本机的 documents 目录拷贝到远程主机，
# 会在远程主机创建 documents 目录
$ scp -r documents username@server_ip:/path_to_remote_directory

# 将本机整个目录拷贝到远程目录下
$ scp -r localmachine/path_to_the_directory username@server_ip:/path_to_remote_directory/

# 将本机目录下的所有内容拷贝到远程目录下
$ scp -r localmachine/path_to_the_directory/* username@server_ip:/path_to_remote_directory/
```

**（2）远程文件复制到本地**

- 从远程主机复制文件到本地的用法如下。


```bash
# 语法
$ scp user@host:directory/SourceFile TargetFile

# 示例
$ scp remote_username@10.10.0.2:/remote/file.txt /local/directory
```

- 下面是复制整个目录的例子。


```bash
# 拷贝一个远程目录到本机目录下
$ scp -r username@server_ip:/path_to_remote_directory local-machine/path_to_the_directory/

# 拷贝远程目录下的所有内容，到本机目录下
$ scp -r username@server_ip:/path_to_remote_directory/* local-machine/path_to_the_directory/
$ scp -r user@host:directory/SourceFolder TargetFolder
```

**（3）两个远程系统之间的复制**

- 本机发出指令，从远程主机 A 拷贝到远程主机 B 的用法如下。


```bash
# 语法
$ scp user@host1:directory/SourceFile user@host2:directory/SourceFile

# 示例
$ scp user1@host1.com:/files/file.txt user2@host2.com:/files
```

#### 7. SFTP 命令

> `sftp`是 SSH 提供的一个客户端应用程序，主要用来安全地访问 FTP。因为 FTP 是不加密协议，很不安全，`sftp`就相当于将 FTP 放入了 SSH。

```bash
sftp username@hostname
```

- `ls [directory]`：列出一个远程目录的内容。如果没有指定目标目录，则默认列出当前目录。
- `cd directory`：从当前目录改到指定目录。
- `mkdir directory`：创建一个远程目录。
- `rmdir path`：删除一个远程目录。
- `put localfile [remotefile]`：本地文件传输到远程主机。
- `get remotefile [localfile]`：远程文件传输到本地。
- `help`：显示帮助信息。
- `bye`：退出 sftp。
- `quit`：退出 sftp。
- `exit`：退出 sftp。

#### 8. RSYNC

>在本地计算机与远程计算机之间，或者两个本地目录之间`同步文件`（但`不支持两台远程计算机之间的同步`）。它也可以当作文件复制工具，替代`cp`和`mv`命令。

```shell
#`-r`表示递归，即包含子目录。注意，`-r`是必须的，否则 rsync 运行不会成功。`source`目录表示源目录，`destination`表示目标目录。上面命令执行以后，目标目录下就会出现`destination/source`这个子目录。
rsync -r source destination
#`source1`、`source2`都会被同步到`destination`目录
rsync -r source1 source2 destination
```

- 可以将本地内容，`同步到远程服务器`。


```bash
$ rsync -av source/ username@remote_host:destination
```

- 也可以`将远程内容同步到本地`。


```bash
$ rsync -av username@remote_host:source/ destination
```

```bash
$ rsync -av source/ 192.168.122.32::module/destination
```

注意，上面地址中的`module`并不是实际路径名，而是 rsync 守护程序指定的一个资源名，由管理员分配。

如果想知道 rsync 守护程序分配的所有 module 列表，可以执行下面命令。

```bash
$ rsync rsync://192.168.122.32
```

rsync 协议除了使用双冒号，也可以直接用`rsync://`协议指定地址。

```bash
$ rsync -av source/ rsync://192.168.122.32/module/destination
```

> 第一次同步是全量备份，所有文件在基准目录里面同步一份。以后每一次同步都是增量备份，只同步源目录与基准目录之间有变动的部分，将这部分保存在一个新的目标目录。这个新的目标目录之中，也是包含所有文件，但实际上，只有那些变动过的文件是存在于该目录，其他没有变动的文件都是指向基准目录文件的硬链接。

`--link-dest`参数用来指定同步时的基准目录。

```bash
$ rsync -a --delete --link-dest /compare/path /source/path /target/path
```

> 上面命令中，`--link-dest`参数指定基准目录`/compare/path`，然后源目录`/source/path`跟基准目录进行比较，找出变动的文件，将它们拷贝到目标目录`/target/path`。那些没变动的文件则会生成硬链接。这个命令的第一次备份时是全量备份，后面就都是增量备份了。

下面是一个脚本示例，备份用户的主目录。

```bash
#!/bin/bash

# A script to perform incremental backups using rsync

set -o errexit
set -o nounset
set -o pipefail

readonly SOURCE_DIR="${HOME}"
readonly BACKUP_DIR="/mnt/data/backups"
readonly DATETIME="$(date '+%Y-%m-%d_%H:%M:%S')"
readonly BACKUP_PATH="${BACKUP_DIR}/${DATETIME}"
readonly LATEST_LINK="${BACKUP_DIR}/latest"

mkdir -p "${BACKUP_DIR}"

rsync -av --delete \
  "${SOURCE_DIR}/" \
  --link-dest "${LATEST_LINK}" \
  --exclude=".cache" \
  "${BACKUP_PATH}"

rm -rf "${LATEST_LINK}"
ln -s "${BACKUP_PATH}" "${LATEST_LINK}"
```

> 上面脚本中，每一次同步都会生成一个新目录`${BACKUP_DIR}/${DATETIME}`，并将软链接`${BACKUP_DIR}/latest`指向这个目录。下一次备份时，就将`${BACKUP_DIR}/latest`作为基准目录，生成新的备份目录。最后，再将软链接`${BACKUP_DIR}/latest`指向新的备份目录。

#### Resource

- https://wangdoc.com/ssh/
- [How To Use Rsync to Sync Local and Remote Directories on a VPS](https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories-on-a-vps), Justin Ellingwood
- [Mirror Your Web Site With rsync](https://www.howtoforge.com/mirroring_with_rsync), Falko Timme
- [Examples on how to use Rsync](https://linuxconfig.org/examples-on-how-to-use-rsync-for-local-and-remote-data-backups-and-synchonization), Egidio Docile
- [How to create incremental backups using rsync on Linux](https://linuxconfig.org/how-to-create-incremental-backups-using-rsync-on-linux), Egidio Docile

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/ssh_usage/  

