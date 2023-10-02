# Kubernetes-存储NFS实践


# 一、storage存储原理

![img](https://img-bc.icode.best/82445a0629624def895f0d35785235f0.png)

# 二、storage存储实战

## 2.1 安装nfs存储集群

```shell
#所有机器安装
yum install -y nfs-utils

#nfs主节点
echo "/nfs/data/ *(insecure,rw,sync,no_root_squash)" > /etc/exports
mkdir -p /nfs/data
systemctl enable rpcbind --now
systemctl enable nfs-server --now

#配置生效
exportfs -r
showmount -e 172.31.0.4

#执行以下命令挂载 nfs 服务器上的共享目录到本机路径 /root/nfsmount
mkdir -p /nfs/data
mount -t nfs 172.31.0.4:/nfs/data /nfs/data

# 写入一个测试文件
echo "hello nfs server" > /nfs/data/test.txt
```

##  2.2 原生方式数据挂载

```
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx-pv-demo
  name: nginx-pv-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx-pv-demo
  template:
    metadata:
      labels:
        app: nginx-pv-demo
    spec:
      containers:
      - image: nginx
        name: nginx
        volumeMounts:
        - name: html
          mountPath: /usr/share/nginx/html
      volumes:
        - name: html
          nfs:
            server: 172.31.0.4
            path: /nfs/data/nginx-pv
```

# 三、 PV&PVC实战

PV：持久卷（Persistent Volume），将应用需要持久化的数据保存到指定位置

PVC：持久卷申明（**Persistent Volume Claim**），申明需要使用的持久卷规格

## 静态方式

![](../../../../blogimgv2022/48276d871abf896f043930526ff3f228.png)

- 静态供应

```
#nfs主节点
mkdir -p /nfs/data/01
mkdir -p /nfs/data/02
mkdir -p /nfs/data/03
```

- **创建PV**

```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv01-10m
spec:
  capacity:
    storage: 10M
  accessModes:
    - ReadWriteMany
  storageClassName: nfs
  nfs:
    path: /nfs/data/01
    server: 172.31.0.4（修改为自己的Ip地址）
    
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv02-1gi
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteMany
  storageClassName: nfs
  nfs:
    path: /nfs/data/02
    server: 172.31.0.4（修改为自己的Ip地址）
    
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv03-3gi
spec:
  capacity:
    storage: 3Gi
  accessModes:
    - ReadWriteMany
  storageClassName: nfs
  nfs:
    path: /nfs/data/03
    server: 172.31.0.4（修改为自己的Ip地址）
```

- **PVC创建与绑定**

```
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: nginx-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 200Mi
  storageClassName: nfs
```

- **创建Pod绑定PVC**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx-deploy-pvc
  name: nginx-deploy-pvc
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx-deploy-pvc
  template:
    metadata:
      labels:
        app: nginx-deploy-pvc
    spec:
      containers:
      - image: nginx
        name: nginx
        volumeMounts:
        - name: html
          mountPath: /usr/share/nginx/html
      volumes:
        - name: html
          persistentVolumeClaim:
            claimName: nginx-pvc
```

## 动态方式

![](../../../../blogimgv2022/b0f106a28b78150de22297059b2a8aa1.png)

```yaml
#集群管理员只需要保证环境中有 NFS 相关的 storageclass 即可：
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: nfs-sc
provisioner: example.com/nfs    # 对于的分配资源的插件
mountOptions:
  - vers=4.1


#用户创建 PVC，此处 PVC 的 storageClassName 指定为上面 NFS 的 storageclass 名称
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: nfs
  annotations:
    volume.beta.kubernetes.io/storage-class: "example-nfs"
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Mi
  storageClassName: nfs-sc

```



# 四、ConfigMap

- 抽取应用配置，并且可以自动更新
- 把之前的配置文件创建为配置集

```yaml
# 创建配置，redis保存到k8s的etcd
kubectl create cm redis-conf --from-file=redis.conf
apiVersion: v1
data:    #data是所有真正的数据，key：默认是文件名   value：配置文件的内容
  redis.conf: |
    appendonly yes
kind: ConfigMap
metadata:
  name: redis-conf
  namespace: default
```

- 创建Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: redis
spec:
  containers:
  - name: redis
    image: redis
    command:
      - redis-server
      - "/redis-master/redis.conf"  #指的是redis容器内部的位置
    ports:
    - containerPort: 6379
    volumeMounts:
    - mountPath: /data
      name: data
    - mountPath: /redis-master
      name: config
  volumes:
    - name: data
      emptyDir: {}
    - name: config
      configMap:
        name: redis-conf
        items:
        - key: redis.conf
          path: redis.conf
```

检查默认配置

```
kubectl exec -it redis -- redis-cli

127.0.0.1:6379> CONFIG GET appendonly

127.0.0.1:6379> CONFIG GET requirepass
```

- 修改ConfigMap

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: example-redis-config
data:
  redis-config: |
    maxmemory 2mb
    maxmemory-policy allkeys-lru 
```

- 检查配置是否更新

```
kubectl exec -it redis -- redis-cli

127.0.0.1:6379> CONFIG GET maxmemory
127.0.0.1:6379> CONFIG GET maxmemory-policy
```

检查指定文件内容是否已经更新，修改了CM。Pod里面的配置文件会跟着变。***配置值未更改，因为需要重新启动 Pod 才能从关联的 ConfigMap 中获取更新的值。 原因：我们的Pod部署的中间件自己本身没有热更新能力。\***

# 五、Secret

Secret 对象类型用来`保存敏感信息，例如密码、OAuth 令牌和 SSH 密钥`。 将这些信息放在 secret 中比放在 [Pod](https://icode.best/go?go=aHR0cHM6Ly9rdWJlcm5ldGVzLmlvL2RvY3MvY29uY2VwdHMvd29ya2xvYWRzL3BvZHMvcG9kLW92ZXJ2aWV3Lw==) 的定义或者 [容器镜像](https://icode.best/go?go=aHR0cHM6Ly9rdWJlcm5ldGVzLmlvL3poL2RvY3MvcmVmZXJlbmNlL2dsb3NzYXJ5Lz9hbGw9dHJ1ZSN0ZXJtLWltYWdl) 中来说更加安全和灵活。

```
https://icode.best/i/78604243851406kubectl create secret docker-registry leifengyang-docker \
--docker-username=leifengyang \
--docker-password=Lfy123456 \
--docker-email=534096094@qq.com

##命令格式
kubectl create secret docker-registry regcred \
  --docker-server=<你的镜像仓库服务器> \
  --docker-username=<你的用户名> \
  --docker-password=<你的密码> \
  --docker-email=<你的邮箱地址>
apiVersion: v1
kind: Pod
metadata:
  name: private-nginx
spec:
  containers:
  - name: private-nginx
    image: leifengyang/guignginx:v1.0
  imagePullSecrets:
  - name: leifengyang-docker
```

## Resource

- https://icode.best/i/78604243851406

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kubernetes_%E5%AD%98%E5%82%A8nfs%E5%AE%9E%E8%B7%B5/  

