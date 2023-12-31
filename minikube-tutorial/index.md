# Minikube-tutorial


> Minikube 是一种轻量化的 Kubernetes 集群，是 Kubernetes 社区为了帮助开发者和学习者能够更好学习和体验 k8s 功能而推出的，借助个人 PC 的虚拟化环境就可以实现 Kubernetes 的快速构建启动。

### 1. Minikube 和 Kubernetes 的区别

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16254840-44f55035f12879c9.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/16254840-4bec1f9098962451.png)

### 2. Minikube 基础操作

```shell
$ minikube start  #启动并运访问 minikube 中的 k8s dashboard行一个集群
$ minikube dashboard  #访问 minikube 中的 k8s dashboard

$kubectl create deployment hello-node --image=k8s.gcr.io/echoserver:1.4 #创建一个deployment

$ kubectl expose deployment hello-node --type=LoadBalancer --port=8080  #创建一个service
$ minikube service hello-node

$ minikube addons list   #List the currently supported addons
$ minikube addons enable metrics-server #Enable an addon, for example, metrics-server
$ minikube addons disable metrics-server

$ kubectl delete service hello-node
$ kubectl delete deployment hello-node
$ minikube stop          #停止一个集群
$ minikube delete      #删除一个集群
$ minikube delete --all   #删除本地所有集群和配置
```

### 3. kubernetes 基本操作

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220504165556289.png)

#### .1. create a cluster

```bash
$ minikube start
$ minikube version
$ kubectl version
$ kubectl cluster-info
$ kubectl get nodes
```

#### .2. Deploying an app

```sh
$ kubectl create deployment hello-minikube1 --image=registry.cn-hangzhou.aliyuncs.com/google_containers/echoserver:1.10

$ kubectl get deployments

$ kubectl expose deployment hello-minikube1 --type=LoadBalancer --port=8080 #创建 LB 服务，将服务暴露出来
$ kubectl get svc #查看外部访问 IP
```

```shell
$ kubectl proxy  # use proxy to direct access to the api
curl http://localhost:8001/version

export POD_NAME=$(kubectl get pods -o go-template --template '{{range.items}} {{.metadata.name}}{{"\n"}}{{end}}')

$ curl http://localhost:8001/api/v1/namespaces/default/pods/$POD_NAME/proxy/

$ kubectl logs $POD_NAME

$ kubectl exec $POD_NAMES -- env  #查看环境变量
$ kubectl exec -ti $POD_NAME -- bash
```

#### .3. exploring the app

```shell
$ kubectl describe pods  # 查看pods 详细信息

$ kubectl logs $POD_NAME

$ kubectl exec $POD_NAMES -- env  #查看环境变量
$ kubectl exec -ti $POD_NAME -- bash
```

#### .4. Expose your app

```shell
$ kubectl get services

$ kubectl expose deployment/kubernetes-bootcamp --type="NodePort" --port 8080

$ export NODE_PORT=$(kubectl get services/kubernetes-bootcamp -o go-template='{{(index .spec.ports O).nodePort}}')  #32457

$ kubectl delete service -l app=kubernetes-bootcamp
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220504181352083.png)

#### .5. Scale the app

```shell
$ kubectl scale deployments/kubernetes-bootcamp --replicas=4
```

#### .6. update the app

```shell
$ kubectl set image deployments/kubernetes-bootcamp kubernetes-bootcamp=jocatalin/kubernetes-bootcamp:v2

$ kubectl rollout undo deployments/kubernetes-bootcamp  # roll back to the last working version
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/minikube-tutorial/  

