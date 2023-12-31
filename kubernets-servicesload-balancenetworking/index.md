# Kubernetes-Service&Load Balancing&Networking


> Kubernetes IP addresses exist at the `Pod` scope - containers within a `Pod` share their network namespaces - including their IP address and MAC address. This means that containers within a `Pod` can all reach each other's ports on `localhost`. This also means that containers within a `Pod` must coordinate port usage, but this is no different from processes in a VM. 
>
> - Containers within a Pod [use networking to communicate](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) via loopback.
> - Cluster networking provides communication between different Pods.
> - The [Service resource](https://kubernetes.io/docs/concepts/services-networking/service/) lets you [expose an application running in Pods](https://kubernetes.io/docs/concepts/services-networking/connect-applications-service/) to be reachable from outside your cluster.
> - You can also use Services to [publish services only for consumption inside your cluster](https://kubernetes.io/docs/concepts/services-networking/service-traffic-policy/).

### 1. Service

- a Service is an abstraction which defines a logical set of Pods and a policy by which to access them.
- The set of Pods targeted by a Service is usually determined by a [selector](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/). 

#### .1. service with selectors

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220506102656641.png)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app.kubernetes.io/name: proxy
spec:
  containers:
  - name: nginx
    image: nginx:11.14.2
    ports:
      - containerPort: 80
        name: http-web-svc
        
---
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app.kubernetes.io/name: proxy
  ports:
  - name: name-of-service-port
    protocol: TCP
    port: 80
    targetPort: http-web-svc
#This specification creates a new Service object named "my-service", which targets TCP port 9376 on any Pod with the app=MyApp label.
```

#### .2. service without selectors

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220506102934990.png)

when used with a corresponding Endpoints object and without a selector, the Service can abstract other kinds of backends, including ones that run outside the cluster. 

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  ports:
    - protocol: TCP
      port: 80          #service 提供给前端的端口
      targetPort: 9376   # 流量发往Pod的这个端口上。
```

Service has no selector, the corresponding Endpoints object is not created automatically. You can manually map the Service to the network address and port where it's running, by adding an Endpoints object manually

```yaml
apiVersion: v1
kind: Endpoints
metadata:
  name: my-service   # name需要和service相同
subsets:
  - addresses:
      - ip: 192.0.2.42  #后端podip地址
    ports:
      - port: 9376
```

#### 3.Multi-Port Services

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: 9376
    - name: https
      protocol: TCP
      port: 443
      targetPort: 9377
```

#### 4. publishing service

For some parts of your application (for example, frontends) you may want to `expose a Service onto an external IP address`, that's outside of your cluster.

Kubernetes `ServiceTypes` allow you to specify what kind of Service you want. The default is `ClusterIP`.

`Type` values and their behaviors are:

- `ClusterIP`: Exposes the Service on a cluster-internal IP. Choosing this value makes the Service only `reachable from within the cluster`. This is the default `ServiceType`.
- [`NodePort`](https://kubernetes.io/docs/concepts/services-networking/service/#type-nodeport): Exposes the Service on e`ach Node's IP at a static port` (the `NodePort`). A `ClusterIP` Service, to which the `NodePort` Service routes, is automatically created. You'll be able to contact the `NodePort` Service, from outside the cluster, by requesting `<NodeIP>:<NodePort>`.
- [`LoadBalancer`](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer): Exposes the Service externally using a cloud provider's load balancer. `NodePort` and `ClusterIP` Services, to which the external load balancer routes, are automatically created.
- [`ExternalName`](https://kubernetes.io/docs/concepts/services-networking/service/#externalname): Maps the Service to the contents of the `externalName` field (e.g. `foo.bar.example.com`), by returning a `CNAME` record with its value. No proxying of any kind is set up.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: NodePort
  selector:
    app: MyApp
  ports:
      # By default and for convenience, the `targetPort` is set to the same value as the `port` field.
    - port: 80
      targetPort: 80
      # Optional field
      # By default and for convenience, the Kubernetes control plane will allocate a port from a range (default: 30000-32767)
      nodePort: 30007
#this Service is visible as <NodeIP>:spec.ports[*].nodePort and .spec.clusterIP:spec.ports[*].port
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  clusterIP: 10.0.171.239
  type: LoadBalancer
status:
  loadBalancer:
    ingress:
    - ip: 192.0.2.127
```

### 2. Topology-aware traffic routing

*Service Topology* enables a service to route traffic based upon the Node topology of the cluster.

#### .1. only on local endpoints

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  topologyKeys:
    - "kubernetes.io/hostname"
```

#### .2. prefer node local endpoints

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  topologyKeys:
    - "kubernetes.io/hostname"
    - "*"
```

#### .3. only zonal or regional endpoints

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  topologyKeys:
    - "topology.kubernetes.io/zone"
    - "topology.kubernetes.io/region"
```

#### .4. prefer node local, zonal, regional endpoints

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
  topologyKeys:
    - "kubernetes.io/hostname"
    - "topology.kubernetes.io/zone"
    - "topology.kubernetes.io/region"
    - "*"
```

### 3. DNS for services and Pods

- Kubernetes creates DNS records for services and pods. You can contact services with consistent DNS names instead of IP addresses
- A DNS query may return different results `based on the namespace of the pod making it`. `DNS queries that don't specify a namespace are limited to the pod's namespace`. Access services in other namespaces by specifying it in the DNS query.

#### .1. Pod's DNS Policy

- "`Default`": The Pod inherits the name resolution configuration from the node that the pods run on. See [related discussion](https://kubernetes.io/docs/tasks/administer-cluster/dns-custom-nameservers) for more details.
- "`ClusterFirst`": Any DNS query that does not match the configured cluster domain suffix, such as "`www.kubernetes.io`", is forwarded to the upstream nameserver inherited from the node. Cluster administrators may have extra stub-domain and upstream DNS servers configured. See [related discussion](https://kubernetes.io/docs/tasks/administer-cluster/dns-custom-nameservers) for details on how DNS queries are handled in those cases.
- "`ClusterFirstWithHostNet`": For Pods running with hostNetwork, you should explicitly set its DNS policy "`ClusterFirstWithHostNet`".
- "`None`": It allows a Pod to ignore DNS settings from the Kubernetes environment. All DNS settings are supposed to be provided using the `dnsConfig` field in the Pod Spec. See [Pod's DNS config](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/#pod-dns-config) subsection below.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: default
spec:
  containers:
  - image: busybox:1.28
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: busybox
  restartPolicy: Always
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
```

#### .2. Pod's DNS Config

The `dnsConfig` field is optional and it can work with any `dnsPolicy` settings. However, when a Pod's `dnsPolicy` is set to "`None`", the `dnsConfig` field has to be specified.

- `nameservers`: a list of IP addresses that will be used as DNS servers for the Pod. There can be at most 3 IP addresses specified. When the Pod's `dnsPolicy` is set to "`None`", the list must contain at least one IP address, otherwise this property is optional. The servers listed will be combined to the base nameservers generated from the specified DNS policy with duplicate addresses removed.
- `searches`: `a list of DNS search domains for hostname lookup in the Pod`. This property is optional. When specified, the provided list will be merged into the base search domain names generated from the chosen DNS policy. Duplicate domain names are removed. Kubernetes allows for at most 6 search domains.
- `options`: an optional list of objects where each object may have a `name` property (required) and a `value` property (optional). The contents in this property will be merged to the options generated from the specified DNS policy. Duplicate entries are removed.

```yaml
apiVersion: v1
kind: Pod
metadata:
  namespace: default
  name: dns-example
spec:
  containers:
    - name: test
      image: nginx
  dnsPolicy: "None"
  dnsConfig:
    nameservers:
      - 1.2.3.4
    searches:
      - ns1.svc.cluster-domain.example
      - my.dns.search.suffix
    options:
      - name: ndots
        value: "2"
      - name: edns0
```

### 4. connecting applications with services

#### .1. create&expose service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-nginx
spec:
  selector:
    matchLabels:
      run: my-nginx
  replicas: 2
  template:
    metadata:
      labels:
        run: my-nginx
    spec:
      containers:
      - name: my-nginx
        image: nginx
        ports:
        - containerPort: 80
```

```shell
$ kubectl apply -f ./run-my-nginx.yaml
$ kubectl expose deployment/my-nginx  # 该行命令同下面的yaml配置文件
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-nginx
  labels:
    run: my-nginx
spec:
  ports:
  - port: 80
    protocol: TCP
  selector:
    run: my-nginx
```

#### .2. accessing service

```shell
$ kubectl exec my-nginx-3800858182-jr4a2 -- printenv | grep SERVICE
```

### 5. Ingress

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220506153948247.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernets-servicesload-balancenetworking/  

