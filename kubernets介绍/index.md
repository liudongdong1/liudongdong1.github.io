# Kubernetes-介绍


> Kubernetes is a portable, extensible, open source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation. 
>
> - **Service discovery and load balancing** Kubernetes can expose a container using the DNS name or using their own IP address. If traffic to a container is high, Kubernetes is able to load balance and distribute the network traffic so that the deployment is stable.
> - **Storage orchestration** Kubernetes allows you to automatically mount a storage system of your choice, such as local storages, public cloud providers, and more.
> - **Automated rollouts and rollbacks** You can describe the desired state for your deployed containers using Kubernetes, and it can change the actual state to the desired state at a controlled rate. For example, you can automate Kubernetes to create new containers for your deployment, remove existing containers and adopt all their resources to the new container.
> - **Automatic bin packing** You provide Kubernetes with a cluster of nodes that it can use to run containerized tasks. You tell Kubernetes how much CPU and memory (RAM) each container needs. Kubernetes can fit containers onto your nodes to make the best use of your resources.
> - **Self-healing** Kubernetes restarts containers that fail, replaces containers, kills containers that don't respond to your user-defined health check, and doesn't advertise them to clients until they are ready to serve.
> - **Secret and configuration management** Kubernetes lets you store and manage sensitive information, such as passwords, OAuth tokens, and SSH keys. You can deploy and update secrets and application configuration without rebuilding your container images, and without exposing secrets in your stack configuration.

 ![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220504121148438.png)

### 1. Control Plane

#### .1. kube-apiserver

- The API server is a component of the Kubernetes [control plane](https://kubernetes.io/docs/reference/glossary/?all=true#term-control-plane) that exposes the Kubernetes API. The API server is the front end for the Kubernetes control plane.
- kube-apiserver is designed to scale horizontally—that is, it scales by deploying more instances. You can run several instances of kube-apiserver and balance traffic between those instances.

#### .2. etcd

- Consistent and highly-available `key value store` used as Kubernetes' backing store for all cluster data.

#### .3. kube-scheduler

- Watches for newly created [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) with no assigned [node](https://kubernetes.io/docs/concepts/architecture/nodes/), and selects a node for them to run on.
- Factors taken into account for scheduling decisions include: `individual and collective resource requirements`, `hardware/software/policy constraints`, `affinity and anti-affinity specifications`,` data locality`, `inter-workload interference`, and `deadlines`.

#### .4. kube-controller-manager

> Control plane component that runs [controller](https://kubernetes.io/docs/concepts/architecture/controller/) processes.

- `Node controller`: Responsible for noticing and responding when nodes go down.
- `Job controller:` Watches for Job objects that represent one-off tasks, then creates Pods to run those tasks to completion.
- `Endpoints controller`: Populates the Endpoints object (that is, joins Services & Pods).
- `Service Account & Token controllers`: Create default accounts and API access tokens for new namespaces.

#### .5. cloud-controller-manager

> link your cluster into your cloud provider's API, and separates out the components that interact with that cloud platform from components that only interact with your cluster.

- `Node controller:` For checking the cloud provider to determine if a node has been deleted in the cloud after it stops responding
- `Route controller:` For setting up routes in the underlying cloud infrastructure
- `Service controller: `For creating, updating and deleting cloud provider load balancers

### 2. node components

#### .1. kubelet

- An agent that runs on each [node](https://kubernetes.io/docs/concepts/architecture/nodes/) in the cluster. It makes sure that [containers](https://kubernetes.io/docs/concepts/containers/) are running in a [Pod](https://kubernetes.io/docs/concepts/workloads/pods/).

- The kubelet takes a set of PodSpecs that are provided through various mechanisms and ensures that the containers described in those PodSpecs are running and healthy. The kubelet doesn't manage containers which were not created by Kubernetes.

#### .2. kube-proxy

- [kube-proxy](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/) maintains network rules on nodes. These network rules allow network communication to your Pods from network sessions inside or outside of your cluster.

#### .3. container runtime

- the software that is responsible for running containers.
- supports container runtimes such as [containerd](https://containerd.io/docs/), [CRI-O](https://cri-o.io/#what-is-cri-o), and any other implementation of the [Kubernetes CRI (Container Runtime Interface)](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-node/container-runtime-interface.md).

### 3. Addon

#### .1. DNS

- Cluster DNS is a DNS server, in addition to the other DNS server(s) in your environment, which serves DNS records for Kubernetes services.

#### .2. Web UI

- [Dashboard](https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/) is a general purpose, web-based UI for Kubernetes clusters. It allows users to manage and troubleshoot applications running in the cluster, as well as the cluster itself

#### .3. Container Resource Monitoring

-  records generic time-series metrics about containers in a central database, and provides a UI for browsing that data.

#### .4. Cluster-level Logging

- saving container logs to a central log store with search/browsing interface

### 4. Kubernetes Objects

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220504154924591.png)

> *Kubernetes objects* are `persistent entities` in the Kubernetes system. Kubernetes uses these entities to represent the state of your cluster.
>
> - What containerized applications are running (and on which nodes)
> - The resources available to those applications
> - The policies around how those applications behave, such as restart policies, upgrades, and fault-tolerance.

#### .1. object spec & status

- For objects that have a `spec`, you have to set this when you create the object, providing a description of the characteristics you want the resource to have
- The `status` describes the *current state* of the object, supplied and updated by the Kubernetes system and its components.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2 # tells deployment to run 2 pods matching the template
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

- `apiVersion` - Which version of the Kubernetes API you're using to create this object
- `kind` - What kind of object you want to create
- `metadata` - Data that helps uniquely identify the object, including a `name` string, `UID`, and optional `namespace`
- `spec` - What state you desire for the object

#### .2. object manager

> A Kubernetes object should be managed using only one technique. Mixing and matching techniques for the same object results in undefined behavior.

##### 1. Imperative commands

```sh
kubectl create deployment nginx --image nginx
```

##### 2. Imperative object configuration[ ](https://kubernetes.io/docs/concepts/overview/working-with-objects/object-management/#imperative-object-configuration)

```sh
kubectl create -f nginx.yaml
```

##### 3. Declarative object configuration

When using declarative object configuration, a user operates on object configuration files stored locally, however the user does not define the operations to be taken on the files. Create, update, and delete operations are automatically detected per-object by `kubectl`. This enables working on directories, where different operations might be needed for different objects.

```sh
kubectl diff -f configs/
kubectl apply -f configs/
```

#### .3. Namespace

> namespaces provides a mechanism for isolating groups of resources within a single cluster. Names of resources need to be unique within a namespace, but not across namespaces. Namespace-based scoping is applicable only for namespaced objects *(e.g. Deployments, Services, etc)* and not for cluster-wide objects *(e.g. StorageClass, Nodes, PersistentVolumes, etc)*.

Kubernetes starts with four initial namespaces:  `kubectl get namespace`

- `default` The default namespace for objects with no other namespace
- `kube-system` The namespace for objects created by the Kubernetes system
- `kube-public` This namespace is created automatically and is readable by all users (including those not authenticated). This namespace is mostly `reserved for cluster usage`, in case that some resources should be visible and readable publicly throughout the whole cluster. The public aspect of this namespace is only a convention, not a requirement.
- `kube-node-lease` This namespace holds [Lease](https://kubernetes.io/docs/reference/kubernetes-api/cluster-resources/lease-v1/) objects associated with each node. Node leases allow the kubelet to send [heartbeats](https://kubernetes.io/docs/concepts/architecture/nodes/#heartbeats) so that the control plane can detect node failure.

#### .4. Annotations & Labels & Selectors

- Annotations: to attach arbitrary non-identifying metadata to objects. Clients such as tools and libraries can retrieve this metadata.
- Labels can be used to select objects and to find collections of objects that satisfy certain conditions. 
- 第一个元数据，也是最重要的一个元数据是：资源标签。

#### .5. Field Selectors

-  [select Kubernetes resources](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects) based on the value of one or more resource fields

```shell
kubectl get pods --field-selector status.phase=Running
```

#### .6. Finalizers

- Finalizers are namespaced keys that tell Kubernetes to wait until specific conditions are met before it fully deletes resources marked for deletion. 

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/kubernets%E4%BB%8B%E7%BB%8D/  

