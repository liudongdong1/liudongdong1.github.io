# Kubernetes-workload


> A workload is an application running on Kubernetes. Whether your workload is a single component or several that work together, on Kubernetes you run it inside a set of [*pods*](https://kubernetes.io/docs/concepts/workloads/pods). In Kubernetes, a `Pod` represents a set of running [containers](https://kubernetes.io/docs/concepts/containers/) on your cluster.
>
> - [`Deployment`](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) and [`ReplicaSet`](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/) (replacing the legacy resource [ReplicationController](https://kubernetes.io/docs/reference/glossary/?all=true#term-replication-controller)). `Deployment` is a good fit for managing a stateless application workload on your cluster, where any `Pod` in the `Deployment` is interchangeable and can be replaced if needed.
> - [`StatefulSet`](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/) lets you run one or more related Pods that do track state somehow. For example, if your workload records data persistently, you can run a `StatefulSet` that matches each `Pod` with a [`PersistentVolume`](https://kubernetes.io/docs/concepts/storage/persistent-volumes/). Your code, running in the `Pods` for that `StatefulSet`, can replicate data to other `Pods` in the same `StatefulSet` to improve overall resilience.
> - [`DaemonSet`](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) defines `Pods` that provide node-local facilities. These might be fundamental to the operation of your cluster, such as a networking helper tool, or be part of an [add-on](https://kubernetes.io/docs/concepts/cluster-administration/addons/).
>   Every time you add a node to your cluster that matches the specification in a `DaemonSet`, the control plane schedules a `Pod` for that `DaemonSet` onto the new node.
> - [`Job`](https://kubernetes.io/docs/concepts/workloads/controllers/job/) and [`CronJob`](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/) define tasks that run to completion and then stop. Jobs represent one-off tasks, whereas `CronJobs` recur according to a schedule.

### 1. Deployment

#### .1. create a deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3   # create 3 pods
  selector:
    matchLabels:
      app: nginx
  template:     #create a pod from template
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

```shell
$ kubectl apply -f nginx-deployment.yaml
$ kubectl rollout status deployment/nginx-deployment  #查看部署状态
$ kubectl get rs # 查看 replicaset
$ kubectl get pods --show-labels  
```

#### .2. updating a deployment

```shell
$ kubectl set image deployment/nginx-deployment nginx=nginx:1.16.1
# 或者
$ kubectl edit deployment/nginx-deployment
$ kubectl get rs  #the Deployment updated the Pods by creating a new ReplicaSet and scaling it up to 3 replicas, as well as scaling down the old ReplicaSet to 0 replicas.
$ kubectl describe deployments  #get the detail of the deployments
```

#### .3. rolling back a deployment

```shell
$ kubectl rollout history deployment/nginx-deployment #check the revisions of this Deployment
$ kubectl rollout history deployment/nginx-deployment --revision=2  #see the details of a revision
$ kubectl rollout undo deployment/nginx-deployment
$ kubectl rollout undo deployment/nginx-deployment --to-revision=2 # roll back to specific version
```

#### .4. scaling a deployment

```shell
$ kubectl scale deployment/nginx-deployment --replicas=10
$ kubectl autoscale deployment/nginx-deployment --min=10 --max=15 --cpu-percent=80
```

#### .5. pausing a deployment

> When you `update a Deployment, or plan to`, you can` pause rollouts for that Deployment before you trigger one or more updates`. When you're ready to apply those changes, you resume rollouts for the Deployment. This approach allows you to apply multiple fixes in between pausing and resuming without triggering unnecessary rollouts. `You cannot rollback a paused Deployment until you resume it.`

```shell
$ kubectl rollout pause deployment/nginx-deployment
$ kubectl set image deployment/nginx-deployment nginx=nginx:1.16.1
$ kubectl set resources deployment/nginx-deployment -c=nginx --limits=cpu=200m,memory=512Mi
$ kubectl rollout resume deployment/nginx-deployment
```

### 2. ReplicaSet

-  purpose is to maintain a stable set of replica Pods running at any given time.
- A ReplicaSet is defined with fields, including `a selector that specifies how to identify Pods it can acquire`, `a number of replicas indicating how many Pods it should be maintaining`, and` a pod template specifying the data of new Pods it should create to meet the number of replicas criteria.`
- A ReplicaSet is linked to its Pods via the Pods' [metadata.ownerReferences](https://kubernetes.io/docs/concepts/workloads/controllers/garbage-collection/#owners-and-dependents) field, which `specifies what resource the current object is owned by`. All Pods acquired by a ReplicaSet have their owning ReplicaSet's identifying information within their ownerReferences field. It's through this link that the ReplicaSet knows of the state of the Pods it is maintaining and plans accordingly.
- recommend `using Deployments instead of directly using ReplicaSets`, unless you require custom update orchestration or don't require updates at all.
- a ReplicaSet is not limited to owning Pods specified by its template-- it can acquire other Pods in the manner specified in the previous sections.  `Pods create with no template don't have a Controller as their owner reference`

#### .1. Deleting a ReplicaSet and its Pods

```shell
kubectl proxy --port=8080
curl -X DELETE  'localhost:8080/apis/apps/v1/namespaces/default/replicasets/frontend' \
> -d '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Foreground"}' \
> -H "Content-Type: application/json"
```

#### .2. Deleting just a ReplicaSet

```shell
kubectl proxy --port=8080
curl -X DELETE  'localhost:8080/apis/apps/v1/namespaces/default/replicasets/frontend' \
> -d '{"kind":"DeleteOptions","apiVersion":"v1","propagationPolicy":"Orphan"}' \
> -H "Content-Type: application/json"
```

### 3. StatefulSets

- Manages the deployment and scaling of a set of [Pods](https://kubernetes.io/docs/concepts/workloads/pods/), *and provides guarantees about the ordering and uniqueness* of these Pods.
- a StatefulSet manages Pods that are based on an identical container spec. `Unlike a Deployment, a StatefulSet maintains a sticky identity for each of their Pods`. These pods are created from the same spec, but are not interchangeable: each has a persistent identifier that it maintains across any rescheduling.

### 4. DaemonSet

A *DaemonSet* `ensures that all (or some) Nodes run a copy of a Pod`. As nodes are added to the cluster, Pods are added to them. As nodes are removed from the cluster, those Pods are garbage collected. Deleting a DaemonSet will clean up the Pods it created.

If you specify a `.spec.template.spec.nodeSelector`, then the DaemonSet controller will create Pods on nodes which match that [node selector](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/). Likewise if you specify a `.spec.template.spec.affinity`, then DaemonSet controller will create Pods on nodes which match that [node affinity](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/). If you do not specify either, then the DaemonSet controller will create Pods on all nodes.

- running a` cluster storage daemon on every node`
- running a `logs collection daemon on every node`
- running a `node monitoring daemon on every node`

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd-elasticsearch
  namespace: kube-system
  labels:
    k8s-app: fluentd-logging
spec:
  selector:
    matchLabels:
      name: fluentd-elasticsearch
  template:
    metadata:
      labels:
        name: fluentd-elasticsearch
    spec:
      tolerations:
      # these tolerations are to have the daemonset runnable on control plane nodes
      # remove them if your control plane nodes should not run pods
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      containers:
      - name: fluentd-elasticsearch
        image: quay.io/fluentd_elasticsearch/fluentd:v2.5.2
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 200Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
```

#### .1. How Daemon Pods are scheduled

##### 1. Scheduled by default scheduler

DaemonSet pods are created and scheduled by the `DaemonSet controller `instead. That introduces the following issues:

- `Inconsistent Pod behavior`: Normal Pods waiting to be scheduled are created and in `Pending` state, but DaemonSet pods are not created in `Pending` state. This is confusing to the user.
- [Pod preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/) is handled by default scheduler. When preemption is enabled, the DaemonSet controller will make scheduling decisions without considering pod priority and preemption.

```yaml
nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchFields:
      - key: metadata.name
        operator: In
        values:
        - target-host-name
```

##### 2.Taints and Tolerations

| Toleration Key                           | Effect     | Version | Description                                                  |
| ---------------------------------------- | ---------- | ------- | ------------------------------------------------------------ |
| `node.kubernetes.io/not-ready`           | NoExecute  | 1.13+   | DaemonSet pods will not be evicted when there are node problems such as a network partition. |
| `node.kubernetes.io/unreachable`         | NoExecute  | 1.13+   | DaemonSet pods will not be evicted when there are node problems such as a network partition. |
| `node.kubernetes.io/disk-pressure`       | NoSchedule | 1.8+    | DaemonSet pods tolerate disk-pressure attributes by default scheduler. |
| `node.kubernetes.io/memory-pressure`     | NoSchedule | 1.8+    | DaemonSet pods tolerate memory-pressure attributes by default scheduler. |
| `node.kubernetes.io/unschedulable`       | NoSchedule | 1.12+   | DaemonSet pods tolerate unschedulable attributes by default scheduler. |
| `node.kubernetes.io/network-unavailable` | NoSchedule | 1.12+   | DaemonSet pods, who uses host network, tolerate network-unavailable attributes by default scheduler. |

### 5. [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)

`A Job creates one or more Pods` and will continue to retry execution of the Pods until a specified number of them successfully terminate. `As pods successfully complete, the Job tracks the successful completions.` `When a specified number of successful completions is reached, the task (ie, Job) is complete. `Deleting a Job will clean up the Pods it created. Suspending a Job will delete its active Pods until the Job is resumed again.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pi
spec:
  template:
    spec:
      containers:
      - name: pi
        image: perl
        command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
      restartPolicy: Never
  backoffLimit: 4
```

```shell
pods=$(kubectl get pods --selector=job-name=pi --output=jsonpath='{.items[*].metadata.name}')
echo $pods  #list all the Pods that belong to a Job in a machine readable form,
```

### 6. Automatic Clean-up for Finished Jobs

 provides a TTL (time to live) mechanism to limit the lifetime of resource objects that have finished execution. TTL controller only handles [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/).

### 7. [CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)

CronJobs are meant for performing` regular scheduled actions such as backups, report generation, and so on`. Each of those tasks should be configured to recur indefinitely (for example: once a day / week / month); you can define the point in time within that interval when the job should start.

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hello
spec:
  schedule: "* * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: hello
            image: busybox:1.28
            imagePullPolicy: IfNotPresent
            command:
            - /bin/sh
            - -c
            - date; echo Hello from the Kubernetes cluster
          restartPolicy: OnFailure
```

| Entry                  | Description                                                | Equivalent to |
| ---------------------- | ---------------------------------------------------------- | ------------- |
| @yearly (or @annually) | Run once a year at midnight of 1 January                   | 0 0 1 1 *     |
| @monthly               | Run once a month at midnight of the first day of the month | 0 0 1 * *     |
| @weekly                | Run once a week at midnight on Sunday morning              | 0 0 * * 0     |
| @daily (or @midnight)  | Run once a day at midnight                                 | 0 0 * * *     |
| @hourly                | Run once an hour at the beginning of the hour              | 0 * * * *     |

### 8. ReplicationControl

- a ReplicationController makes sure that a pod or a homogeneous set of pods is always up and available.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

```shell
pods=$(kubectl get pods --selector=app=nginx --output=jsonpath={.items..metadata.name})
echo $pods  #list all the pods that belong to the ReplicationController in a machine readable form
```


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/kubernetes-practice/  

