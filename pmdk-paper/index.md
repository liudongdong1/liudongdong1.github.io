# PMDK-paper


> 

------

# Paper: [MOSIQS](https://sci-hubtw.hkvisa.net/10.1109/mchpc51950.2020.00006)

<div align=center>
<br/>
<b>Persistent Memory Object Storage and Indexing for
Scientific Computing
</b>
</div>



#### Summary

1. a persistent memory object storage framework with `metadata indexing and querying` for scientific computing
2. MOSIQS provides an aggregate memory pool atop an array of persistent memory devices to store and access memory objects.
3.  MOSIQS uses a lightweight persistent memory key-value store to manage the metadata of memory objects such as persistent pointer mappings

#### Challenge

> access, select and share a PMO without additional descriptive metadata.

-  PMOs are memory allocated objects and can only be accessed and shared via persistent pointers. For instance, with Intel’s PMDK libpmemobj API, each stored object on PM is represented by an object handle of type PMEMoid

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220806225625291.png)

- accessing and sharing a PMO requires an additional metadata mapping or index of objects with user or application provided semantics, and persisted along with memory objects

> PMO should be crash consistent

- system should ensure access and consistency of memory object in case of application crash or ungraceful power failures

#### Implement

- employs PMDK provided transactions to ensure atomicity and consistency. 
- The metadata is indexed and managed in a lightweight persistent key-value (KV) store with a persistent B+-tree storage backend

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220806230051614.png)

> - Multiple compute nodes can create a shared namespace abstraction atop the shared PM pool via the MOSIQS library and directly store and manage memory objects on these namespaces.
>
> - .Multiple processes running at these compute nodes can access and share PMOs via namespace abstraction.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220806230302500.png)

> - The `metadata extractor `is responsible to `extract and populate the object name` and `PMEMoid mappings`.
> -  The `sharing manager` is responsible to enable the `data sharing among applications and collaborators. `
>   - For objectlevel sharing, the sharing manager receives the request and checks the requested object mapping in the pool KV store. If the object entry is found, the sharing manager checks the object scope and properties. If the object is shareable, then the sharing manager returns the PMEMoid to requesting application or scientist.
>   - sharing manager validates the group scope and properties from the pool KV store. If the group is annotated with a global and shared scope then, returns the list of OIDs enclosed in the group data part to the requesting application or collaborator
> - `Group Manager` provides logical organization of PMOs defined by application and/or scientists.
> - The pool `KV store is metadata storage backend` for all the metadata of MOSIQS objects. 
> - The namespace manager enables` flexible controls via partitioning large shared PM pool into application or user-defined namespaces.`
>   - Each namespace has its `own metadata KV storage engine` to store and locate PMOs inside the namespace
>   - Applications or scientists using a shared PM pool can access PMOs in another namespace
> - `query manager` is to serve the query requests from the users/scientists and applications.
> - MOSIQS `creates the shared PM pool` via libpmempool API [19]. Any object inside the PM pool is reachable via `Root object pointer`. When an application opens a pool, it is given a privilege to access the global memory Root pointer, which allows applications to locate the PMOs by accessing metadata stored in the pool KV store. The memory allocations and deallocations are conducted via libpmem at the lower level inside libpmemobj.

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220806231548711.png)

-  The OID denotes the PMO object, whereas GID refers to the group metadata object.
- encapsulate each operation as a transaction backed by a logging approach


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/pmdk-paper/  

