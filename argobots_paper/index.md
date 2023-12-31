# argobots


> 

------

# Paper: argobots

<div align=center>
<br/>
<b>Argobots: A Lightweight Low-Level Threading and Tasking Framework
Library
</b>
</div>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220730211342377-16691013698119.png)

> SM1 in ES1 has one associated private pool, PM11, and SM2 in ES2 has two private pools, PM21 and PM22.
>
> PS is shared between ES1 and ES2, and thus both SM1 in ES1 and SM2 in ES2 can access the pool to push or pop work units.
>
> PE denotes an event pool. The event pool is meant for lightweight notification. It is periodically checked by a scheduler to handle the arrival of events (e.g., messages from the network).
>
> S1 and S2 in PM11 are stacked schedulers that will be executed by the main scheduler SM1.

#### Execution Model

- An `ES `maps to one` OS thread`, is explicitly created by the user, and executes independently of other ESs.
- A work unit is a lightweight execution unit, a ULT or a tasklet, that runs within an ES.
- Each ES is associated with its own scheduler that is in charge of scheduling work units according to its scheduling policy
- A ULT has its own `stack region`, whereas a tasklet borrows the stack of its host ES’s scheduler
- A ULT is an independent execution unit in user space and provides standard thread semantics at a low context-switching cost.
- ULTs are suitable for expressing parallelism in terms of persistent contexts whose flow of control can be paused and resumed
- Tasklets do not yield control and run to completion before returning control to the scheduler that invoked them. tasklets are used for atomic work without suspending

#### Scheduler

- Argobots provides an infrastructure for stackable or nested schedulers, with pluggable scheduling policies
- Plugging in custom policies enables higher levels of the software stack to use their special policies while Argobots handles the lowlevel scheduling mechanisms.
-  stacking schedulers empowers the user to switch schedulers when multiple software modules or programming models interact in an application. 例如，当应用程序执行具有自己的调度程序的外部库时，它会暂停当前调度程序并调用该库的调度程序。

#### Primitive Operations

- **Creation**. When ULTs or tasklets are created, they are `inserted into a specific pool in a ready state`. Thus, they will be scheduled by the scheduler associated with the target pool and executed in the ES associated with the scheduler. If the pool is shared with more than one scheduler and the schedulers run in different ESs, `the work units may be scheduled in any of the ESs.` 
- **Join**: Work units can be joined by other ULTs. When a work unit is joined, it is guaranteed to have terminated. 
- **Yield**. When a ULT yields control(让出控制权）, the control goes to the scheduler that was in charge of scheduling in the ES at the point of yield time. The target scheduler schedules the next work unit according to its scheduling policy.
- **Yield to**. When a ULT calls yield to, it `yields control to a specific ULT `instead of the scheduler. Yield to is cheaper than yield because it bypasses the scheduler and eliminates the overhead of one context switch. Yield to can be used `only among ULTs associated with the same ES`. 
- **Migration**. Work units can be migrated between pools. 
- **Synchronizations**. `Mutex, condition variable, future, and barrier operations `are supported, but only for `ULTs`.

#### Implementation

- An `ES` is mapped to a `Pthread` and can be bound to a hardware processing element (e.g., CPU core or hardware thread).
- `Context switching` between ULTs can be achieved through various methods, such as `ucontext`, setjmp/longjmp with sigaltstack [21], or Boost library’s `fcontext` [22].
-  The `user context` includes CPU registers, a stack pointer, and an instruction pointer.
  - When a ULT is created, we create a `ULT context that contains a user context, a stack, the information for the function` that the ULT will execute, and `its argument`.
  - A stack for each ULT is dynamically allocated, and its size can be specified by the user. 
  - The ULT context also includes a pointer to the scheduler context in order to yield control to the scheduler or return to the scheduler upon completion
  - tasklet  contains` a function pointer, argument, and some bookkeeping information`, such as an associated pool or ES. Tasklets are `executed on the scheduler’s stack space`.
- A pool is a container data structure that can hold a set of work units and provides operations for insertion and deletion. 
- A scheduler is implemented similarly to a work unit; it has its own function (i.e., scheduling function) and a stack. 
- Argobots relies on cooperative scheduling of ULTs to improve resource utilization. , a ULT may voluntarily yield control when idle in order to allow the underlying ES to make progress on other work units.
- Argobots synchronization primitives, such as mutex locking or thread join operations, automatically yield control when blocking is inevitable.

#### High-level runtimes--colocated I/O services

- distributed I/O service daemons that are deployed alongside application processes.
  - This service model can be used to provide `dynamically provisioned`, `compute-node-funded services` [41], `in situ analysis and coupling services` [42], or `distributed access to on-node storage devices`
  - balance three competing goals: `programmability (i.e., ensuring that the service itself is easy to debug and maintain), performance for concurrent workloads, and minimal interference with colocated applications.`
- Unlike conventional OS-level threads, ULTs are inexpensive to create and consume minimal resources while waiting for a blocking I/O operation. Each ULT can cooperatively yield when appropriate so that other ULTs (i.e., concurrent requests) can make progress, thereby enabling a high degree of I/O operation concurrency with minimal resource consumption

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220730215322082.png)

- **abt-io**, provides thin wrappers for common POSIX I/O function calls such as open(), pwrite(), and close().
  - Internally, the wrappers `delegate blocking system calls to a separate Argobots pool `as shown in Figure 12. `The calling ULT is suspended while the I/O operation is in progress`, thereby allowing other service threads to make progress until the I/O operation completes.
  - by spawning a new tasklet that coordinates with the calling ULT via an eventual, an Argobots future-like synchronization construct.
  - 请求服务池和 I/O 系统调用服务池之间的这种职责划分可以被认为是 I/O 转发的一种形式，它允许独立供应 I/O 资源，而不会干扰主应用程序例程的执行。
  - If the `I/O resource provides a native asynchronous API `(such as the Mercury RPC library [44]), then one need not delegate operations to a dedicated pool; `the resource can use its normal completion notification mechanism to signal eventuals.`
- **abt-snoozer,** implements an I/O-aware Argobots scheduler that causes the ES to block (i.e., sleep) when no work units are eligible for execution and wake up when new work units are inserted
  - The scheduler can use the `epoll() system call to block`, and the pool can `write() to an eventfd() file descriptor to notify it when new work units are added.`
  -  The abt-snoozer library uses the `libev [45] event loop` and `asynchronous event watchers `to abstract this functionality for greater portability

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20220730215517985.png)

### Resource

- https://www.mcs.anl.gov/~aamer/papers/tpds17_argobots.pdf

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/argobots_paper/  

