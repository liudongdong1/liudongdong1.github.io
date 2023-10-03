# RandomFunction


#### 1. Random

> Random 使用的随机算法为 linear congruential pseudorandom number generator (LGC) 线性同余法伪随机数。在随机数生成时，随机算法的起源数字称为种子数（seed），在种子数的基础上进行一定的变换，从而产生需要的随机数字。Random 对象在种子数相同的情况下，相同次数生成的随机数是相同的。默认情况下 new Random() 使用的是`当前纳秒时间`作为种子数的。

##### .1. 基本使用

```java
// 生成 Random 对象
Random random = new Random();
for (int i = 0; i < 10; i++) {
    // 生成 0-9 随机整数
    int number = random.nextInt(10);
    System.out.println("生成随机数：" + number);
}
```

##### .2. 源码

> Random 底层使用的是 CAS（Compare and Swap，比较并替换）来解决线程安全问题的。
>
> CAS 在线程竞争比较激烈的场景中效率是非常低的**，原因是 CAS 对比时老有其他的线程在修改原来的值，所以导致 CAS 对比失败，所以它要一直循环来尝试进行 CAS 操作。**

```java
public Random() {
    this(seedUniquifier() ^ System.nanoTime());
}
public int nextInt() {
    return next(32);
}
protected int next(int bits) {
    long oldseed, nextseed;
    AtomicLong seed = this.seed;
    do {
        oldseed = seed.get();
        nextseed = (oldseed * multiplier + addend) & mask;
    } while (!seed.compareAndSet(oldseed, nextseed)); // CAS（Compare and Swap）生成随机数
    return (int)(nextseed >>> (48 - bits));
}
```

#### 2. ThreadLocalRandom

> ThreadLocalRandom 的实现原理与 ThreadLocal 类似，它相当于给每个线程一个自己的本地种子，从而就可以避免因多个线程竞争一个种子，而带来的额外性能开销了。
>
> **在多线程中就可以因为启动时间相同，而导致多个线程在每一步操作中都会生成相同的随机数**。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211202230246216.png)

##### .1. 基本使用

```java
// 得到 ThreadLocalRandom 对象
ThreadLocalRandom random = ThreadLocalRandom.current();
for (int i = 0; i < 10; i++) {
    // 生成 0-9 随机整数
    int number = random.nextInt(10);
    // 打印结果
    System.out.println("生成随机数：" + number);
}
```

##### .2.  源码

```java
public int nextInt(int bound) {
    // 参数效验
    if (bound <= 0)
        throw new IllegalArgumentException(BadBound);
    // 根据当前线程中种子计算新种子
    int r = mix32(nextSeed());
    int m = bound - 1;
    // 根据新种子和 bound 计算随机数
    if ((bound & m) == 0) // power of two
        r &= m;
    else { // reject over-represented candidates
        for (int u = r >>> 1;
             u + m - (r = u % bound) < 0;
             u = mix32(nextSeed()) >>> 1);
    }
    return r;
}

final long nextSeed() {
    Thread t; long r; // read and update per-thread seed
    // 获取当前线程中 threadLocalRandomSeed 变量，然后在种子的基础上累加 GAMMA 值作为新种子
    // 再使用 UNSAFE.putLong 将新种子存放到当前线程的 threadLocalRandomSeed 变量中
    UNSAFE.putLong(t = Thread.currentThread(), SEED,
                   r = UNSAFE.getLong(t, SEED) + GAMMA); 
    return r;
}
```

#### 3. SecureRandom

> SecureRandom 继承自 Random，该类提供加密强随机数生成器。**SecureRandom 不同于 Random，它收集了一些随机事件，比如鼠标点击，键盘点击等，SecureRandom 使用这些随机事件作为种子。这意味着，种子是不可预测的**，而不像 Random 默认使用系统当前时间的毫秒数作为种子，从而避免了生成相同随机数的可能性。
>
>  SecureRandom 默认支持两种加密算法：
>
> 1. SHA1PRNG 算法，提供者 sun.security.provider.SecureRandom；
> 2. NativePRNG 算法，提供者 sun.security.provider.NativePRNG。

##### .1. 基本使用

```
// 创建 SecureRandom 对象，并设置加密算法
SecureRandom random = SecureRandom.getInstance("SHA1PRNG");
for (int i = 0; i < 10; i++) {
    // 生成 0-9 随机整数
    int number = random.nextInt(10);
    // 打印结果
    System.out.println("生成随机数：" + number);
}
```

#### 4. Math

> Math 类诞生于 JDK 1.0，它里面包含了用于执行基本数学运算的属性和方法，如初等指数、对数、平方根和三角函数，当然它里面也包含了生成随机数的静态方法 `Math.random()` ，**此方法会产生一个 0 到 1 的 double 值**

```java
for (int i = 0; i < 10; i++) {
    // 产生随机数
    double number = Math.random();
    System.out.println("生成随机数：" + number);
}
```

#### Resource

- https://cloud.tencent.com/developer/article/1836767

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/randomfunction/  

