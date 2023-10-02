# FFTNN


#### 1. DFT

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210606123257246.png)

> - **k**表示每N个样本的循环次数；
> - **N**表示信号的长度；
> - $x_n$: 表示信号在样本n处的值。
> - $y_k$: 是一个复值，它给出了信号x中频率为k的正弦信号的信息；从$y_k$: 我们可以计算正弦的振幅和相位。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210606123359667.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210606123409975.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210606123531609.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210606123539877.png)

-  用傅里叶权重计算傅里叶变换

```python

import matplotlib.pyplot as plt

y_real = y[:, :signal_length]
y_imag = y[:, signal_length:]
tvals = np.arange(signal_length).reshape([-1, 1])
freqs = np.arange(signal_length).reshape([1, -1])
arg_vals = 2 * np.pi * tvals * freqs / signal_length
sinusoids = (y_real * np.cos(arg_vals) - y_imag * np.sin(arg_vals)) / signal_length
reconstructed_signal = np.sum(sinusoids, axis=1)

print('rmse:', np.sqrt(np.mean((x - reconstructed_signal)**2)))
plt.subplot(2, 1, 1)
plt.plot(x[0,:])
plt.title('Original signal')
plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal)
plt.title('Signal reconstructed from sinusoids after DFT')
plt.tight_layout()
plt.show()
```

- 用FFT来训练神经网络学习离散傅里叶变换

```python
import tensorflow as tf
signal_length = 32
# Initialise weight vector to train:
W_learned = tf.Variable(np.random.random([signal_length, 2 * signal_length]) - 0.5)
# Expected weights, for comparison:
W_expected = create_fourier_weights(signal_length)
losses = []
rmses = []
for i in range(1000):
    # Generate a random signal each iteration:
    x = np.random.random([1, signal_length]) - 0.5
    # Compute the expected result using the FFT:
    fft = np.fft.fft(x)
    y_true = np.hstack([fft.real, fft.imag])
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x, W_learned)
        loss = tf.reduce_sum(tf.square(y_pred - y_true))
    # Train weights, via gradient descent:
    W_gradient = tape.gradient(loss, W_learned)    
    W_learned = tf.Variable(W_learned - 0.1 * W_gradient)
    losses.append(loss)
    rmses.append(np.sqrt(np.mean((W_learned - W_expected)**2)))
```

#### 2. 学习链接

- https://mp.weixin.qq.com/s/ONgOLzVX5I0TreV2yTNUTQ

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/fftnn/  

