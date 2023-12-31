# TIDeviceCoding


> Since each of the samples are taken at equally spaced time intervals, we have a time log of what happened during the chirp. `Every sample is a complex number`, meaning we have captured some `magnitude of power as well as the phase` of the wave at that time. So, our object will theoretically appear as an increase in power in our samples. On the other hand, we can use the distinct phases of each sample to obtain distance. ` (Frame, chirp,virtual antennas(tx*rx),Datasample)`
>
> - Power - $P = \sqrt{I^2+Q^2}$
> - Phase - $\angle = \arctan{(\frac{I}{Q})}$
>
> ($I$​​​=Imaginary Component $Q$​​​=Real Component)
>
> - `rangeFFT （对一个chirp 的sample）`和`dopplerFFT（对一个Frame的不同chirp) `，`AzimuthRRF(对virtual antenna 维度)`

![chirp data](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901095219270.png)

### 1. Profile

#### .1. 配置参数

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901170138432.png)

##### .1. start frequency

##### .2. frequency slope

##### .3. Idle time

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901170333232.png)

##### .4. ADC start time(不懂)

- `Synthesizer PLL ramp-up settling time, which is a function of ramp slope`
- `HPF step response settling, which is a function of HPF corner frequencies`
- `IF/DFE LPF settling time, which is a function of DFE output mode and sampling rate`

##### .5. Ramp end time

`The ramp end time is the sum of (a) the ADC start time, (b) the ADC sampling time and (c) the excess ramping time at the end of the ramp.  `

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901170741261.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901170108543.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901164155446.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901164232154.png)

#### .2. procedure

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901171106832.png)

```python
# Imports
import numpy as np
import matplotlib.pyplot as plt
```

### 1. 距离计算

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901100023673.png)

#### 1. Range FFT

> The first step to `obtaining range is by performing an FFT across our ADC samples for a single chirp`. This unfortunately does not leave you with a single range. Instead, we obtain multiple "range bins". These are exactly what they sound like, bins that store the information for various ranges.  `对sample维度`

```python
# Read in chirp data
adc_samples = np.loadtxt('../assets/chirp.txt', dtype=np.complex_)
print("datainformation:",type(adc_samples),adc_samples.shape,"data[0]",adc_samples[0]) #<class 'numpy.ndarray'> (128,)data[0] (19+65375j)
# Manually cast to signed ints
adc_samples.real = adc_samples.real.astype(np.int16)
adc_samples.imag = adc_samples.imag.astype(np.int16)
print("adc_samples:",type(adc_samples),adc_samples.shape,"data[0]",adc_samples[0])
# Take a FFT across ADC samples
range_bins = np.fft.fft(adc_samples)  #<class 'numpy.ndarray'> (128,) data[0] (19-161j)
print("range_bins:",type(range_bins),range_bins.shape,"data[0]",range_bins[0])#(128,) data[0] (1212-87j)
# Plot the magnitudes of the range bins
plt.plot(np.abs(range_bins)) 
plt.xlabel('Range Bins')
plt.ylabel('Reflected Power')
plt.title('Interpreting a Single Chirp')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901095419698.png)

#### 2. 单位转化

##### .1. 原理推导

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211026185621170.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211026185933268.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211026190042611.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211026190139735.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901164424552.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901164659831.png)

> complex 2x and real smapling modes: IF=0.9*(ADC_sampling)/2

`the objects at some range bin index are indeed farther than the objects at the previous index` and closer than the objects in the next index. That's probably what you didn't want to hear however.

- $f = \frac{S2 d}{c}$ - The *IF signal* frequency produced by a single object at distance $d$ (where the object appears in the frequency spectrum after the range FFT) 
    - $f$ - Frequency
    - $S$ - Frequency slope of the signal emitted by the chirp
    - $d$ - Distance relative to the radar
    - $c$ - Speed of light <br> 
- $\Delta f > \frac{1}{T}$ - The minimum separation needed in the frequency spectrum to be resolved by the radar <br>
    - $T$ - Sampling period

Looking at the first equation and we can see there is a direct relationship between $f$ and $d$...

- $f = \frac{S2 d}{c} \Rightarrow \Delta f = \frac{S2 \Delta d}{c}$

So now we have two separate equations that define $\Delta f$. Substitution can be now used.

- $\frac{S2 \Delta d}{c} = \Delta f \gt \frac{1}{T}$
- $\frac{S2 \Delta d}{c} \gt \frac{1}{T}$

Finally, we can solve for $\Delta d$, or the range resolution we can achieve.

- $\Delta d \gt \frac{c}{2} \cdot \frac{1}{ST}$

Since we know $S$ is in some unit of frequency over time, we can simplify $ST$ to just $B$, or the bandwidth of chirp.

- $\Delta d > \frac{c}{2B}$

In other words,` the range resolution is only dependent on how large a bandwidth the chirp has`. Let's see what information we have to use to try and find this range resolution.

Not exactly what we wanted, but the only thing we're missing is our bandwidth $B$. We can still use these parameters to find bandwidth since it is just the span of frequency of the chirp. So, we just need to calculate how much of a frequency span the chirp takes. Ignoring converting units for now, this should be our equation:

- $B = S \cdot \frac{N}{F_s}$
    - $S$ - Frequency slope (frequency/time)
    - $N$ - Number of ADC samples (samples)
    - $F_s$ - Frequency at which we sample ADC samples (samples / time)

##### .2. 计算代码

```python
# Data sampling configuration
c = 3e8 # Speed of light (m/s)
sample_rate = 2500 # Rate at which the radar samples from ADC (ksps - kilosamples per second)
freq_slope = 60 # Frequency slope of the chirp (MHz/us)
adc_samples = 128 # Number of samples from a single chirpb

# Calculating bandwidth of the chirp, accounting for unit conversion
chirp_bandwidth = (freq_slope * 1e12 * adc_samples) / (sample_rate * 1e3)

# Using our derived equation for range resolution
range_res = c / (2 * chirp_bandwidth)
print(f'Range Resolution: {range_res} [meters]')

# Apply the range resolution factor to the range indices
ranges = np.arange(adc_samples) * range_res
powers = np.abs(range_bins)

# Now we can plot again with an x-axis that makes sense
plt.plot(ranges, powers)
plt.xlabel('Range (meters)')
plt.ylabel('Reflected Power')
plt.title('Interpreting a Single Chirp')
plt.show()

chirp_bandwidth
print(ranges)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901095734148.png)

### 2. 速度计算

#### .1. Doppler Effect

> In addition to `multiple ADC samples`, we have `multiple chirps.` For an object at range 𝑥 from the radar, when we receive the respective ADC sample, the product will be a complex number with some phase. If the object is moving away from the radar, the respective ADC sample of the second chirp will come in at a very slightly delayed time. This is because the object also moved slightly away in that miniscule amount of time. Althought this movement is miniscule, `the change in phase of the wave can be clearly seen`. `对一个frame chirp维度`

$$
f' = \frac{v+v_0}{v-v_s}f
$$

#### .2. Range FFT

```python
# Read in frame data
frame = np.load('../assets/simple_frame_1.npy')

# Manually cast to signed ints
frame.real = frame.real.astype(np.int16)
frame.imag = frame.imag.astype(np.int16)

# Meta data about the data
num_chirps = 128 # Number of chirps in the frame
num_samples = 128 # Number of ADC samples per chirp

range_plot = np.fft.fft(frame, axis=1)  # axis=1:按行计算
print("range infor:",type(range_plot),range_plot.shape)   #(128,128)
# Visualize Results
plt.imshow(np.abs(range_plot).T)
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Range')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901101058961.png)

#### .3. Doppler FFT

```python
# Take a sequential FFT across the chirps
range_doppler = np.fft.fft(range_plot, axis=0)

# FFT shift the values (explained later)
range_doppler = np.fft.fftshift(range_doppler, axes=0)

# Visualize the range-doppler plot
# plt.imshow(np.log(np.abs(range_doppler).T))
plt.imshow(np.abs(range_doppler).T)
plt.xlabel('Doppler Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()

plt.plot(np.abs(range_doppler))
plt.xlabel('Doppler Bins')
plt.ylabel('Signal Strength')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901101436884.png)

> That line in the `middle at doppler bin 64 is called zero doppler`, meaning everything along that line is `static/not moving relative to the radar`. This means everything to the `left (bins<64) is negative doppler, or moving towards the radar `and the opposite for the other half of the doppler bins.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028103918183.png)

- `rangeFFT （对一个chirp 的sample）`和`dopplerFFT（对一个Frame的不同chirp) `先后顺序影响不大；

```python
print("frame infor:",type(frame),frame.shape)  #frame infor: <class 'numpy.ndarray'> (128, 128)
# Range FFT -> Doppler FFT
range_bins = np.fft.fft(frame, axis=1)
fft_2d = np.fft.fft(range_bins, axis=0)

# Doppler FFT -> Range FFT
doppler_bins = np.fft.fft(frame, axis=0)
rfft_2d = np.fft.fft(doppler_bins, axis=1)

print('Max power difference: ', np.abs(fft_2d - rfft_2d).max())  #Max power difference:  5.64766185425834e-11
```

#### .4. 单位转化

##### .1. 原理推导

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028092818353.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028093021780.png)

![image-20211028093131613](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028093131613.png)

All the units of the data we produced are of some type of "bin". Similarly to range resolution, we `have a doppler resolution aka velocity resolution`. 

- $\omega = \frac{4\pi vT_c}{\lambda}$ - Rotational frequency of phasor due to object moving at $v$ velocity
    - $v$ - Velocity
    - $T_c$ - Sampling period
    - $\lambda$ - Wavelength
- $\Delta\omega \gt \frac{2\pi}{N}$ - Minimum change in rotation of phasor to be resolved by radar
    - $N$​ - Number of sample points

##### .2. 计算代码

```python
# Data sampling configuration
c = 3e8 # Speed of light (m/s)
sample_rate = 2500 # Rate at which the radar samples from ADC (ksps - kilosamples per second)
freq_slope = 60 # Frequency slope of the chirp (MHz/us)
adc_samples = 128 # Number of samples from a single chirp

start_freq = 77.4201 # Starting frequency of the chirp (GHz)
idle_time = 30 # Time before starting next chirp (us)
ramp_end_time = 62 # Time after sending each chirp (us)
num_chirps = 128 # Number of chirps per frame
num_tx = 2 # Number of transmitters

# Range resolution    ？？ 这里没有看懂
range_res = (c * sample_rate * 1e3) / (2 * freq_slope * 1e12 * adc_samples)
print(f'Range Resolution: {range_res} [meters/second]')  #Range Resolution: 0.048828125 [meters/second]

# Apply the range resolution factor to the range indices
ranges = np.arange(adc_samples) * range_res
```

```python
# Make sure your equation translates to the following
velocity_res = c / (2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_chirps * num_tx)
print(f'Velocity Resolution: {velocity_res} [meters/second]')

# Apply the velocity resolution factor to the doppler indicies
velocities = np.arange(num_chirps) - (num_chirps // 2)
velocities = velocities * velocity_res

powers = np.abs(range_doppler)

# Plot with units
plt.imshow(powers.T, extent=[velocities.min(), velocities.max(), ranges.max(), ranges.min()])
plt.xlabel('Velocity (meters per second)')
plt.ylabel('Range (meters)')
plt.show()

plt.plot(velocities, powers)
plt.xlabel('Velocity (meters per second)')
plt.ylabel('Reflected Power')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901105825831.png)

### 3. 角度计算

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901111120724.png)

#### .1. Range FFT

```python
# Read in frame data
frame = np.load('../assets/doppler_example_1.npy')

# Manually cast to signed ints
frame.real = frame.real.astype(np.int16)
frame.imag = frame.imag.astype(np.int16)

print(f'Shape of frame: {frame.shape}')

# Meta data about the data
num_chirps = 128 # Number of chirps in the frame
num_samples = 128 # Number of ADC samples per chirp

num_tx = 2
num_rx = 4
num_vx = num_tx * num_rx # Number of virtual antennas
```

```python
range_plot = np.fft.fft(frame, axis=2)

# Visualize Results
plt.imshow(np.abs(range_plot.sum(1)).T)
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Range')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901111228073.png)

#### .2. Doppler FFT

```python
range_doppler = np.fft.fft(range_plot, axis=0)
range_doppler = np.fft.fftshift(range_doppler, axes=0)

# Visualize Results
plt.imshow(np.log(np.abs(range_doppler).T).sum(1))
plt.xlabel('Doppler Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Doppler')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901111309395.png)

#### .3.  Azimuth FFT

```python
num_angle_bins = 64
padding = ((0,0), (0,num_angle_bins-range_doppler.shape[1]), (0,0))
range_azimuth = np.pad(range_doppler, padding, mode='constant')
range_azimuth = np.fft.fft(range_azimuth, axis=1)
range_azimuth = range_azimuth
# Visualize Results
plt.imshow(np.log(np.abs(range_azimuth).sum(0).T))
plt.xlabel('Azimuth (Angle) Bins')
plt.ylabel('Range Bins')
plt.title('Interpreting a Single Frame - Azimuth')
plt.show()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901111409348.png)

#### .4. 原理推导

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210901165336582.png)

### 4. Noise Removal

#### .1. 去除具有0的多普勒速度信号

> 0多普勒滤波的缺点是雷达不能检测到路径中的静止目标, 这将导致检测失败.

```python
def clutter_removal(input_val, axis=0):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.

    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.

    Returns:
        ndarray: Array with static clutter removed.

    """
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))   #[0,1,2,] 
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.transpose(reordering).mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering)
```

#### .2. **固定杂波阈值分割( fixed clutter thresholding)**

> 在固定阈值的情况下, 对阈值以下的信号进行剔除. 该方法在检测阈值设置过高的情况下, 会出现极少的虚警(false alarms), 但同时也会掩盖有效目标. 如果阈值设置得太低, 则会导致过多的错误警报. 如在下图中, 固定阈值导致虚警和漏检弱目标.
>
> - 虚警率(false alarm rate）是雷达通过噪声或其他干扰信号发现错误信号的速率. 它是在没有有效目标存在的情况下, 检测到雷达目标存在的一种度量.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028104240757.png)

#### .3. **动态阈值分割(dynamic thresholding)**

> 动态阈值分割通过改变阈值水平来降低误报率. 利用这种名为 **CFAR(Constant False Alarm Rate)**的技术, 可以`监测每一个或每一组距离多普勒bin的噪声`, 并将信号与本地的噪声水平进行比较. 此比较用于创建一个阈值, 该阈值为**CFAR**.
>
> - **CFAR** 根据车辆周围环境变化检测阈值. 通过实现恒定的虚警率, 可以解决虚警问题.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028104419947.png)

##### 1. Cell Averaging CFAR (CA-CFAR)

> **CA-CFAR**测量被测单元(CUT)两侧的训练单元的干扰程度. 然后用这个测量来决定目标是否在被测单元(CUT)中. 该过程遍历所有的距离多普勒单元, 并根据噪声估计确定目标的存在.

FFT bins是在通过多个啁啾的`Range Doppler FFT`生成的. **CA-CFAR**使用滑动窗口遍历整个FFT bins . 每个窗口由以下单元格组成：

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028105734872.png)

- **Cell Under Test**：通过`比较信号电平和噪声估计值(阈值)来检测目标是否存在的单元`.
- **Training Cells**：在训练单元上`测量噪声水平`. 训练单元可以分为两个区域, 滞后于CUT的单元称为`滞后训练单元,` 而`领先于CUT的单元称为前导训练单元`. 通过对训练单元下的噪声进行平均来估计噪声. 在某些情况下, 采用前导或滞后的噪声平均值, 而在其他情况下, 则合并前导和滞后的噪声平均值, 并考虑两者中较高的一个用于噪声水平估计. 训练单元的数量应根据环境确定. 如果交通场景繁忙, 则应使用较少的训练单元, 因为间隔较近的目标会影响噪声估计.
- **Guard Cells** ：紧邻CUT的单元被指定为保护单元. `保护单元的目的是避免目标信号泄漏到训练单元中`, 这可能会对噪声估计产生不利影响. 保护单元的数量应`根据目标信号从被测单元中泄漏出来的情况来确定. 如果目标反射很强, 它们通常会进入周围的单元`. 　
- **Threshold Factor (Offset)**：使用`偏移值来缩放噪声阈值`. 如果信号强度以对数形式定义, 则将此偏移值添加到平均噪声估计中, 否则相乘.

###### .1. 1d CA-CFAR

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028111153793.png)

1. 确定训练单元和保卫单元的数量
2. 开始在整个FFT 1D阵列上一次滑动一个单元格的窗口. 总窗口大小应为：2（T + G）+ CUT
3. 对于每一步, 将`所有前导或滞后训练单元内的信号（噪声）相加`
4. 对总和求平均值以确定噪声阈值
5. 使用适当的偏移量值缩放阈值
6. 在CUT中的被测信号从窗口起点T + G + 1开始的
7. 将5中测量的信号与4中测量的阈值进行比较
8. 如果在CUT中测量的信号电平小于所测量的阈值, 则将0值分配给CUT中的信号, 否则分配1.

```matlab
% Implement 1D CFAR using lagging cells on the given noise and target scenario.

% Close and delete all currently open figures
close all;

% Data_points
Ns = 1000;

% Generate random noise
s=abs(randn(Ns,1));

%Targets location. Assigning bin 100, 200, 300 and 700 as Targets with the amplitudes of 8, 9, 4, 11.
s([100 ,200, 350, 700])=[8 15 7 13];

%plot the output
plot(s);

% TODO: Apply CFAR to detect the targets by filtering the noise.

% 1. Define the following:
% 1a. Training Cells
T = 12;
% 1b. Guard Cells 
G = 4;

% Offset : Adding room above noise threshold for desired SNR 
offset=5;

% Vector to hold threshold values 
threshold_cfar = [];

%Vector to hold final signal after thresholding
signal_cfar = [];

% 2. Slide window across the signal length
for i = 1:(Ns-(G+T+1))     

    % 2. - 5. Determine the noise threshold by measuring it within the training cells

    noise_level =sum(s(i:i+T-1));

    % 6. Measuring the signal within the CUT

    threshold = (noise_level/T)*offset;
    threshold_cfar=[threshold_cfar,{threshold}];

    signal=s(i+T+G);

    % 8. Filter the signal above the threshold
    if (signal<threshold)
        signal=0;
    end
    signal_cfar = [signal_cfar, {signal}];
end
% plot the filtered signal
plot (cell2mat(signal_cfar),'g--');
% plot original sig, threshold and filtered signal within the same figure.
figure,plot(s);
hold on,plot(cell2mat(circshift(threshold_cfar,G)),'r--','LineWidth',2)
hold on, plot (cell2mat(circshift(signal_cfar,(T+G))),'g--','LineWidth',4);
legend('Signal','CFAR Threshold','detection')
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028110327085.png)

###### .2. 2D CA-CFAR

> 二维恒虚警类似于一维恒虚警, 但在`距离多普勒块的两个维度上都实现了`. 2D CA-CFAR包括训练单元,被测单元以及保护单元, 以防止目标信号对噪声估计的影响.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028110438197.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028111210885.png)

1. 确定每个维度Tr和Td的训练单元数. 同样, 选择保护单元格Gr和Gd的数量.
2. 在整个单元矩阵上滑动待测单元(CUT).
3. 选择总的包含训练, 保护和测试单元的网格. Grid Size = (2Tr+2Gr+1)(2Td+2Gd+1).
4. 保护区域和被测单元的总数：(2Gr+1)(2Gd+1).
5. 训练单元的总数为：(2Tr+2Gr+1)(2Td+2Gd+1) - (2Gr+1)(2Gd+1)
6. `测量并平均所有训练单元的噪声, 并获得阈值`.
7. `将偏移量（如果以dB为单位的信号强度）添加到阈值, 以将错误警报保持在最低水平.`
8. 确定`被测单元的信号电平.`
9. 如果CUT信号电平大于阈值, 则将值分配为1, 否则将其等于零.
10. 由于被测单元不位于边缘, 而训练单元占据了边缘, 我们将边缘抑制为零. 任何既不是1也不是0的单元格值, 为其分配一个０.

```matlab
% *%TODO* :
%Select the number of Training Cells in both the dimensions.

Tr=10;
Td=8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation

Gr=4;
Gd=4;

% *%TODO* :
% offset the threshold by SNR value in dB

off_set=1.4;
% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.

% Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
% CFAR

RDM = RDM/max(max(RDM)); % Normalizing

% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

%Slide the cell under test across the complete martix,to note: start point
%Tr+Td+1 and Td+Gd+1
for i = Tr+Gr+1:(Nr/2)-(Tr+Gr)
    for j = Td+Gd+1:(Nd)-(Td+Gd)
        %Create a vector to store noise_level for each iteration on training cells
        noise_level = zeros(1,1);
        %Step through each of bins and the surroundings of the CUT
        for p = i-(Tr+Gr) : i+(Tr+Gr)
            for q = j-(Td+Gd) : j+(Td+Gd)
                %Exclude the Guard cells and CUT cells
                if (abs(i-p) > Gr || abs(j-q) > Gd)
                    %Convert db to power
                    noise_level = noise_level + db2pow(RDM(p,q));
                end
            end
        end

        %Calculate threshould from noise average then add the offset
        threshold = pow2db(noise_level/(2*(Td+Gd+1)*2*(Tr+Gr+1)-(Gr*Gd)-1));
        %Add the SNR to the threshold
        threshold = threshold + off_set;
        %Measure the signal in Cell Under Test(CUT) and compare against
        CUT = RDM(i,j);

        if (CUT < threshold)
            RDM(i,j) = 0;
        else
            RDM(i,j) = 1;
        end

    end
end

RDM(RDM~=0 & RDM~=1) = 0;
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211028110714250.png)

##### 2. Ordered Statistics CFAR (OS CFAR)

##### 3. Maximum Minimum Statistic (MAMIS CFAR)

##### 4. multiple variants of CA-CFAR

### 5. 相关知识

#### 1. FFT

> numpy.fft.**fft**(x, n = 10) 和 scipy.fftpack.fft(x, n = 10)
>
> - 第一个参数x表示输入的序列，
> - 第二个参数n制定FFT的点数，n值如果没有的话，那么就默认输入序列的个数为FFT的点数
> - 两者虽然相同，但是scipy.fftpack.fft的效率更高，推荐优先使用。
>
> numpy和scipy中都有**fftshift**，用于将FFT变换之后的频谱显示范围从[0, N]变为：[-N/2, N/2-1](N为偶数)         或者[-(N-1)/2, (N-1)/2](N为奇数)
>
> **fftfreq：**在画频谱图的时候，要给出横坐标的数字频率；scipy.fftpack.fftfreq(n, d=1.0)
>
> - 第一个参数n是FFT的点数，一般取FFT之后的数据的长度（size）
> - 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs

#### 2. resource

- [设备连接](https://blog.csdn.net/qq_40603614/article/details/112706620)
- [噪音处理](https://zhuanlan.zhihu.com/p/128514763)
- [CFAR python code](https://zhuanlan.zhihu.com/p/344519720)
- [csdn matlab handle](https://blog.csdn.net/qq_35605018/article/details/108816709)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/tidevicecoding/  

