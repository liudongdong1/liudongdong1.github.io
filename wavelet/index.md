# Wavelet


> - **Fourier Transform is the dot product between real signal and various frequency of sine wave**. We get a **stats of frequency but we don't know when that “frequency” happen, we lost the time resolution of the real signal**. 
> - To get both frequency and time resolution we can be **`dividing the original signal into several parts` and apply Fourier Transform to each part**. The problem of STFT is the choice of time windows;  low frequencies require large time windows and high frequencies need a short time windows to provide required resolution in time and frequency. 
> - **Wavelets come as a solution to the lack of Fourier Transform**.  we need **a bigger time window to catch low frequency and smaller window for higher frequency and That is the Idea of Wavelets.** 
> - simultaneous localization both in frequency spectrogram& time;
> - Sparsity of the representation: Many of the coefficients c(j,k) in a wavelet representation are either zero or very small
> - linear computational time complexity

### 1. Formula

> Where x is the real signal, ψ is an arbitrary mother wavelet, a is the scale and b is the translation (X is the processed signal of course). The scale **is the same as the size of the window.** 
>
> -  **Non zero magnitudes of the mother wavelet are “the window”**
>   -  **wavelet has many kinds of mother wavelet and you can define a new one** (with several requirements that need to satisfy of course)
>   - **Fourier Transform just has 1 kind of transformation but Wavelet Transform can have many kinds of transformation**
> - **The scale** is inversely proportional to the frequency of the mother wavelet (the window).
> - The approximation, or scaling, coefficients are the `lowpass representation of the signal `and `the details are the wavelet coefficients.` At each subsequent level,` the approximation coefficients` are divided into a coarser `approximation (lowpass) and highpass (detail) part`.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104151612.png)

![Scale](https://miro.medium.com/max/1600/1*QaiFGKiYrmkpe6Ihc9Xg0g.gif)

![Transition](https://miro.medium.com/max/1600/1*WywGvOeBt2-koSp4UqME3w.gif)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521093253737.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521093323523.png)

### 2. Wavelet Transform

- **Mother wavelet:** a wavelet function
  - characterizes the basic wavelet shape
  - be oscillatory and have a finite energy
- **Father wavelet:** a scaling function
  - characterizes the basic wavelet scale
  - allow to express needed details of the approximated function in the domain of interest;
- **Daughter wavelets:** all other derived wavelets
  - generating function( scale & shifts)

#### .1. **Continuous Wavelet Transform (CWT)**

> CWT is a Wavelet Transform where **we can set the scale and translation arbitrary**.

- Morlet Wavelet

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521095350793.png)

- Meyer Wavelet

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521095404161.png)

- Mexican Hat Wavelet

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521095418433.png)

#### .2. **Discrete Wavelet Transform (DWT)**

> DWT is a kind of wavelets that **restrict the value of scale and translation**. The restriction is like the scale is increasing in the power of 2 (a = 1, 2, 4, 8,…) and the translation is the integer (b = 1, 2, 3, 4, …). 

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521103558404.png)

- Haar Wavelet

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521095526979.png)

- Daubechies Wavelet

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521095535899.png)

##### .1. .1. Wavelet Denoising(Fast Wavelet Transform)

- perform a multilevel wavelet decomposition

  ![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521100657986.png)

  ![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104408077.png)

  ![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104423063.png)

  ![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521100854377.png)

- Identify a thresholding technique
  - the universal threshold
  - SureShrink or the rigrsure method
  - the Heursure method
  - the minimax method
- threshold and reconstruction
  - soft and hard thresholding

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521101137602.png)

##### .2. 2D discrete wavelet transform

![image-20210521104806366](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104806366.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104730761.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521104741237.png)

### 3. Application

#### .1. communication

> `handle non stationary behavior` and` to segregate information into uncorrelated segments`. This section provides a brief about various methodologies to extract benefits of wavelets into communication systems.

> `Wavelet based OFDM systems` provides improved Bit Error Rate (BER) and peak-to-average-power ratio (PAPR) performance compared to conventional OFDM system using FFT. Due to large number of sub band carriers, OFDM has high PAPR and makes it sensitive to nonlinear effects. Wavelet based system controls the PAPR ratio and results in improved performance compared to FFT based systems.

#### .2. data/image Detection

> - represent functions with discontinuities or corners (images)
> - some wavelets have discontinuities themselves or sharp corners in 2D cases.

> Bio medical signals are generally `one dimensional time series data (Electro Cardiogram- ECG, electroencephalogram -EEG)` or `an image (X ray, ultrasound scan, MRI)`. Accordingly a `1D or 2D wavelet transform` can be used to process the signal. Wavelet transform helps to `divide the signal to uncorrelated sub bands due to orthogonality property.` The transform coefficients or a part of it (say certain level coefficients) are used as feature for classifying the signal is a common methodology that can be adopted for a variety of applications. Recent advancement in `neural networks like CNN with wavelet coefficients as input features `opens up stage for a wide variety of research solutions. Another promising category of application is in signal preprocessing to` remove unwanted information in biomedical signals` [[9](https://www.intechopen.com/books/wavelet-theory/wavelet-theory-and-application-in-communication-and-signal-processing#B9)] `with thresholding techniques`. 

##### .1. EEG signal processing

>  selection of wavelets, level of transform, threshold calculations, selection of neural network of appropriate level and availability of data to train the network to achieve a desired accuracy.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210521105302276.png)

##### .2. cancer detection

> wavelet decomposition coefficients are used to extract features by calculating 2 level Haar wavelet transform and extract mean, standard deviation and energy of the transform coefficients as features for extraction of abnormal areas in image.

####  .4.  compression

> JPEG image compression technique is the standard technique used. It uses Discrete Cosine Transform (DCT) for the frequency domain conversion followed by Huffman coding.

### 4. Resource

- https://cn.mathworks.com/help/wavelet/gs/continuous-and-discrete-wavelet-transforms.html?requestedDomain=www.mathworks.com

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/wavelet/  

