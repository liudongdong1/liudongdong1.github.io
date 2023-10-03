# KeywordSpotting


> Alvarez, Raziel, and Hyun-Jin Park. "End-to-end streaming keyword spotting." *ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019.
>

- keywords:  keyword

------

# Paper: keywrod spotting

<div align=center>
<br/>
<b>END-TO-END STREAMING KEYWORD SPOTTING
</b>
</div>


#### Summary

1. an efficient memorized neural network topology that aims at `making better use of the parameters and associated computations in  the DNN` by holding a memory of previous  activations distributed over the depth of DNN.
2. a method to train the DNN, to produce the keyword spotting score.
3. outperform in terms of quality of detection as well as size and computation.

#### Functions

- Efficient memorized neural network topology
  - the memory keeps each inference's state isolated from subsequent runs, just pushing new entries and popping old ones based on the memory size T configured for the layer.
  - by stacking the SVDF layers to extending the receptive field of the network.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201203171840369.png)

- Method to train end-to-end neural network
  - label generation
    - input sequence pairs$<X_t,c>$, $x_t$ is the 1D tensor corresponding to log-mel filter-band energies produced by  front end handle;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201203182041755.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201203182151232.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/keywordspoting/  

