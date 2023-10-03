# CTC_Introduce


> Connectionist Temporal Classification, an algorithm used to train deep neural networks in speech recognition, handwriting recognition and other sequence problems.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712120505577.png)

## 1. Problem

- don't know the characters in the transcript align to the audio when having a dataset of audio clips and corresponding transcripts.
- people's rates of speech vary.
- hand-align takes lots of time.
- Speech recognition, handwriting recognition from images, sequences of pen strokes, action labeling in videos.

## 2. Question Define

> when mapping input sequences $X = [x_1, x_2, \ldots, x_T]$,, such as audio, to corresponding output sequences $Y = [y_1, y_2, \ldots, y_U]$, such as transcripts. We want to find an accurate mapping from $X's$ to $Y's$.

- Both $X$ and $Y$  can vary in length.
- The ratio of the lengths of $X$ and $Y$ can vary.
- we don't have an accurate alignment(correspondence of the elements) of $X$ and $Y$.

> The CTC algorithm, for a given $X$ it gives us an output distribution over all possible $Y's$, we can use this distribution either to infer a likely output or to assess the probability of a given output.

- **Loss Function:** maximize the probability it assigns to the right answer, compute the conditional probability $p(Y|X)$;
- **Inference:** infer a likely $Y$ given an $X$, $Y^*=argmaxp(Y|X)$；

## 3. Alignment

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712122146574.png)

- Often, it <font color=red>doesn’t make sense to force every input step to align to some output</font>. In speech recognition, for example, the input can have stretches of silence with no corresponding output.
- We <font color=red>have no way to produce outputs with multiple characters in a row</font>. Consider the alignment [h, h, e, l, l, l, o]. Collapsing repeats will produce “helo” instead of “hello”.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712122728313.png)

- the allowed alignments between  $X$ and $Y$ are monotonic
- the alignment of $X$ to $Y$ is many-to-one.
- the length of $Y$  cannot be greater than the length of  $X$.

## 4. Searching Methods

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712123313273.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712123742166.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712123934369.png)
$$
Z=[ϵ, y_1, ϵ, y_2, …, ϵ, y_U, ϵ]​
$$


- **Case 1:**  can’t jump over $z_{s-1}$, the previous token in $Z$.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712143004744.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712143018219.png)

- **Case 2:**  allowed to skip the previous token in $Z$.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712143150923.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712143205370.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712143315752.png)

- **Loss Function:**  for a training set D, the model's parameters are tuned to minimize the negative log-likelihood instead of maximizing the likelihood directly.

$$
\sum_{(X,Y)\epsilon D}-logP(Y|X)
$$

- **Inference:** (3) don't take into account the fact that a single output can have many alignments.

$$
Y^*=argmax_Yp(Y|X)\\
A^*=argmax_A\prod_{t=1}^Tp_t(a_t|X)
$$

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712144439025.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712144545657.png)

## 5. Properties of CTC

- **Conditional Independence**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200712145053749.png)

- **Alignment Properties**

> CTC only allows *monotonic* alignments. In problems such as speech recognition this may be a valid assumption. For other problems like machine translation where a future word in a target sentence can align to an earlier part of the source sentence, this assumption is a deal-breaker.

## 6. Usage

- Baidu Research has open-sourced [warp-ctc](https://github.com/baidu-research/warp-ctc). The package is written in C++ and CUDA. The CTC loss function runs on either the CPU or the GPU. Bindings are available for Torch, TensorFlow and [PyTorch](https://github.com/awni/warp-ctc).
- TensorFlow has built in [CTC loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss) and [CTC beam search](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder) functions for the CPU.
- Nvidia also provides a GPU implementation of CTC in [cuDNN](https://developer.nvidia.com/cudnn) versions 7 and up.

> to normalize the $\alpha$’s at each time-step to deal with CTC loss numerically unstable problem.
>
> A common question when using a beam search decoder is the size of the beam to use. There is a trade-off between accuracy and runtime.

**转载：**https://distill.pub/2017/ctc/



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/ctc_introduce/  

