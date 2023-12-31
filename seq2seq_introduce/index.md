# Seq2seq_Introduce


> A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items. 

The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the context). After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.

<video src="https://jalammar.github.io/images/seq2seq_3.mp4" width="100%" height="auto" loop="" autoplay="" controls="" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font: inherit;"></video>

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200715211325677.png)

## 1. RNN step

<video src="https://jalammar.github.io/images/RNN_1.mp4" width="100%" height="auto" loop="" autoplay="" controls="" __idm_id__="482845699" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font: inherit;"><div pseudo="-webkit-media-controls" class="sizing-small phase-ready state-playing" style="cursor: none;"><br class="Apple-interchange-newline"><div pseudo="-webkit-media-controls-overlay-enclosure"></div><div pseudo="-webkit-media-controls-enclosure"></div></div></video>

Let’s look at the hidden states for the encoder. Notice how the last hidden state is actually the context we pass along to the decoder.

<!-- <video src="https://jalammar.github.io/images/seq2seq_5.mp4" width="100%" height="auto" loop="" autoplay="" controls="" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-variant-numeric: inherit; font-variant-east-asian: inherit; font-weight: 400; font-stretch: inherit; font-size: 18px; line-height: inherit; font-family: Helvetica, Arial, sans-serif; color: rgb(34, 34, 34); letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-style: initial; text-decoration-color: initial;"></video> -->

## 2. Attention

> First, the encoder passes a lot more data to the decoder. Instead of passing the last hidden state of the encoding stage, the encoder passes *all* the hidden states to the decoder:

<video src="https://jalammar.github.io/images/seq2seq_7.mp4" width="100%" height="auto" loop="" autoplay="" controls="" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-variant-numeric: inherit; font-variant-east-asian: inherit; font-weight: 400; font-stretch: inherit; font-size: 18px; line-height: inherit; font-family: Helvetica, Arial, sans-serif; color: rgb(34, 34, 34); letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-style: initial; text-decoration-color: initial;"></video>

Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:

1. Look at the set of encoder hidden states it received – each encoder hidden states is most associated with a certain word in the input sentence
2. Give each hidden states a score
3. Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores

<!-- <video  src=" https://jalammar.github.io/images/attention_process.mp4"width="100%" height="auto" loop="" autoplay="" controls="" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-variant-numeric: inherit; font-variant-east-asian: inherit; font-weight: 400; font-stretch: inherit; font-size: 18px; line-height: inherit; font-family: Helvetica, Arial, sans-serif; color: rgb(34, 34, 34); letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-style: initial; text-decoration-color: initial;"><div pseudo="-webkit-media-controls" class="sizing-small phase-ready state-playing" style="cursor: none;"><br class="Apple-interchange-newline"><div pseudo="-webkit-media-controls-overlay-enclosure"></div><div pseudo="-webkit-media-controls-enclosure"></div></div></video> -->

> 1. The attention decoder RNN takes in the embedding of the <END> token, and an initial decoder hidden state.
> 2. The RNN processes its inputs, producing an output and a new hidden state vector (h4). The output is discarded.
> 3. Attention Step: We use the encoder hidden states and the h4 vector to calculate a context vector (C4) for this time step.
> 4. We concatenate h4 and C4 into one vector.
> 5. We pass this vector through a feedforward neural network (one trained jointly with the model).
> 6. The output of the feedforward neural networks indicates the output word of this time step.
> 7. Repeat for the next time steps

<!-- <video src="https://jalammar.github.io/images/attention_tensor_dance.mp4" width="100%" height="auto" loop="" autoplay="" controls="" style="box-sizing: border-box; display: inline-block; vertical-align: baseline; margin: 0px; padding: 0px; border: 0px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-variant-numeric: inherit; font-variant-east-asian: inherit; font-weight: 400; font-stretch: inherit; font-size: 18px; line-height: inherit; font-family: Helvetica, Arial, sans-serif; color: rgb(34, 34, 34); letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; background-color: rgb(255, 255, 255); text-decoration-style: initial; text-decoration-color: initial;"><div pseudo="-webkit-media-controls" class="sizing-small phase-ready state-stopped"><br class="Apple-interchange-newline"><div pseudo="-webkit-media-controls-overlay-enclosure"></div><div pseudo="-webkit-media-controls-enclosure"><div pseudo="-webkit-media-controls-panel"><div pseudo="-internal-media-controls-button-panel"><input type="button" pseudo="-webkit-media-controls-play-button" aria-label="播放" class="pause"><div aria-label="已播放时间：0:03" pseudo="-webkit-media-controls-current-time-display">0:03</div><div aria-label="总时间：/ 0:31" pseudo="-webkit-media-controls-time-remaining-display">/ 0:31</div><div pseudo="-internal-media-controls-button-spacer"></div><div pseudo="-webkit-media-controls-volume-control-container" class="closed"><div pseudo="-webkit-media-controls-volume-control-hover-background"></div><input type="button" pseudo="-webkit-media-controls-mute-button" aria-label="静音" disabled=""></div><input type="button" pseudo="-webkit-media-controls-fullscreen-button" aria-label="进入全屏模式"><input type="button" aria-label="显示更多媒体控件" title="更多选项" pseudo="-internal-media-controls-overflow-button"></div><input type="range" step="any" pseudo="-webkit-media-controls-timeline" max="31.033333" aria-label="视频时间进度条 0:03 / 0:31" aria-valuetext="已播放时间：0:03"></div></div></div></video> -->

- 原文学习链接：https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html  



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/seq2seq_introduce/  

