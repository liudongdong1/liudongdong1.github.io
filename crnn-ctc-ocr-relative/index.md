# CRNN+CTC OCR Relative


> 从自然场景图片中进行文字识别，需要包括2个步骤：文字检测：解决的问题是哪里有文字，文字的范围有多少;文字识别：对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域进转化为字符信息。

## 1. DAR&OCR&STR

- **Document Analysis& Recognition:** aims at the automatic extraction information presented on paper.
  - Documnet Image Enhancement; Layout Analysis; Character Recognition;
- **Optical Character Recognition:**
  - conversion of images of handwritten or printed text into machine-encoded text
  - the image may come from a scanned document, photo of a document, a scene-photo
- **Scene Text Recognition:** recognizing text in natural scene.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802085017178.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802085325293.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802085427659.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802090607760.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802095248267.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802095717531.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802095815004.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200802095916880.png)

- https://mp.weixin.qq.com/s/F1d_pZQoVeUd9Uy5Z0Hc1Q

## 1. 基于RNN文字识别框架

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200711220418971.png)

1. CNN+RNN+CTC(CRNN+CTC)
2. CNN+Seq2Seq+Attention

From: https://zhuanlan.zhihu.com/p/43534801

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/crnn-ctc-ocr-relative/  

