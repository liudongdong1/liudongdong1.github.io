# Non-maximumSuppression


**NMS:**

**Input:** A list of Proposal boxes B, corresponding confidence scores S and overlap threshold N.

**Output:** A list of filtered proposals D.

**Algorithm:**

1. Select the proposal with highest confidence score, remove it from B and add it to the final proposal list D. (Initially D is empty).
2. Now compare this proposal with all the proposals — calculate the IOU (Intersection over Union) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from B.
3. Again take the proposal with the highest confidence from the remaining proposals in B and remove it from B and add it to D.
4. Once again calculate the IOU of this proposal with all the proposals in B and eliminate the boxes which have high IOU than threshold.
5. This process is repeated until there are no more proposals left in B.

IOU calculation is actually used to measure the overlap between two proposals.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200819182123933.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200819183146527.png)

> <font color=red>**Soft-NMS**</font> The idea is very simple — **“instead of completely removing the proposals with high IOU and high confidence, reduce the confidences of the proposals proportional to IOU value”**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200819191158638.png)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/non-maximumsuppression/  

