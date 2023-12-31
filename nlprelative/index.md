# NLPRelative


> Semantic Parsing: aims to translate a natural languages sentence into its corresponding executable programming language, which relieves users from the burden of learning techniques behind the programming language.

## 1. Grammar-based 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200815111221673.png)

- **Recent Question as context**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200815111403109.png)

- **Precedent SQL as context**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200815111706047.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200815120255640.png)

**level**: 
**author**: Kelvin Guu
**date**: 2020
**keyword**:

- NLP

> Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). Realm: Retrieval-augmented language model pre-training. *arXiv preprint arXiv:2002.08909*.

------

# Paper: REALM

<div align=center>
<br/>
<b>REALM: Retrieval-Augmented Language Model Pre-Training</b>
</div>

#### Proble Statement

- models such as BERT, RoBERTa, T5 store a surprising amount of world kownledge,;
  - the learned world knowledge is stored implicitly in the parameters of the underlying neural network;
  - Storage&Database space is limited by the size of network;

previous work:

- **Language model pre-training:** the pre-trained model can be further trained for a downstream task of primary interest, leading to better generalization;
  - **Masked Language model:** predict the missing tokens in an input text message;
- **Open-domain question answer(OpenQA):**  unlike traditional reading comprehension tasks, QA doesn't receive a pre-identified document that is known to contain the answer; 
  - retrieval-based approach: given a question x, retrieve potentially relevant documents z and extract the answer;

#### Methods

- **Pipeline**
  - z: the retrieved document
  - x: the question;
  - y: the answer;

$$
 P(y|x)=\sum_{z\epsilon Z}P(y|z,x)P(z|x)					
$$



- **system overview**:



![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916084925834.png)

**【Module One】Knowledge Retriever**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916085649504.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916090002016.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916090015028.png)

**【Module Two】Knowledge-Augmented Encoder** : join x and z into a single sequence that we feed into a Transformer (distinct from the one used in the retriever).

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916090049408.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200916090253803.png)

#### Notes <font color=orange>去加强了解</font>

  - BERT methods



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/nlprelative/  

