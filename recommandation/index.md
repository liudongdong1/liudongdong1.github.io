# Recommandation


- [Recommendation Summary](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247503939&idx=3&sn=4b921c9715e1152ef30ab0005cd6bdbd&chksm=fbd1bce2cca635f48e03120686e4515aa77a6bdf0a8ff8611b69d35e994aaceb169116ff76e5&scene=126&sessionid=1607828429&key=3dc5cb16e3a2ef1c6211a76f1db9a9b97ced4bea2b66da02217b941b8900be173e28b0ff4ccca81e65486cae44ef072eb891080684b31311c63b0c244deda29914a92b915f5542781fedccd1167a716403c4ba2b1197efe60af36909d0bf8a0f20db36e59436d8d950f9400f61dbc6a0c6da2c3cb9455d51cb90401f75b406f1&ascene=1&uin=MzE0ODMxOTQzMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A8%2FPXJOzUJbPO2elsS19WT4%3D&pass_ticket=w%2Fpc6C5KDtVj%2Beh2vLjGFeKNhX9PO7R%2BDceH7UrCSuY6uEGbKjF5cq30Ri5W20h2&wx_header=0)

- User 数据（用户的`基础属性数据，如性别、年龄、关系链、兴趣偏好`等）
- - 对于用户`兴趣偏好`，一般简单地采用`文本 embedding` 方法来得到各标签的 embedding 向量，然后根据`用户对个标签的偏好程度做向量加权`；
  - 对于`关系链数据`（如同玩好友、游戏中心相互关注等），构造`用户关系图`，然后  采用基于`图的 embedding` 方法来得到用户的 embedding 向量；
- Item 数据（Item 基本信息数据，如标题、作者、游戏简介、标签等）
- - 对于文本、简介和标签等可以采用基于文本的 embedding 方法来`在已有语料上预训练模型`，然后得到对应的 embedding 向量（如 word2vec 或者 BERT）；
  - 此外对于有`明确关系的（如 item->文本->标签 or 关键词）`可以采用对关键词/标签的向量均值来表示 item 的文本向量（这里安利一下 FaceBook 开源的`StarSpace`）；
- User 行为数据（用户在场景中的行为数据，如点击、互动、下载等）
- - 针对用户对` Item 的操作`	（如点击、互动、下载）构造用户->item+Item 标签体系，构造用户-item-tag 的异构网络，然后可以采用 Metapath2vec 来得到各节点的 embedding 向量；
  - 通过记录用户在整个场景访问 item，构造 Item-Item 关系图，然后采用 DeepWalk 算法得到 item 的向量，用来挖掘 Item 间的关系特征；
- 额外数据（外部扩充数据，如用户游戏行为、用户微信其他场景活跃等）
- - 标签型（主要是用户在各场景的兴趣偏好）：
  - 关系链型（如游戏中心好友、游戏内好友、开黑好友）可以采用用户关系构造用户关系图，采用 Graph embedding 方法（如 GraphSAGE）来表示用户抽象特征

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201216095340156.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/recommandation/  

