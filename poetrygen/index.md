# PoetryGen


> AI和文学艺术不断交融，产生了很多有趣的研究方向，如`自动绘画生成`、`诗歌生成`、`音乐生成`、`小说生成`等。这些研究在学术界和普通人群中都引起了热烈的讨论，并且具有娱乐、教育、辅助文艺研究等广泛的应用价值。1.`中文古典诗歌(绝句、宋词等)生成`，2.`中文对联生成`，3.`中文现代诗生成`，4.`外文诗生成`，5.`多模态诗歌生成`, 6.`诗歌自动分析`, 7.`诗歌自动翻译`, 8. Demo及Survey 

### 1. 技术发展

#### .1. 传统方法

- **Word Salada（词语沙拉）**：是最早期的诗歌生成模型，被称作只是`简单将词语进行随机组合和堆砌而不考虑语义语法要求`。
- **基于模板和模式的方法**：`基于模板的方法类似于完形填空`，将一首现有诗歌挖去一些词，作为模板，再用一些其他词进行替换，产生新的诗歌。这种方法生成的诗歌在语法上有所提升，但是灵活性太差。因此后来出现了`基于模式的方法，通过对每个位置词的词性，韵律平仄进行限制`，来进行诗歌生成。
- **基于遗传算法的方法**：周昌乐等[1]提出并应用到宋词生成上。这里将`诗歌生成看成状态空间搜索问题`。先从随机诗句开始，然后借助人工定义的诗句评估函数，不断进行评估，进化的迭代，最终得到诗歌。这种方法在单句上有较好的结果，但是`句子之间缺乏语义连贯性`。
- **基于摘要生成的方法**：严睿等[2]将`诗歌生成看成给定写作意图的摘要生成问题`，同时加入了诗歌相关的一些`优化约束`。
- **基于统计机器翻译的方法**：MSRA的何晶和周明[3]将诗歌生成看成一个`机器翻译问题`，将`上一句看成源语言，下一句看成目标语言`，用统计机器翻译模型进行翻译，并加上平仄押韵等约束，得到下一句。通过不断重复这个过程，得到一首完整的诗歌。

#### .2. 深度学习

##### .1. RNNLM

> 基于RNN语言模型[1]的方法，将`诗歌的整体内容，作为训练语料`送给RNN语言模型进行训练。训练完成后，`先给定一些初始内容`，然后就可以按照语言模型`输出的概率分布进行采样得到下一个词`，不断重复这个过程就产生完整的诗歌。

##### .2. RNNPG

> 基于RNN语言模型[2]的方法，将诗歌的整体内容，作为训练语料送给RNN语言模型进行训练。训练完成后，先给定一些初始内容，然后就可以按照语言模型输出的概率分布进行采样得到下一个词，不断重复这个过程就产生完整的诗歌。

- **Convolutional Sentence Model（CSM）**：CNN模型，用于获取一句话的`向量表示`。
- **Recurrent Context Model (RCM)**：`句子级别的RNN`，根据历史生成句子的向量，输出下一个要生成句子的Context向量。
- **Recurrent Generation Model (RGM)**：`字符级别RNN`，根据RCM输出的Context向量和该句之前已经生成的字符，输出下一个字符的概率分布。解码的时候根据RGM模型输出的概率和语言模型概率加权以后，生成下一句诗歌，由人工规则保证押韵。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523103709542.png)

##### .3. Attention-base model

> 模型[3]是`基于attention的encoder-decoder框架`，将`历史已经生成的内容作为源语言`，将下一句话作为目标语言进行翻译。需要用户提供第一句话，然后由第一句生成第二句，第一，二句生成第三句，并不断重复这个过程，直到生成完整诗歌。基于Attention机制配合LSTM，可以学习更长的诗歌，同时在一定程度上，可以保证前后语义的连贯性。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523103833003.png)

##### .4. Planning based Neural Network PG

> 模型[5]不需要专家知识，是一个`端到端的模型`。它试图模仿人类开始写作前，先规划一个写作大纲的过程。整个诗歌生成框架由两部分组成：`规划模型和生成模型`。
>
> - **规划模型**：将代表用户写作意图的Query作为输入，生成一个写作大纲。`写作大纲是一个由主题词组成的序列`，第i个主题词代表第i句的主题。
> - **生成模型**：基于encoder-decoder框架。有两个encoder,`其中一个encoder将主题词作为输入`，`另外一个encoder将历史生成的句子拼在一起作为输入`，`由decoder生成下一句话`。decoder生成的时候，利用Attention机制，对主题词和历史生成内容的向量一起做打分，由模型来决定生成的过程中各部分的重要性。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523104102116.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523104130020.png)

##### .5. RNN with Iterative Polishing Shema

> 模型[4]基于encoder-decoder框架。`encoder阶段，用户提供一个Query作为自己的写作意图,由CNN模型获取Query的向量表示`。decoder阶段，使用了`hierarchical的RNN生成框架，由句子级别和词级别两个RNN组成。`
>
> - **句子级别RNN**：输入句子向量表示，`输出下一个句子的Context向量`。
> - **字符级别RNN**：`输入Context向量和历史生成字符`，输出`下一个字符的概率分布`。当一句生成结束的时候，字符级别RNN的最后一个向量，作为表示这个句子的向量，送给句子级别RNN。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523104417394.png)

##### .6. Generating Topical Poetry

> 模型[6]基于encoder-decoder框架，分为两步。先根据`用户输入的关键词得到每句话的最后一个词`，这些词都押韵且与用户输入相关。`再将这些押韵词作为一个序列，送给encoder,由decoder生成整个诗歌`。这种机制一方面保证了押韵，另外一方面，和之前提到的规划模型类似，在一定程度上避免了主题漂移问题。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523104620302.png)

##### .7. SeqGAN

> 模型[7]将图像中的`对抗生成网络`，用到文本生成上。`生成网络是一个RNN，直接生成整首诗歌`。而`判别网络是一个CNN。用于判断这首诗歌是人写的`，还是机器生成的，并通过强化学习的方式，将梯度回传给生成网络。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210523104730231.png)

##### .8. [GPT2-Chinese**](https://github.com/Morizeyao/GPT2-Chinese)

> 中文的GPT2训练代码，使用BERT的Tokenizer。可以写诗，新闻，小说，或是训练通用语言模型。支持字为单位或是分词模式。支持大语料训练。

### 2. [九歌团队](https://github.com/THUNLP-AIPoet/)

#### .1. 介绍

> “九歌”是清华大学自然语言处理与社会人文计算实验室（THUNLP）在负责人`孙茂松教授`带领下研发的中文诗歌自动生成系统。作为目前最好的中文诗歌生成系统之一，“九歌”曾于2017年登上央视一套大型科技类挑战节目《机智过人》第一季的舞台，与当代优秀青年诗人同台竞技比拼诗词创作。2017年上线至今，“九歌”已累计为用户创作超过1000万首诗词，并荣获全国计算语言学学术会议最佳系统展示奖(2017，2019)和最佳论文奖(2018)。

#### .3. 开源模型

- WMPoetry

> 基于Memory Network的诗歌生成模型。该模型支持多关键词输入，并将中文古典诗歌的格律拆解为字级别的格式embeding，能够较好地控制生成诗歌的格律和韵脚，并提升诗歌的上下文关联性和扣题程度。相关论文发表于IJCAI 2018。

- StylisticPoetry

> 基于互信息解耦的无监督风格诗歌生成模型。该模型无需任何标注数据，能够自动将生成的诗歌划分为用户指定的任意数量个不同风格。 相关论文发表于EMNLP 2018。

- MixPoet

> 基于对抗因素混合的半监督风格诗歌生成模型。该模型利用少量标注数据，通过组合不同的影响因素，创造出多种可控的诗歌风格。相关论文发表于AAAI 2020。

- 预训练资源BERT-CCPoem

> AIPoet基于超过90万首古诗文训练的BERT模型，该模型能提供任何一首古典诗词的任何一个句子的向量表示，可广泛应用于古典诗词智能检索与推荐、风格分析及情感计算等诸多下游任务。

#### .4. 开源数据集

- 中文古典诗歌数据集THU-CCPC：包含约13万首中文绝句(已划分训练、测试、开发集)，可用于相关模型的训练。
- 中文格律及韵律数据集THU-CRRD：包含整理好的平声字表、仄声字表以及平水韵表，可用于诗歌生成以及诗歌自动分析研究。
- 中文诗歌细粒度情感标注语料THU-FSPC：包含5,000首人工标注的绝句，每首诗包含诗歌整体以及每一句的情感标签。可用于训练情感可控的诗歌生成模型，以及进行诗歌情感自动分析。
- 中文诗歌质量标注数据集THU-PQED：包含173首古人诗作，每一首诗附有诗歌质量不同侧面(如通顺性、上下文连贯性等)的人工评分。可用于诗歌评价指标分析和研究。

> • 数据集共分为训练集、验证集及测试集三部分。
>
> • 训练集和验证集每行均代表一首完整的古诗，体裁为七言绝句（每句7 字，一共4 句）。
>
> • 测试集中的每行为一个样本，只有古诗的第一句话，要求模型能以古诗的所给的第一句为输入来生成剩余的三句。

### 3. 相关案例

- animalize / QuanTangshi *离线全唐诗 Android*
- justdark / pytorch-poetry-gen *a char-RNN based on pytorch*
- Clover27 / ancient-Chinese-poem-generator *Ancient-Chinese-Poem-Generator*
- chinese-poetry / poetry-calendar *诗词周历*
- chenyuntc / pytorch-book *简体唐诗生成(char-RNN), 可生成藏头诗,自定义诗歌意境,前缀等*
- okcy1016 / poetry-desktop *诗词桌面*
- huangjianke / weapp-poem *诗词墨客 小程序版*
- [风云三尺剑，花鸟一床书---对联数据集和自动对联机器人](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650408997&idx=1&sn=93395c083d85cf15490cf36cb5251a0f&scene=21#wechat_redirect)
- [自动对联活动获奖结果以及机器对联赏析](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650409170&idx=1&sn=852dba6972fd26e91f0c2fff9f458a4a&scene=21#wechat_redirect)
- ["自动作诗机"上线，代码和数据都是公开的](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410297&idx=1&sn=cda7099455083fbd412d0fdcb41acbea&scene=21#wechat_redirect)

### 4. 代码阅读

#### .1. TfversionRNN

- poetry.py

```python
import numpy as np
class Poetry:
    def __init__(self):
        self.poetry_file = 'poetry.txt'   #存储诗词文件
        self.poetry_list = self._get_poetry()
        self.poetry_vectors, self.word_to_int, self.int_to_word = self._gen_poetry_vectors()
        self.batch_size = 64
        self.chunk_size = len(self.poetry_vectors) // self.batch_size

    def _get_poetry(self):
        with open(self.poetry_file, "r", encoding='utf-8') as f:
            poetry_list = [line for line in f]
        return poetry_list

    def _gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetry_list)+' '))  #所有的words集合
        # 每一个字符分配一个索引 为后续诗词向量化做准备
        int_to_word = {i: word for i, word in enumerate(words)}
        word_to_int = {v: k for k, v in int_to_word.items()}
        to_int = lambda word: word_to_int.get(word)
        poetry_vectors = [list(map(to_int, poetry)) for poetry in self.poetry_list]
        return poetry_vectors, word_to_int, int_to_word

    def batch(self):
        # 生成器
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            batches = self.poetry_vectors[start:end]
            # 输入数据 按每块数据中诗句最大长度初始化数组，缺失数据补全
            x_batch = np.full((self.batch_size, max(map(len, batches))), self.word_to_int[' '], np.int32)
            for row in range(self.batch_size): x_batch[row, :len(batches[row])] = batches[row]
            # 标签数据 根据上一个字符预测下一个字符 所以这里y_batch数据应为x_batch数据向后移一位
            y_batch = np.copy(x_batch)
            y_batch[:, :-1], y_batch[:, -1] = x_batch[:, 1:], x_batch[:, 0]
            yield x_batch, y_batch
            start += self.batch_size
            end += self.batch_size

if __name__ == '__main__':
    data = Poetry().batch()
    for x, y in data:
        print(x)

```

- model

```python
import os
import datetime
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from poetry import Poetry
class PoetryModel:
    def __init__(self):
        # 诗歌生成
        self.poetry = Poetry()
        # 单个cell训练序列个数
        self.batch_size = self.poetry.batch_size
        # 所有出现字符的数量
        self.word_len = len(self.poetry.word_to_int)
        # 隐层的数量
        self.rnn_size = 128
    @staticmethod
    def embedding_variable(inputs, rnn_size, word_len):
        with tf.variable_scope('embedding'):
            # 这里选择使用cpu进行embedding
            with tf.device("/cpu:0"):
                # 默认使用'glorot_uniform_initializer'初始化，来自源码说明:
                # If initializer is `None` (the default), the default initializer passed in
                # the variable scope will be used. If that one is `None` too, a
                # `glorot_uniform_initializer` will be used.
                # 这里实际上是根据字符数量分别生成state_size长度的向量
                embedding = tf.get_variable('embedding', [word_len, rnn_size])
                # 根据inputs序列中每一个字符对应索引 在embedding中寻找对应向量,即字符转为连续向量:[字]==>[1]==>[0,1,0]
                lstm_inputs = tf.nn.embedding_lookup(embedding, inputs)
        return lstm_inputs
    @staticmethod
    def soft_max_variable(rnn_size, word_len):
        # 共享变量
        with tf.variable_scope('soft_max'):
            w = tf.get_variable("w", [rnn_size, word_len])
            b = tf.get_variable("b", [word_len])
        return w, b
    def rnn_graph(self, batch_size, rnn_size, word_len, lstm_inputs, keep_prob):
        # cell.state_size ==> 128
        # 基础cell 也可以选择其他基本cell类型
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        # 多层cell 前一层cell作为后一层cell的输入
        cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 2)
        # 初始状态生成(h0) 默认为0
        # initial_state.shape ==> (64, 128)
        initial_state = cell.zero_state(batch_size, tf.float32)
        # 使用dynamic_rnn自动进行时间维度推进 且 可以使用不同长度的时间维度
        # 因为我们使用的句子长度不一致
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state=initial_state)
        seq_output = tf.concat(lstm_outputs, 1)
        x = tf.reshape(seq_output, [-1, rnn_size])
        # softmax计算概率
        w, b = self.soft_max_variable(rnn_size, word_len)
        logits = tf.matmul(x, w) + b
        prediction = tf.nn.softmax(logits, name='predictions')
        return logits, prediction, initial_state, final_state
    @staticmethod
    def loss_graph(word_len, targets, logits):
        # 将y序列按序列值转为one_hot向量
        y_one_hot = tf.one_hot(targets, word_len)
        y_reshaped = tf.reshape(y_one_hot, [-1, word_len])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss
    @staticmethod
    def optimizer_graph(loss, learning_rate):
        grad_clip = 5
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer
    def train(self, epoch):
        # 输入句子长短不一致 用None自适应
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, None), name='inputs')
        # 输出为预测某个字后续字符 故输出也不一致
        targets = tf.placeholder(tf.int32, shape=(self.batch_size, None), name='targets')
        # 防止过拟合
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 将输入字符对应索引转化为变量
        lstm_inputs = self.embedding_variable(inputs, self.rnn_size, self.word_len)
        # rnn模型
        logits, _, initial_state, final_state = self.rnn_graph(self.batch_size, self.rnn_size, self.word_len, lstm_inputs, keep_prob)
        # 损失
        loss = self.loss_graph(self.word_len, targets, logits)
        # 优化
        learning_rate = tf.Variable(0.0, trainable=False)
        optimizer = self.optimizer_graph(loss, learning_rate)

        # 开始训练
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        new_state = sess.run(initial_state)
        for i in range(epoch):
            # 训练数据生成器
            batches = self.poetry.batch()
            # 随模型进行训练 降低学习率
            sess.run(tf.assign(learning_rate, 0.001 * (0.97 ** i)))
            for batch_x, batch_y in batches:
                feed = {inputs: batch_x, targets: batch_y, initial_state: new_state, keep_prob: 0.5}
                batch_loss, _, new_state = sess.run([loss, optimizer, final_state], feed_dict=feed)
                print(datetime.datetime.now().strftime('%c'), ' i:', i, 'step:', step, ' batch_loss:', batch_loss)
                step += 1
        model_path = os.getcwd() + os.sep + "poetry.model"
        saver.save(sess, model_path, global_step=step)
        sess.close()

    def gen(self, poem_len):
        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))
            return self.poetry.int_to_word[sample]

        # 输入
        # 句子长短不一致 用None自适应
        self.batch_size = 1
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, 1), name='inputs')
        # 防止过拟合
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lstm_inputs = self.embedding_variable(inputs, self.rnn_size, self.word_len)
        # rnn模型
        _, prediction, initial_state, final_state = self.rnn_graph(self.batch_size, self.rnn_size, self.word_len, lstm_inputs, keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            new_state = sess.run(initial_state)

            # 在所有字中随机选择一个作为开始
            x = np.zeros((1, 1))
            x[0, 0] = self.poetry.word_to_int[self.poetry.int_to_word[random.randint(1, self.word_len-1)]]
            feed = {inputs: x, initial_state: new_state, keep_prob: 1}

            predict, new_state = sess.run([prediction, final_state], feed_dict=feed)
            word = to_word(predict)
            poem = ''
            while len(poem) < poem_len:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = self.poetry.word_to_int[word]
                feed = {inputs: x, initial_state: new_state, keep_prob: 1}
                predict, new_state = sess.run([prediction, final_state], feed_dict=feed)
                word = to_word(predict)
            return poem
#train&Generate
from poetry_model import PoetryModel
if __name__ == '__main__':
    poetry = PoetryModel()
    poetry.train(epoch=20)
    
    poetry = PoetryModel()
    poem = poetry.gen(poem_len=100)
    print(poem)

```

#### .2. [Peoms_generator keras](https://github.com/youyuge34/Poems_generator_Keras) 178*

#### .3. [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese/blob/master/generate.py) start 4k

### Resource

[1] [Recurrent neural network based language model](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/47a8/7c2cbdd928bb081974d308b3d9cf678d257e.pdf)
[2] [Chinese Poetry Generation with Recurrent Neural Networks](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/D14-1074)
[3] [Chinese Song Iambics Generation with Neural Attention-based Model](https://link.zhihu.com/?target=http%3A//%5B1604.06274%5D%20Chinese%20Song%20Iambics%20Generation%20with%20Neural%20Attention-based%20Model)
[4] [i, Poet: Automatic Poetry Composition through Recurrent Neural Networks with Iterative Polishing Schema](https://link.zhihu.com/?target=https%3A//www.ijcai.org/Proceedings/16/Papers/319.pdf)
[5] [Chinese Poetry Generation with Planning based Neural Network](https://link.zhihu.com/?target=http%3A//%5B1610.09889%5D%20Chinese%20Poetry%20Generation%20with%20Planning%20based%20Neural%20Network)
[6] [Generating Topical Poetry](https://link.zhihu.com/?target=http%3A//xingshi.me/data/pdf/EMNLP2016poem-slides.pdf)
[7] [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://link.zhihu.com/?target=http%3A//Sequence%20Generative%20Adversarial%20Nets%20with%20Policy%20Gradient)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/poetrygen/  

