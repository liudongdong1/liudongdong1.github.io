# PytorchGNN


> 论文对GNN模型分类如下：
>
> + `图卷积网络(Graph convolutional networks)`和`图注意力网络(graph attention networks)`，因为涉及到传播步骤(propagation step)。
> + `图的空域网络(spatial-temporal networks)`，因为该模型通常用在动态图(dynamic graph)上。
> + 图的自编码(auto-encoder)，因为该模型通常使用无监督学习(unsupervised)的方式。
> + `图生成网络(generative networks)`，因为是生成式网络。

$$
\mathbf{h}_{v}=f\left(\mathbf{x}_{v}, \mathbf{x}_{c o[v]}, \mathbf{h}_{n e[v]}, \mathbf{x}_{n e[v]}\right)\label{eq:1}
$$

$$
\mathbf{o}_{v}=g\left(\mathbf{h}_{v}, \mathbf{x}_{v}\right)
$$

其中，$\mathbf{x}_{v}$，$\mathbf{x}_{c o[v]}$，$\mathbf{h}_{n e[v]}$，$\mathbf{x}_{n e[v]}$分别表示节点$v$的特征向量，节点$v$边的特征向量，节点$v$邻居节点的状态向量和节点$v$邻居节点特征向量。

假设将所有的状态向量，所有的输出向量，所有的特征向量叠加起来分别使用矩阵$\mathbf{H}$，$\mathbf{O}$，$ \mathbf{X}$和 $\mathbf{X}_{N}$来表示，那么可以得到更加紧凑的表示：
$$
\mathbf{H}=F(\mathbf{H}, \mathbf{X})\label{eq:3}
$$

$$
\mathbf{O}=G\left(\mathbf{H}, \mathbf{X}_{N}\right)
$$

其中，$F$表示全局转化函数(global transition function)，$G$表示全局输出函数(global output function)，分别是所有节点$f$和$g$的叠加形式

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/graph_type.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/propa_step.png)

不同类别模型的Aggregator计算方法和Updater计算方法如下表

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/gnn_table.png)

> Fey M, Lenssen J E. Fast graph representation learning with PyTorch Geometric[J]. arXiv preprint arXiv:1903.02428, 2019. [[pdf](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/1903.02428&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=8986807541681358909&ei=M1LMYND3EsSsywSBmKRw&scisig=AAGBfm2Raynm1DnoD_UxQ8L7vr2Nf8M3xQ)]
>
> Rozemberczki B, Scherer P, He Y, et al. PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models[J]. arXiv preprint arXiv:2104.07788, 2021. [geometrictemporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/table.png)

### 1. Structure

> provide easy to use data iterators which are parametrized with spatiotemporal data. These iterators can serve snapshots which are formed by a single graph or multiple graphs which are batched together with the block diagonal batching trick.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210617111513969.png)

- **Temporal signal iterators**
  - `StaticGraphTemporalSignal` - Is designed for **temporal signals** defined on a **static** graph.
  - `DynamicGraphTemporalSignal` - Is designed for **temporal signals** defined on a **dynamic** graph.
  - `DynamicGraphStaticSignal` - Is designed for **static signals** defined on a **dynamic** graph.

- **Temporal Data Snapshots**
  - `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
  - `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
  - `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
  - `data.y`: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`
  - `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618090628020.png)

```python
#创建了一个新的Data
#方式一
import torch
from torch_geometric.data import Data
x = torch.tensor([[2,1],[5,6],[3,7],[12,0]],dtype=torch.float)
y = torch.tensor([[0,2,1,0,3],[3,1,0,1,2]],dtype=torch.long)
edge_index = torch.tensor([[0,1,2,0,3],
                          [1,0,1,3,2]],dtype=torch,long)
data = Data(x=x,y=y,edge_index=edge_index)
#方式二：
import torch
from torch_geometric.data import Data
x = torch.tensor([[2,1],[5,6],[3,7],[12,0]],dtype=torch.float)
y = torch.tensor([[0,2,1,0,3],[3,1,0,1,2]],dtype=torch.long)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [2, 1],
                           [0, 3]
                           [2, 3]], dtype=torch.long)
data = Data(x=x,y=y,edge_index=edge_index.contiguous())


loader = DataLoader(dataset, batch_size=512, shuffle=True)
Batch(x=[1024, 21], edge_index=[2, 1568], y=[512], batch=[1024])
```

- **Train-Test Splitting** && **Integrated Benchmark Dataset Loaders**

### 2. [Dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)

#### .1. offered dataset

- [Hungarian Chickenpox Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.chickenpox.ChickenpoxDatasetLoader)
- [PedalMe London Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.pedalme.PedalMeDatasetLoader)
- [Wikipedia Vital Math Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.wikimath.WikiMathsDatasetLoader)
- [Windmill Output Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.windmill.WindmillOutputDatasetLoader)
- [Pems Bay Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.pems_bay.PemsBayDatasetLoader)
- [Metr LA Dataset.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.metr_la.METRLADatasetLoader)
- [England COVID 19.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.encovid.EnglandCovidDatasetLoader)
- [Twitter Tennis.](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.twitter_tennis.TwitterTennisDatasetLoader)

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    batch
>>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])
    batch.num_graphs
>>> 32
```

```python
#dataset split
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
dataset = dataset.shuffle()   #shuffle dataset
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
```

- **Mini-Batch**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618092943939.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618093025169.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618160343566.png)

##### 1. Planetoid 类实例化流程

```python
dataset = Planetoid(root='dataset/PlanetoidPubMed',transform=NormalizeFeatures())
data = dataset[0].to(device) #这一步才执行transform的函数
```

1. 首先，检查数据原始文件是否已下载：

   - 检查`self.raw_dir`目录下是否存在`raw_file_names()`属性方法返回的每个文件，
   - 如有文件不存在，则调用`download()`方法执行原始文件下载。

2. 其次，检查数据是否经过处理：

   - 首先，检查之前对数据做变换的方法：检查

     ```
     self.processed_dir
     ```

     目录下是否存在

     ```
     pre_transform.pt
     ```

     文件：

     - 如果存在，意味着之前进行过数据变换，接着需要加载该文件，以获取之前所用的数据变换的方法，并检查它与当前

       ```
       pre_transform
       ```

       参数指定的方法是否相同，

       - 如果不相同则会报出一个警告，“The pre_transform argument differs from the one used in ……”。

   - 其次，检查之前的样本过滤的方法：检查

     ```
     self.processed_dir
     ```

     目录下是否存在

     ```
     pre_filter.pt
     ```

     文件：

     - 如果存在，则加载该文件并获取之前所用的样本过滤的方法，并检查它与当前

       ```
       pre_filter
       ```

       参数指定的方法是否相同，

       - 如果不相同则会报出一个警告，“The pre_filter argument differs from the one used in ……”。

   - 接着，检查是否存在处理好的数据：检查

     ```
     self.processed_dir
     ```

     目录下是否存在

     ```
     self.processed_file_names
     ```

     属性方法返回的所有文件，如有文件不存在，则需要执行以下的操作：

     - 调用`process()`方法，进行数据处理。

     - 如果`pre_transform`参数不为`None`，则调用`pre_transform()`函数进行数据处理。

     - 如果`pre_filter`参数不为`None`，则进行样本过滤（此例子中不需要进行样本过滤，`pre_filter`参数为`None`）。

     - 保存处理好的数据到文件，文件存储在

       processed_paths()

       属性方法返回的文件路径。如果将数据保存到多个文件中，则返回的路径有多个。

       - **`processed_paths()`属性方法是在基类（[DataSet](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html?highlight=dataset#torch_geometric.data.Dataset)）中定义的**，它对`self.processed_dir`文件夹与`processed_file_names()`属性方法的返回每一个文件名做拼接，然后返回。

     - 最后保存新的`pre_transform.pt`文件和`pre_filter.pt`文件，它们分别存储当前使用的数据处理方法和样本过滤方法。

3. 保证有预处理的文件后，在`self.data, self.slices = torch.load(self.processed_paths[0])`时从预处理文件路径中加载预处理后的数据。

4. 在执行`data = dataset[0]`时才调用选择的`transform`函数。

#### .2. customed dataset

> PyG提供两种不同的数据集类：Dataset,InMemoryDataset ,InMemoryDataset继承Dataset, 如果要继承InMemoryDataset 需要实现以下几个类
>
> - [`torch_geometric.data.InMemoryDataset.raw_file_names()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.raw_file_names): A list of files in the `raw_dir` which needs to be found in order to skip the download.
> - [`torch_geometric.data.InMemoryDataset.processed_file_names()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.processed_file_names): A list of files in the `processed_dir` which needs to be found in order to skip the processing.
> - [`torch_geometric.data.InMemoryDataset.download()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.download): Downloads raw data into `raw_dir`.
> - [`torch_geometric.data.InMemoryDataset.process()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset.process): Processes raw data and saves it into the `processed_dir`.

- `root`：字符串类型，存储数据集的文件夹的路径下。该文件夹下有两个文件夹：
  - 一个文件夹为记录在**`raw_dir`**，它用于存储未处理的文件，从网络上下载的**数据集原始文件**会被存放到这里；
  - 另一个文件夹记录在**`processed_dir`**，**处理后的数据**被保存到这里，以后从此文件夹下加载文件即可获得`Data`对象。
  - 注：`raw_dir`和`processed_dir`是属性方法，我们可以自定义要使用的文件夹。
- `transform`：函数类型，一个数据转换函数，它接收一个`Data`对象并返回一个转换后的`Data`对象。**此函数在每一次数据获取过程中都会被执行**。获取数据的函数首先使用此函数对`Data`对象做转换，然后才返回数据。此函数应该用于数据增广（Data Augmentation）。该参数默认值为`None`，表示不对数据做转换。
- `pre_transform`：函数类型，一个数据转换函数，它接收一个`Data`对象并返回一个转换后的`Data`对象。**此函数在`Data`对象被保存到文件前调用**。因此它应该用于只执行一次的数据预处理。该参数默认值为`None`，表示不做数据预处理。
- `pre_filter`：函数类型，**一个检查数据是否要保留的函数**，它接收一个`Data`对象，返回此`Data`对象是否应该被包含在最终的数据集中。此函数也在[`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象被保存到文件前调用。该参数默认值为`None`，表示不做数据检查，保留所有的数据。
- raw_file_names(): 属性方法，返回一个**数据集原始文件**的文件名列表，**数据集原始文件应该能在`raw_dir`文件夹中找到**，否则调用`download()`函数下载文件到`raw_dir`文件夹。
- processed_file_names: 属性方法，返回一个存储**处理过的数据的文件**的文件名列表，存储处理过的数据的文件应该能在`processed_dir`文件夹中找到，否则调用`process()`函数对样本做处理，然后保存处理过的数据到`processed_dir`文件夹下的文件里。
- download: 根据定义的`url`属性**下载数据集原始文件**到`raw_dir`文件夹。
- processed: **调用读取数据函数，将数据包装成Data**，然后**处理数据**，保存处理好的数据到`processed_dir`文件夹下的文件。
- raw_dir: 属性方法，原始数据存储的文件夹路径，我们可以自定义要使用的文件夹。
- processed_dir: 属性方法，处理后数据存储的文件夹路径，我们可以自定义要使用的文件夹。

```python
import torch
from torch_geometric.data import InMemoryDataset

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
#它返回一个包含没有处理的数据的名字的list。如果你只有一个文件，那么它返回的list将只包含一个元素。事实上，你可以返回一个空list，然后确定你的文件在后面的函数process()中。
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    #它返回一个包含所有处理过的数据的list。在调用process()这个函数后，通常返回的list只有一个元素，它只保存已经处理过的数据的名字。
    @property
    def processed_file_names(self):
        return ['data.pt']

    #下载数据到你正在工作的目录中，你可以在self.raw_dir中指定。
    def download(self):
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

> - `torch_geometric.data.Dataset.len()`: Returns the number of examples in your dataset.
> - [`torch_geometric.data.Dataset.get()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset.get): Implements the logic to load a single graph.

```python
import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
```

#### .3. [Transformer](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html)

> Transforms can be chained together using [`torch_geometric. transforms. Compose`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Compose) and are applied before saving a processed dataset on disk (`pre_transform`) or before accessing a graph in a dataset (`transform`).

```python
#convert the point cloud dataset into a graph dataset by generating nearest neighbor graphs from the point clouds via transform
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6))
dataset[0]
>>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
```

| [`Compose`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Compose) | Composes several transforms together.                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`ToSparseTensor`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToSparseTensor) | Converts the `edge_index` attribute of a data object into a (transposed) `torch_sparse.SparseTensor` type with key `adj_.t`. |
| [`ToUndirected`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToUndirected) | Converts the graph to an undirected graph, so that (j,i)∈E(j,i)∈E for every edge (i,j)∈E(i,j)∈E. |
| [`Constant`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Constant) | Adds a constant value to each node feature.                  |
| [`Distance`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Distance) | Saves the Euclidean distance of linked nodes in its edge attributes. |
| [`Cartesian`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Cartesian) | Saves the relative Cartesian coordinates of linked nodes in its edge attributes. |
| [`LocalCartesian`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.LocalCartesian) | Saves the relative Cartesian coordinates of linked nodes in its edge attributes. |
| [`Polar`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Polar) | Saves the `polar coordinates` of linked nodes in its edge attributes. |
| [`Spherical`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Spherical) | Saves the spherical coordinates of linked nodes in its edge attributes. |
| [`PointPairFeatures`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.PointPairFeatures) | Computes the rotation-invariant Point Pair Features          |
| [`OneHotDegree`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.OneHotDegree) | Adds the node degree as one hot encodings to the node features. |
| [`TargetIndegree`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.TargetIndegree) | Saves the globally normalized degree of target nodes         |
| [`LocalDegreeProfile`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.LocalDegreeProfile) | Appends the Local Degree Profile (LDP) from the [“A Simple yet Effective Baseline for Non-attribute Graph Classification”](https://arxiv.org/abs/1811.03508) paper |
| [`Center`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Center) | Centers node positions around the origin.                    |
| [`NormalizeRotation`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.NormalizeRotation) | `Rotates all points according to the eigenvectors of the point cloud.` |
| [`NormalizeScale`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.NormalizeScale) | Centers and normalizes node positions to the interval (−1,1)(−1,1). |
| [`RandomTranslate`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomTranslate) | Translates node positions by randomly sampled translation values within a given interval. |
| [`RandomFlip`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomFlip) | Flips node positions along a given axis randomly with a given probability. |
| [`LinearTransformation`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.LinearTransformation) | Transforms node positions with a square transformation matrix computed offline. |
| [`RandomScale`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomScale) | Scales node positions by a randomly sampled factor ss within a given interval, *e.g.*, resulting in the transformation matrix |
| [`RandomRotate`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomRotate) | `Rotates node positions around a specific axis by a randomly sampled factor within a given interval.` |
| [`RandomShear`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomShear) | Shears node positions by randomly sampled factors ss within a given interval, *e.g.*, resulting in the transformation matrix |
| [`NormalizeFeatures`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.NormalizeFeatures) | Row-normalizes node features to sum-up to one.               |
| [`AddSelfLoops`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.AddSelfLoops) | Adds self-loops to edge indices.                             |
| [`RemoveIsolatedNodes`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RemoveIsolatedNodes) | Removes isolated nodes from the graph.                       |
| [`KNNGraph`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.KNNGraph) | Creates a` k-NN graph` based on node positions `pos`.        |
| [`RadiusGraph`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RadiusGraph) | Creates edges based on node positions `pos` to all points within a given distance. |
| [`FaceToEdge`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.FaceToEdge) | Converts` mesh faces` `[3, num_faces]` to edge indices `[2, num_edges]`. |
| [`SamplePoints`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SamplePoints) | Uniformly samples `num` points on the mesh faces according to their face area. |
| [`FixedPoints`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.FixedPoints) | Samples a fixed number of `num` points and features from a point cloud. |
| [`ToDense`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToDense) | Converts a sparse adjacency matrix to a dense adjacency matrix with shape `[num_nodes, num_nodes, *]`. |
| [`TwoHop`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.TwoHop) | Adds the two hop edges to the edge indices.                  |
| [`LineGraph`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.LineGraph) | Converts a graph to its corresponding line-graph:            |
| [`LaplacianLambdaMax`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.LaplacianLambdaMax) | Computes the highest eigenvalue of the graph Laplacian given by [`torch_geometric.utils.get_laplacian()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_laplacian). |
| [`GenerateMeshNormals`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GenerateMeshNormals) | Generate normal vectors for each mesh node based on neighboring faces. |
| [`Delaunay`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Delaunay) | Computes the delaunay triangulation of a set of points.      |
| [`ToSLIC`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToSLIC) | Converts an image to a superpixel representation using the `skimage.segmentation.slic()` algorithm, resulting in a [`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) object holding the centroids of superpixels in `pos` and their mean color in `x`. |
| [`GDC`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GDC) | Processes the graph via Graph Diffusion Convolution (GDC) from the [“Diffusion Improves Graph Learning”](https://www.kdd.in.tum.de/gdc) paper. |
| [`SIGN`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SIGN) | The Scalable Inception Graph Neural Network module (SIGN) from the [“SIGN: Scalable Inception Graph Neural Networks”](https://arxiv.org/abs/2004.11198) paper, which precomputes the fixed representations |
| [`GridSampling`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GridSampling) | Clusters points into voxels with size `size`.                |
| [`GCNNorm`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GCNNorm) | Applies the GCN normalization from the [“Semi-supervised Classification with Graph Convolutional Networks”](https://arxiv.org/abs/1609.02907) paper. |
| [`AddTrainValTestMask`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.AddTrainValTestMask) | Adds a node-level random split via `train_mask`, `val_mask` and `test_mask` attributes to the `data` object. |

### 3. Models

#### .1. MLs

- **Temporal Deep learning:** 
  - `LSTM or GRU` generates in-memory representations of data points which are iteratively updated as it learns by new snapshots;
  - `attention mechanism`: to learn representation of the data points which are adaptively recontextualized based on the temporal history.
- **Static Graph Representation Learning: ** 
  - `message passing formalism`: learning representations of vertices, edges, and whole graphs with GNN.
  - models are differentiated by assumptions about the input graph ( eg. node heterogeneity, multiplexity, presence of edge attributes ), message compression function, propagation scheme, message aggregation function.
- **Spatio-temporal Deep Learning:**  combine temporal deep learning technique and graph representation learning.
- **Predictive Perfromance:** 
  - `Incremental:` the loss is back-propagated and model wights are updated after each temporal snapshot;
  - `Cumulative:` aggregated loss from every temporal snapshot and update weights with optimizer per epoch.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210617112633539.png)

- [Convolutional Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers)
- [Dense Convolutional Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers)
- [Normalization Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#normalization-layers)
- [Global Pooling Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers)
- [Pooling Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers)
- [Dense Pooling Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-pooling-layers)
- [Unpooling Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#unpooling-layers)
- [Models](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models)
- [Functional](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#functional)
- [DataParallel Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.data_parallel)

#### .2. MessagePassing&neighborhood aggregation

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618160144302.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618095312400.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618155312167.png)

![image-20210618095445424](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618095445424.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618160311559.png)

- `MessagePassing(aggr="add", flow="source_to_target", node_dim=-2)`: Defines the aggregation scheme to use (`"add"`, `"mean"` or `"max"`) and the flow direction of message passing (either `"source_to_target"` or `"target_to_source"`). Furthermore, the `node_dim` attribute indicates along which axis to propagate.
- `MessagePassing.propagate(edge_index, size=None, **kwargs)`: The initial call to start propagating messages. Takes in the edge indices and all additional data which is needed to construct messages and to update node embeddings. Note that [`propagate()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.propagate) is not limited to exchange messages in symmetric adjacency matrices of shape `[N, N]` only, but can also exchange messages in general sparse assignment matrices, *.e.g.*, bipartite graphs, of shape `[N, M]` by passing `size=(N, M)` as an additional argument. If set to [`None`](https://docs.python.org/3/library/constants.html#None), the assignment matrix is assumed to be symmetric. For bipartite graphs with two independent sets of nodes and indices, and each set holding its own information, this split can be marked by passing the information as a tuple, *e.g.* `x=(x_N, x_M)`.
- `MessagePassing.message(...)`: Constructs messages to node i in analogy to ϕϕfor each edge in (j,i)∈E(j,i)∈E if `flow="source_to_target"` and (i,j)∈E(i,j)∈E if `flow="target_to_source"`. Can take any argument which was initially passed to `propagate()`. In addition, tensors passed to `propagate()` can be mapped to the respective nodes ii and jj by appending `_i` or `_j` to the variable name, *.e.g.* `x_i` and `x_j`. Note that we generally refer to ii as the central nodes that aggregates information, and refer to jj as the neighboring nodes, since this is the most common notation.
- `MessagePassing.update(aggr_out, ...)`: Updates node embeddings in analogy to γγ for each node i∈Vi∈V. Takes in the output of aggregation as first argument and any argument which was initially passed to [`propagate()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.message_passing.MessagePassing.propagate).

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618104838632.png)

##### .1. [GCN Layer](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#id2)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618095657408.png)

> 1. Add self-loops to the adjacency matrix.
> 2. Linearly transform node feature matrix.
> 3. Compute normalization coefficients.
> 4. Normalize node features in ϕϕ.
> 5. Sum up neighboring node features (`"add"` aggregation).

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)  #当我们调用 propagate() 的时候，内部会自动的调用 message() 和 update() 函数，传递的参数是 x 。

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618103522520.png)

##### .2.[Edge Convolution](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#id3)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618100719167.png)

```python
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
  

from torch_geometric.nn import knn_graph

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)
```

##### .3. Global Pooling

> PyG also supports `graph-level outputs `as opposed to node-level outputs by providing a variety of `readout functions such as global add, mean or max pooling`. We additionaly offer more sophisticated methods such as set-to-set (Vinyals et al., 2016), sort pooling (Zhang et al., 2018) or the global soft attention layer from Li et al. (2016).

##### .4. Hierarchical Pooling

> To further `extract hierarchical information` and to allow deeper GNN models, various` pooling approaches can be applied in a spatial or data-dependent manner.` We currently provide implementation examples for `Graclus` (Dhillon et al., 2007; Fagginger Auer & Bisseling, 2011) and `voxel grid pooling` (Simonovsky & Komodakis, 2017), the `iterative farthest point sampling algorithm` (Qi et al., 2017) followed by `k-NN or query ball graph generation` (Qi et al., 2017; Wang et al., 2018b), and differentiable pooling mechanisms such as `DiffPool` (Ying et al., 2018) and` topk pooling` (Gao & Ji, 2018; Cangea et al., 2018)

### 4.  Application

> epidemiological forecasting, ride-hail demand prediction, web-traffic management, document labeling, fraud detection, traffic forecasting, chem-informatics systems

#### .1. Epidemiological Forecasting

```python
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

from tqdm import tqdm
model = RecurrentGCN(node_features = 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
>>> MSE: 0.6866
```

#### .2. Web Traffic Prediction

```python
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
loader = WikiMathsDatasetLoader()
dataset = loader.get_dataset(lags=14)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

from tqdm import tqdm
model = RecurrentGCN(node_features=14, filters=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in tqdm(range(50)):
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
>>> MSE: 0.7760
```

#### .3. Cora 2layerGCN

> 一个epoch中的一个data包含一个完整的数据集

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
>>> Accuracy: 0.8150
```

```python
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
data = dataset[0]
>>> Data(adj_t=[2708, 2708, nnz=10556], x=[2708, 1433], y=[2708], ...)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return float(loss)

for epoch in range(1, 201):
    loss = train(data)
```

#### .4. karate club  

> **Zachary's karate club** is a social network of a university karate club, described in the paper "An Information Flow Model for Conflict and Fission in Small Groups" by Wayne W. Zachary. The network became a popular example of [community structure](https://en.wikipedia.org/wiki/Community_structure) in networks after its use by [Michelle Girvan](https://en.wikipedia.org/wiki/Michelle_Girvan) and [Mark Newman](https://en.wikipedia.org/wiki/Mark_Newman) in 2002.[[1\]](https://en.wikipedia.org/wiki/Zachary's_karate_club#cite_note-GN-1)   

- **Node Classification**

```python
from torch_geometric.datasets import KarateClub
dataset = KarateClub()  #1 graph, number of features: 34, classes:4, which represent the community each node belongs to.
#Data(edge_index=[2, 156], train_mask=[34], x=[34, 34], y=[34])


import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
 
        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h

model = GCN()  #34→4→4→2->num_classes, 每一个row表示一个节点，对每一个节点进行分类
print(model)

model = GCN()
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize(h, color=data.y)  #h:<class 'torch.Tensor'>, grad_fn=<TanhBackward>) torch.Size([34, 2]


import time
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    if epoch % 10 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
```

#### .5. [Planetoid](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=9r_VmGMukf5R)

- **Node Classification**

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
#Number of classes: 7
```

- **MLP**

```python
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)

from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

- **GCN**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618121248908.png)

```python
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)


from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

#### .6. TUDdataset

- **Graph classification**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618131022161.png)

```python
import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#Data(edge_attr=[38, 4], edge_index=[2, 38], x=[17, 7], y=[1])
#Number of graphs: 188
#Number of features: 7
#Number of classes: 2
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data) #Batch(batch=[1169], edge_attr=[2592, 4], edge_index=[2, 2592], x=[1169, 7], y=[64])
    print()
    
    
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64)
print(model)
#GCN(
#  (conv1): GCNConv(7, 64)
#  (conv2): GCNConv(64, 64)
#  (conv3): GCNConv(64, 64)
#  (lin): Linear(in_features=64, out_features=2, bias=True)
#)

from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 201):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
```

#### .7. PointCloudClassification

- GeometricShapes

```python
# Install required packages.
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install -q torch-geometric

# Helper functions for visualization.
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_mesh(pos, face):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=data.face.t(), antialiased=False)
    plt.show()


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()
    
#load dataset
from torch_geometric.datasets import GeometricShapes
dataset = GeometricShapes(root='data/GeometricShapes')
```

> transform our meshes into points via the usage of "transforms". Here, PyTorch Geometric provides the [`torch_geometric.transforms.SamplePoints`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SamplePoints) transformation, which will uniformly sample a fixed number of points on the mesh faces according to their face area.

- **RandomRotate**

```python
from torch_geometric.transforms import Compose, RandomRotate

torch.manual_seed(123)

random_rotate = Compose([
    RandomRotate(degrees=180, axis=0),
    RandomRotate(degrees=180, axis=1),
    RandomRotate(degrees=180, axis=2),
])

dataset = GeometricShapes(root='data/GeometricShapes', transform=random_rotate)

data = dataset[0]
print(data)
visualize_mesh(data.pos, data.face)

data = dataset[4]
print(data)
visualize_mesh(data.pos, data.face)
```

- **SamplePoints**

```python
import torch
from torch_geometric.transforms import SamplePoints

torch.manual_seed(42)

dataset.transform = SamplePoints(num=256)

data = dataset[0]
print(data)   #Data(face=[3, 30], pos=[32, 3], y=[1]) =>Data(pos=[256, 3], y=[1])
visualize_points(data.pos, data.edge_index)

data = dataset[4]
print(data)   #Data(face=[3, 2], pos=[4, 3], y=[1])=>Data(pos=[256, 3], y=[1])
visualize_points(data.pos)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618133509868.png)

> `PointNet++` processes `point clouds iteratively` by following a `simple grouping, neighborhood aggregation and downsampling scheme:`
>
> 1. The **grouping phase** constructs a graph in which `nearby points are connected`. Typically, this is either done via `k-nearest neighbor search` or via `ball queries (which connects all points that are within a radius to the query point).`
> 2. The **neighborhood aggregation phase** executes a Graph Neural Network layer that, for each point,` aggregates information from its direct neighbors `(given by the graph constructed in the previous phase). This allows PointNet++ to capture local context at different scales.
> 3. The **downsampling phase** implements a pooling scheme suitable for point clouds with potentially different sizes. We will ignore this phase for now and will come back later to it.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618132211963.png)

- **knn_graph**

```python
from torch_cluster import knn_graph

data = dataset[0]
data.edge_index = knn_graph(data.pos, k=6)
print(data.edge_index.shape)
visualize_points(data.pos, edge_index=data.edge_index)

data = dataset[4]
data.edge_index = knn_graph(data.pos, k=6)
print(data.edge_index.shape)
visualize_points(data.pos, edge_index=data.edge_index)
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618132356302.png)

- **Neighborhood Aggregation**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618132502775.png)

```python
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.
```

```python
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool


class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, dataset.num_classes)
        
    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        
        # 5. Classifier.
        return self.classifier(h)


model = PointNet()
print(model)
```

```python
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

from torch_geometric.data import DataLoader

train_dataset = GeometricShapes(root='data/GeometricShapes', train=True,
                                transform=SamplePoints(128))
test_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                               transform=SamplePoints(128))


train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = PointNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader):
    model.train()
    
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        #batch(batch=[1280], pos=[1280, 3], ptr=[11], y=[10])
        logits = model(data.pos, data.batch)  # Forward pass.
        loss = criterion(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)

for epoch in range(1, 51):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
```

#### .8. BigGraph

> **Cluster-GCN** ([Chiang et al. (2019)](https://arxiv.org/abs/1905.07953), which is based on` pre-partitioning the graph into subgraphs on which one can operate in a mini-batch fashion`.

![image-20210618134507044](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210618134507044.png)

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
#Data(edge_index=[2, 88648], test_mask=[19717], train_mask=[19717], val_mask=[19717], x=[19717, 500], y=[19717])

from torch_geometric.data import ClusterData, ClusterLoader
torch.manual_seed(12345)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.
print()
total_num_nodes = 0
for step, sub_data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)#Data(edge_index=[2, 15230], test_mask=[4946], train_mask=[4946], val_mask=[4946], x=[4946, 500], y=[4946])
    print()
    total_num_nodes += sub_data.num_nodes
print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')


import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)



from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
def train():
      model.train()
      for sub_data in train_loader:  # Iterate over each mini-batch.
          out = model(sub_data.x, sub_data.edge_index)  # Perform a single forward pass.
          loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          optimizer.zero_grad()  # Clear gradients.
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      
      accs = []
      for mask in [data.train_mask, data.val_mask, data.test_mask]:
          correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
          accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
      return accs
for epoch in range(1, 51):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
```

#### .9. GNNModelExplain

```python
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
path = '.'
dataset = TUDataset(path, name='Mutagenicity').shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

#model define
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, GraphConv
class Net(torch.nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()

        num_features = dataset.num_features
        self.dim = dim

        self.conv1 = GraphConv(num_features, dim)
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
# train&test function
def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #Batch(batch=[3977], edge_attr=[7906, 3], edge_index=[2, 7906], ptr=[129], x=[3977, 14], y=[128])
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 2):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
    
    
    
from captum.attr import Saliency, IntegratedGradients
def model_forward(edge_mask, data):
    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    out = model(data.x, data.edge_index, batch, edge_mask)
    return out


def explain(method, data, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask



import random
from collections import defaultdict
def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict
  
data = random.choice([t for t in test_dataset if not t.y.item()])
mol = to_molecule(data)
for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
    edge_mask = explain(method, data, target=0)
    edge_mask_dict = aggregate_edge_directions(edge_mask, data)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    draw_molecule(mol, edge_mask_dict)
```

### 5. Visualization

```python
from torch_geometric.utils import to_networkx
%matplotlib inline
import torch
import networkx as nx
import matplotlib.pyplot as plt


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()
    
#data: Data(edge_index=[2, 156], train_mask=[34], x=[34, 34], y=[34])    
G = to_networkx(data, to_undirected=True)
#G <class 'networkx.classes.graph.Graph'>
visualize(G, color=data.y)
```

```python
import networkx as nx
import numpy as np

from torch_geometric.utils import to_networkx


def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')
    
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}    
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red')
    plt.show()


def to_molecule(data):
    ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F',
                'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = ATOM_MAP[data['x'].index(1.0)]
        del data['x']
    return g

import random
data = random.choice([t for t in train_dataset])
print(type(data),data)
mol = to_molecule(data)
plt.figure(figsize=(10, 5))
print(type(mol))
draw_molecule(mol)
#<class 'torch_geometric.data.data.Data'> Data(edge_attr=[76, 3], edge_index=[2, 76], x=[34, 14], y=[1])
#<class 'networkx.classes.digraph.DiGraph'>
```

### 6. Demo

#### .1. Customed dataset

- dataset generate

> - 10 graphs and 30 nodes per graph with random edges connections
>
> - number of node feature = 3
>
> - number of edge feature = 1
>
> - node's classification and graph classification
>
>     Adj [num_graph, num_node, num_node] be the adjacent matrices (sparse)
>     node_feature [num_graph, num_node, num_node_feature]
>     edge_feature [num_graph, num_node, num_node] (sparse)

```python
import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.utils as ut
np.random.seed(42)

num_graph = 10
num_node = 50
num_node_features = 3
num_edge_features = 1

Adj = np.random.rand(num_graph, num_node, num_node)
Adj[Adj >= 0.8] = True
Adj[Adj <= 0.8] = False
node_feature = np.random.rand(num_graph, num_node, num_node_features)
edge_feature = np.random.rand(num_graph, num_node, num_node) * Adj

graph_label = np.random.rand(num_graph)
graph_label[graph_label>0.5] = 1
graph_label[graph_label<0.5] = 0
graph_label = graph_label.astype(int)

node_label = np. random.rand(num_graph, num_node)
node_label[node_label>0.5] = 1
node_label[node_label<0.5] = 0
node_label = node_label.astype(int)

print(Adj[0, :,:], edge_feature[0, :, :], node_feature[0, :, :])
```

![image-20210621114008066](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20210621114008066.png)

#### .2. Graph Classification

> 一个graph数据对应一个Data， 可以将多个graph存储到一个data文件里面，也可以将每个graph存在对应单独的data文件里面。

- multi-graph&one data

```python
class GraphDatasetInMem(InMemoryDataset):
    """
    Graph classification 
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDatasetInMem, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\GraphDatasetInMem.dataset']
    
    def download(self):
        pass

    def process(self):
        data_list = [] # graph classification need to define data_list for multiple graph
        for i in range(num_graph):
            source_nodes, target_nodes = np.nonzero(Adj[i, :, :])
            source_nodes = source_nodes.reshape((1, -1))
            target_nodes = target_nodes.reshape((1, -1))

            edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long) # edge_index should be long type

            edge_weight = edge_feature[i, source_nodes, target_nodes]
            edge_weight = torch.tensor(edge_weight.reshape((-1, num_edge_features)), dtype=torch.float) # edge_index should be float
            type

            x = torch.tensor(node_feature[i, :, :], dtype=torch.float) 
            
            # y should be long type, graph label should not be a 0-dimesion tensor
            # use [graph_label[i]] ranther than graph_label[i]
            y = torch.tensor([graph_label[i]], dtype=torch.long) 

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)
            data_list.append(data)
            
        data, slices = self.collate(data_list) # Here used to be [data] for one graph
        torch.save((data, slices), self.processed_paths[0])
        
#usage
dataset_graph_InMem = GraphDatasetInMem(root='./')
print(dataset_graph_InMem[0])
print(dataset_graph_InMem[1])
#output
#Data(edge_attr=[504, 1], edge_index=[2, 504], x=[50, 3], y=[1])
#Data(edge_attr=[495, 1], edge_index=[2, 495], x=[50, 3], y=[1])
```

- one graph one pt file

> 区别在于：没有data, slices = self.collate(data_list) # Here used to be [data] for one graph，但是有以下函数：
>
> ```python
> def get(self, idx):
>         data = torch.load(osp.join(self.processed_dir, 'graphDataset1_{}.pt'.format(idx)))
>         return data
> ```

```python
class GraphDataset_1(Dataset):
    """
    Graph classification 
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset_1, self).__init__(root,transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\GraphDataset1_0.pt', r'.\GraphDataset1_1.pt', r'.\GraphDataset1_2.pt', r'.\GraphDataset1_3.pt', r'.\GraphDataset1_4.pt', r'.\GraphDataset1_5.pt', r'.\GraphDataset1_6.pt', r'.\GraphDataset1_7.pt', r'.\GraphDataset1_8.pt', r'.\GraphDataset1_9.pt']
    
    def download(self):
        pass

    def process(self):
        #data_list = [] # graph classification need to define data_list for multiple graph
        for i in range(num_graph):
            source_nodes, target_nodes = np.nonzero(Adj[i, :, :])
            source_nodes = source_nodes.reshape((1, -1))
            target_nodes = target_nodes.reshape((1, -1))

            edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long) # edge_index should be long type

            edge_weight = edge_feature[i, source_nodes, target_nodes]
            edge_weight = torch.tensor(edge_weight.reshape((-1, num_edge_features)), dtype=torch.float) # edge_index should be float
            type

            x = torch.tensor(node_feature[i, :, :], dtype=torch.float) 
            
            # y should be long type, graph label should not be a 0-dimesion tensor
            # use [graph_label[i]] ranther than graph_label[i]
            y = torch.tensor([graph_label[i]], dtype=torch.long) 

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)
            #data_list.append(data)
            # save one graph per time
            torch.save(data, osp.join(self.processed_dir, 'graphDataset1_{}.pt'.format(i)))
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graphDataset1_{}.pt'.format(idx)))
        return data

# usage
dataset_graph_1 = GraphDataset_1(root='./')
print(dataset_graph_1[0])
print(dataset_graph_1[1])
#Data(edge_attr=[504, 1], edge_index=[2, 504], x=[50, 3], y=[1])
#Data(edge_attr=[495, 1], edge_index=[2, 495], x=[50, 3], y=[1])
```

#### .3. Node Classification

- in on graph

```python
import os.path as osp
from torch_geometric.data import Dataset
class NodeDatasetInMem(InMemoryDataset):
    """
    node classification in one graph
    Should define the mask for training, validation and test
    """
    def __init__(self, root, num_train_per_class=15, num_val=10, num_test=10, transform=None, pre_transform=None):
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        super(NodeDatasetInMem, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\NodeDatasetInMem.dataset']
    
    def download(self):
        pass

    def process(self):
        num_train_per_class = self.num_train_per_class
        num_val = self.num_val
        num_test = self.num_test
        #data_list = []  # node classification do not neet to define data_list just data (one graph)
        i=0
        source_nodes, target_nodes = np.nonzero(Adj[i, :, :])
        source_nodes = source_nodes.reshape((1, -1))
        target_nodes = target_nodes.reshape((1, -1))

        edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long) # edge_index should be long type

        edge_weight = edge_feature[i, source_nodes, target_nodes]
        edge_weight = torch.tensor(edge_weight.reshape((-1, num_edge_features)), dtype=torch.float) # edge_index should be float
        type
        train_mask = np.zeros((num_node,), dtype=bool)
        val_mask = np.zeros((num_node,), dtype=bool)
        test_mask = np.zeros((num_node,), dtype=bool)

        label = node_label[i, :]
        [org_class_0_ind] =  np.nonzero(label == 0) 
        org_class_0_ind = org_class_0_ind.reshape(-1)
        perm_class_0_ind = org_class_0_ind[np.random.permutation(org_class_0_ind.shape[0])]

        [org_class_1_ind] =  np.nonzero(label == 1) 
        org_class_1_ind = org_class_1_ind.reshape(-1)
        perm_class_1_ind = org_class_1_ind[np.random.permutation(org_class_1_ind.shape[0])]


        train_ind = np.concatenate((perm_class_0_ind[:num_train_per_class], perm_class_1_ind[:num_train_per_class]), axis=0)
        train_mask[train_ind] = True

        [remaining] = np.nonzero(~train_mask)
        remaining = remaining.reshape(-1)

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:num_val+num_test]] = True

        train_mask = torch.tensor(train_mask, dtype=torch.bool) # mask should be long type
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        x = torch.tensor(node_feature[i, :, :], dtype=torch.float) 
        y = torch.tensor(node_label[i, :], dtype=torch.long) # y should be long type

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
            
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
        
 #output
dataset_node_InMem = NodeDatasetInMem(root='./')
print(dataset_node_InMem[0].y)
print(dataset_node_InMem[0].y.shape)

#tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1,0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1,1, 1])
#torch.Size([50])
```

```python
class NodeDataset(Dataset):
    """
    node classification in one graph
    Should define the mask for training, validation and test
    """
    def __init__(self, root, num_train_per_class=15, num_val=10, num_test=10, transform=None, pre_transform=None):
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        super(NodeDataset, self).__init__(root,transform, pre_transform)
        # Do not load the data and slices here
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'./NodeDataset_0.pt']
    
    def download(self):
        pass

    def process(self):
        num_train_per_class = self.num_train_per_class
        num_val = self.num_val
        num_test = self.num_test
        #data_list = []  # node classification do not neet to define data_list just data (one graph)
        i=0
        source_nodes, target_nodes = np.nonzero(Adj[i, :, :])
        source_nodes = source_nodes.reshape((1, -1))
        target_nodes = target_nodes.reshape((1, -1))

        edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0), dtype=torch.long) # edge_index should be long type

        edge_weight = edge_feature[i, source_nodes, target_nodes]
        edge_weight = torch.tensor(edge_weight.reshape((-1, num_edge_features)), dtype=torch.float) # edge_index should be float
        type
        train_mask = np.zeros((num_node,), dtype=bool)
        val_mask = np.zeros((num_node,), dtype=bool)
        test_mask = np.zeros((num_node,), dtype=bool)

        label = node_label[i, :]
        [org_class_0_ind] =  np.nonzero(label == 0) 
        org_class_0_ind = org_class_0_ind.reshape(-1)
        perm_class_0_ind = org_class_0_ind[np.random.permutation(org_class_0_ind.shape[0])]

        [org_class_1_ind] =  np.nonzero(label == 1) 
        org_class_1_ind = org_class_1_ind.reshape(-1)
        perm_class_1_ind = org_class_1_ind[np.random.permutation(org_class_1_ind.shape[0])]


        train_ind = np.concatenate((perm_class_0_ind[:num_train_per_class], perm_class_1_ind[:num_train_per_class]), axis=0)
        train_mask[train_ind] = True

        [remaining] = np.nonzero(~train_mask)
        remaining = remaining.reshape(-1)

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:num_val+num_test]] = True

        train_mask = torch.tensor(train_mask, dtype=torch.bool) # mask should be long type
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        x = torch.tensor(node_feature[i, :, :], dtype=torch.float) 
        y = torch.tensor(node_label[i, :], dtype=torch.long) # y should be long type

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
        # Directly save the data in order as .pt form
        torch.save(data, osp.join(self.processed_dir, 'NodeDataset_{}.pt'.format(i)))
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'NodeDataset_{}.pt'.format(idx)))
        return data
   #
dataset_node = NodeDataset(root='./')
dataset_node[0]
#Data(edge_attr=[504, 1], edge_index=[2, 504], test_mask=[50], train_mask=[50], val_mask=[50], x=[50, 3], y=[50])
```

### Resouce

- [pytorch_geometric11.3k ](https://github.com/rusty1s/pytorch_geometric) [pytorch_geometric11.3k_demo](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)
- [pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)  : A Temporal Extension Library for PyTorch Geometric
- https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
- [Introduction: Hands-on Graph Neural Networks](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing)
- [Node Classification with Graph Neural Networks](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing)
- [Graph Classification with Graph Neural Networks](https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing)
- [Scaling Graph Neural Networks](https://colab.research.google.com/drive/1XAjcjRHrSR_ypCk_feIWFbcBKyT4Lirs?usp=sharing)
- [Point Cloud Classification with Graph Neural Networks](https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing)
- [Explaining GNN Model Predictions using Captum](https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing)
- [Fast Graph Representation](http://htmlpreview.github.io/?https://github.com/rusty1s/rusty1s.github.io/blob/master/pyg_notebook.html)
- [custom dataset](https://github.com/GQ93/Pytorch-geometric-notes/blob/master/costumed_graph_datasets.ipynb) 



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pytorchgnn/  

