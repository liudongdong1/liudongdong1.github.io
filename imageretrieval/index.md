# ImageRetrieval


> 给定一个包含特定实例(例如特定目标、场景、建筑等)的查询图像，图像检索旨在从`数据库图像`中找到包含`相同实例的图像`。但由于`不同图像的拍摄视角、光照、或遮挡情况不同`，如何设计出能应对这些类内差异的有效且高效的图像检索算法仍是一项研究难题。

# 1. Survey

> Chen W, Liu Y, Wang W, et al. Deep image retrieval: A survey[J]. arXiv preprint arXiv:2101.11282, 2021. [url](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2101.11282.pdf)

- **content based image retrieval(CBIR)** is the problem of searching for semantically matched or similar image in a large image gallery by analyzing their visua, content.  `campact yet rich feature representations`
  -  `Application era:` Person re-identification, remote sensing, medical image search, shopping recommendation in onlene markets;
  - `instance level`: a query image of a particular object or scene is given and the goal is to find images containing the same object or scene that ma be captured under different conditions;
  - `cactegory level:` find images of the same class as the query;
  - `feature engineering era:` hand-engineered feature descriptors SIFT;
  - `feature learning era: `AlexNet, ImageNet;
- **Challenges & Goal:**
  - reducing the `semantic gap`
  - improving retrival scalability: ` domain shift`
  - balancing` retrieval accuracy and efficiency`:  high dimensional anc contain more semantic-aware information

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006104414562.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006104630732.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006105237367.png)

- **single feedforward pass methods:** take the whole image and feed it into model to extract features.   lacks geometric invariance and spatial information.
- **multiple feedforward pass methods:** using sliding windows or spatial pyramid model to create multi-scale image patches, and each patch is fed into the model before being encodeded as a final global feature.  instead of generating multi-scale image patches randomly and densely, `region proposal methods` are introduced like RPNs, CKNs.

> (a)-(b) `Non-parametric mechanisms:` The attention is based on convolutional feature maps x with size H ×W ×C. `Channel-wise attention in (a)` produces a C-dimensional importance vector α1 [10], [30]. `Spatial-wise attention in (b) `computes a 2-dimensional attention map α2 [10], [28], [59], [79]. (c)-(d) `Parametric mechanisms`: The attention weights β are provided by a sub-network with trainable parameters (e.g. θ in (c)) [97], [98]. Likewise, some off-the-shelf models [91], [99] can predict the attention maps from the input image directly.  

![Attention mechanisms](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006122356475.png)



> Representative methods in single feedforward frameworks, focusing on convolutional feature maps x with
> size H ×W ×C: MAC [47], R-MAC [27], GeM pooling [41], SPoC with the Gaussian weighting scheme [7], CroW [10], and CAM+CroW [28]. Note that g1(·) and g2(·) represent spatialwise and channel-wise weighting functions, respectively.  

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006110245921.png)

![Image patch generation](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006111137946.png)

- **Deep feature selection:**
  - a fully-connected layer has a global receptive field, but` lack of spatial information (using multiple feedforward passes)`; ` lack of local geometric invariance (leverage intermediate convolutional layers)`
  - a convolutional layer arranges the spatial information well and produces location-adaptive features.  sum-pooling convolutional features(SPoC) to compact descriptors pre-processed with Gaussian center prior. use BoW model to embed convolutional featuers separately, use VLAD to encode local features into VLAD features.
- **Feature fustion strategy**
  - `a layer-level fusion`: fusing feature from different layers with different balancing weights aims at combining different feature properties within a feature extractor.  `features from fully-connected layers(global features) and features from convolutional layers(local features) can complement each other when measuring semantic similarity and guarantee retrival performance`.
  - `model-level fusion`: combine features on different models to achieve improved performance, categorized into intra-model and inter-model.
    - intra-model: multiple deep models having similar or highly compatible structures.
    - inter-model: involves models with more differin structures.
    - `early fusion`: straightforward to fuse all types of features from the candidate models and then learning a metric based on the concatenated feature; `late fusion:` learn optimal metrics separately for the features from each model, and then to uniformly combine these metrics for final retrieval ranking. `What features are the best to combined`
- **Deep feature Enhancement**
  - `feature aggregation:`   sum/average pooling is less disciminative, taking into all activated outputs from conv layer, which weakening the effect of highly activated features. `max pooling`: for sparse features that have a low probability of being active.   conv feature maps can be directly aggregated to produce global features by spatial pooling.
  - `feature embedding:` embed the conv feature maps into a high dimensional space to obtain compact features, lik BoW, VLAD, FV. And using PCA to reduce embedding demision.
  - `Attention Mechanisms`: to highlight th emost relevant features and to avoid the influence of irrelevant activations.
  - `Deep hash Embedding`: to tranform deep features into more compact codes, hash functions can be plugged as a layer into deep networks, so that the hasn codes can be trained and optimized with model simultaneously. The hash codes of originally similar images are embedded as close as possible.
    - preserving image similarity: to minimize the inconsistencies between the real-valued features and corresponding hash codes. 
      - class label available: loss function are designed to learn hash codes in a Hamming space. like optimize the difference between  matrices computed from the binary codes and their supervision lavels.  `Siamese loss, triplet loss, adversarial learning` is used to retain semantic similarity where only dissiilar pairs keep their distance within a margin.
      - unsupervised hash learning: using Bayes classifiers, KNN graphs, K-means algorithms, AutoEncoders, Generative adversarial networks
    - improving hash function quality: aims at making the binary codes uniformly distributed.
- **Supervised Fine-tuning**: 
  - classification-based Fine-tuning: improves the model-lebel adaptability for new datasets, but may have some difficulties in learning discriminative intra-class variability to distinguish particular objects.
  - verification-based Fine-tuning: learn an optimal metric which minimizes or maximizes the distance of pairs to validate and maintain their similarity. Focus on both inter-class and intra-class samples.
    - a pair-wise constraint, corresponding to a Siamese network, in which input images are paired with either a positive or negative sample.
    - a triplet constraint, associated with triplet netwroks, in which anchor images are paired with both similar and dissimilar samples.
    - glovally supervised approaces(c,d) learn a metric on gloval features by satisfying all constraints, locally supervised approaches focus on local areas by only satisfying the given local constriants.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006124924358.png)

![sample mining strategies](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006125645651.png)

- **Unsupervised Fine-tuning**: 
  - `mining samples with Manifold learning:` to capture intrinsic correlations on the manifold structure to mine or deduce revelance.
  - `AutoEncoder-based Frameworks`: to reconstruct its output as closely as possible to its input.

> First stage: the affinity matrix is interpreted as a weighted kNN graph, where each vector is represented by a node, and edges are difined by the pairwise affinities of two connected nodes. the pairwise affinities are re-evaluated in the context of all other elements by diffusing the similarity values through the graph. the difference among random walk are lie primarily in three aspects
>
> - similarity initialization: affects the subsequenct KNN graph construction in an affinity matrix;
> - transition matrix definition: a row-stochastic matrix, determines the probabilities of transiting from one node to another in the graph.
> - iteration scheme: to re-valuate and update the values in affinity matrix by the manifold similarity until some kind of convergence is achieved.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006130746382.png)

# 2. Paper Reading

> Yoon, Sangwoong, et al. "Image-to-Image Retrieval by Learning Similarity between Scene Graphs." *arXiv preprint arXiv:2012.14700* (2020).

------

## Paper: Scene Graphs

<div align=center>
<br/>
<b>Image-to-Image Retrieval
by Learning Similarity between Scene Graphs
</b>
</div>

#### Summary

1. performing image retrieval with complex images that have multiple objects and various relationships between them remains challenging:
   - overly sensitive to low-level and local visual features;
   - no publicly available labeled data to train and evaluate the image retrieval system for complex image.
2. propose IRSGS, a novel image retrieval framework that utilizes the  similarty between scene graphs computed from a graph neural network to retrieve semantically similar images;
3. propose to train the proposed retrieval framework with the surrogate relevance measure obtained from image captions and a pre-trained language model;

- the scene graph  S={objects, attributes, relations};  the surrogate relevance measure between two images as the similarity between their captions;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210109154029944.png)

#### Relative work:

- **Image Retrieval:** using visual feature representations; object categories, text descriptions;
- **Scene Graphs**: image captioning; visual question answering; image-ground dialog;
- **Graph Similarity learning:** use the learned graph representations of two graph to calculate similarity;

![image-20210109153411093](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210109153411093.png)

> Teichmann, Marvin, et al. "Detect-to-retrieve: Efficient regional aggregation for image search." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2019.

------

## Paper: Detect-to-retrieve

<div align=center>
<br/>
<b>Detect-to-retrieve: Efficient regional aggregation for image search
</b>
</div>

#### Summary

1. improving region selection: introduce a dataset of manually boxed landmark images, with 86k images from 15k unique classes;
2. leverage the trained detector and produce more efficient regional search systems, which improves accuracy for small objects with only a modest increase to the databases size;
3. propose regional aggregated match kernels to leverage selected image regions and produce a discriminative image representation.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210110083533162.png)

> Deep local features and object regions are extracted from an image.   Regional aggregation proceeds in two steps, using a large codebook of visual words: first, per-region `VLAD description`; second, sum pooling and per-visual word normalization.

#### RelatedWork:

- Region search and aggregation: 
  - regional search: selected regions are encoded independently in the database; using `VLAD` or `Fisher Vectors`;
  - regional aggregation: selected regions are used to improve image representations. like leverage the grid structure form to pool pretained CNN features;

#### Features:

- **Regional Search&&Aggregation: ** `build on top of deep local features(DELF) and aggregated selective match kernels(ASMK)`;

  - match kernel framework: 
    - Image X with M local descriptors: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110084612921.png)
    - code book C comprising C visual words, learned using k-means, is used to quantize the descriptors;
    - ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110084743327.png)
    - encompasses popular local feature aggregation techniques such as Bag-of-Words, VLAD, and ASMK. Similarity: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110084818574.png)
    - an aggregated vector representation: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110085056545.png)
    - a scalar selectivity function: $\sigma(.)$;![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110085307986.png)
    - normalization factor: ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110085732839.png)

- **Regional Search:** query image X && database image $Y^{(n)}$;

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110090030382.png)

- **Regional Aggregated Match Kernels:**

  - storing descriptors of each region independently in the database incurs additional cost for both memory and search computation.

  - utilizing the detected bounding boxes to instead improve the aggregated representations of database images--producing discriminative descriptors at no additional cost.

    ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110090415987.png)

    ![For VLAD](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110090445547.png)

    ![R-ASMK](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110090613560.png)

    ![R-AMK](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110090643427.png)

> Weyand, Tobias, et al. "Google Landmarks Dataset v2-A Large-Scale Benchmark for Instance-Level Recognition and Retrieval." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

------

## Paper: Google Landmarks Dataset v2

<div align=center>
<br/>
<b>A Large-Scale Benchmark for Instance-Level Recognition and Retrieval
</b>
</div>

#### Summary

- the new dataset has several long-tailed class distribution, a large fraction of out-of-domain test photos and large intro-class variability.
  - class distribution;  intra-class variation;  out-of-domain query images;
- introduce the Google Landmarks Dataset v2, a new large-scale dataset for instance-level recognition and retrieval, includes over 5M images of over 200k human-made and natural landmarks.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110093703863.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210110094638916.png)

>Caron M, Touvron H, Misra I, et al. Emerging properties in self-supervised vision transformers[J]. arXiv preprint arXiv:2104.14294, 2021.

------

# Paper: ViT

<div align=center>
<br/>
<b>Emerging properties in self-supervised vision transformersTitle</b>
</div>


#### Summary

1. self-supervised ViT features contain explicit information about the `semantic segmentation of an image`, which does not emerge as clearly with supervised ViTs, nor with convnets.  These features are also excellent k-NN classifiers.

#### **Application Area**:

  - Image Retrieval
- Copy detection
- discovering the semantic layout of scenes
- video instance segmentation

#### Methods

- **Problem Formulation**:

- **system overview**:

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211007135531081.png)

# 3. project Code

### .1. [image-match](https://github.com/ProvenanceLabs/image-match)

> Wong H C, Bern M, Goldberg D. An image signature for any kind of image[C]//Proceedings. International Conference on Image Processing. IEEE, 2002, 1: I-I.  [code](https://github.com/dsys/match)
>
> -  star 2.6k, 1.6k, tutorial: [link](https://image-match.readthedocs.io/en/latest/start.html), traditional methods, including `python, elesticserach`
> - sensitive enough to allow efficient nearest-neighbor search,  to effectively filter a database for possible duplicates, and yet robutst enough to find duplicates that hve been resized, rescanned, or lossily compressed.

1. if the image is color, first convert it to 8-bit gray scale using the standard color-conversion algorithms, include djpeg and ppmtopgm. 255: pure white; 0: pure black;
2. impose a 9\*9 grid of points on the image. fro each column of the image, compute teh sum of absolute values of differences between adjacent pixels, compute the total of all columns, and crop the image at 5% and 95% columns. and crop the rows of the image the same way. Then divede the croped image into 10\*10 grid of blocks, and setting a 9*9 grid of points on the image.
3. at each grid point, compute the average gray level of the P\*P square centered at the grid point.
4. for each grid point, compute an 8-element array whose elements give a comparision of the average gray level of the grid point square with those of its eight neighbors.
5. the signature of an image is simply the concatenation of the 8-element arrays corresponding to the grid points, ordered left-right, top-bottom. 9\*9\*8=648;

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006135153568.png)

```python
from image_match.goldberg import ImageSignature
gis = ImageSignature()
a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
gis.normalized_distance(a, b)  #计算俩个图像之间距离
```

```python
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES

es = Elasticsearch()
ses = SignatureES(es)

ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
ses.add_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
ses.add_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')

ses.search_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
```

### .2. **[EagleEye](https://github.com/ThoughtfulDev/EagleEye)**

> - You enter this data into EagleEye and it tries to find Instagram, Youtube, Facebook, and Twitter Profiles of this person.
> - using python, dlib, face_reognition, Selenum, find person, star 2.8k
> - including facebook, google, imageraider, instagram 代码。

### .3. **[`PyRetri`](https://github.com/PyRetri/PyRetri)**

> Hu B, Song R J, Wei X S, et al. PyRetri: A PyTorch-based library for unsupervised image retrieval by Deep Convolutional Neural Networks[C]//Proceedings of the 28th ACM International Conference on Multimedia. 2020: 4461-4464.   [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2005.02154.pdf)  star: 946

- an open source library for deep learning based unsupervised image retrival, encapsulating the retrieval process in several stages and provides functionality tha tcovers various prominent methods for each stage.
- propose the first open sourve framework to unify the pipeline of deep learning based unsupervised image retrieval.
- provide high quality implementations of CBIR algorithms to solve retrieval tasks with emphasis on usaility

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006140532252.png)

#### .1. Pre-processing methods

- DirectResize (DR): Scaling the height and width of the image to the target size directly.
- PadResize (PR): Scaling the longer side of the image to the target size and filling the remaining pixels with the meanvalues of ImageNet.
- ShorterResize (SR): Scaling the shorter side of the image to the target size.
- TwoFlip (TF): Returning the original image and the corresponding horizontally flipped image.
- CenterCrop (CC): Cropping the image from its center region according to the given size.
- TenCrop (TC): Cropping the original image and the flipping image from up down left right and center, respectively.

#### .2. Feature Represention Methods

- GAP: Global average pooling.
- GMP: Global max pooling.
- R-MAC [14]: Calculating feature vectors based on the regionalmaximum activation of convolutions.
- SPoC [2]: Assigning larger weights to the central descriptorsduring aggregation.
- CroW [7]: A weighted pooling method for both spatial- andchannel-wise.
- SCDA [17]: Keeping useful deep descriptors based on the sum-mation of feature map activations.
- GeM [11]: Exploiting the generalized mean to reserve theinformation of each channel.
- PWA [19]: Aggregating the regional representations weightedby the selected part detectors’ output.
- PCB [13]: Outputting a convolutional descriptor consisting ofseveral part-level features

#### .3. Post-precessing Methods

- SVD [6]: Reducing feature dimension through singular valuedecomposition of matrix.
- PCA [18]: Projecting high-dimensional features into fewerinformative dimensions.
- DBA [1]: Every feature in the database is replaced with aweighted sum of the point’s own value and those of its topknearest neighbors (k-NN).
- QE [3]: Combining the retrieved top-knearest neighbors withthe original query and doing another retrieval.
- k-reciprocal [22]: Encodingk-reciprocal nearest neighbors toenhance the accuracy of retrieval

#### .4. Database

- Oxford5k [9] collects crawling images fromFlickrusing thenames of 11 different landmarks in Oxford, which is a repre-sentative landmark retrieval task.
- CUB-200-2011 [15] contains photos of 200 bird species, whichrepresents fine-grained image retrieval.
- Indoor [10] contains indoor scene images with 67 categories,representing for the scene retrieval/recognition task.
- Caltech101 [4] consists pictures of objects belonging to 101categories, standing for the generic image retrieval task.
- •Market-1501 [20] contains images taken on the Tsinghua cam-pus under six camera viewpoints, which is the benchmarkdataset for person re-identification.
- DukeMTMC-reID [12] contains images captured by eight cam-eras, which is a more challenging person Re-ID dataset.

### 4. [natural-language-image-search](https://github.com/haltakov/natural-language-image-search)

> Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[J]. arXiv preprint arXiv:2103.00020, 2021. ` star 4.9k`  [url](https://github.com/openai/CLIP)   [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2103.00020.pdf)

> OpenAI's [CLIP](https://openai.com/blog/clip/) neural networs is able to transform both images and text into the same latent space, where they can be compared using a similarity measure. all photos from the full [Unsplash Dataset](https://unsplash.com/data) (almost 2M photos) were downloaded and processed with CLIP. The precomputed feature vectors for all images can then be used to find the best match to a natural language search query.

> introducing a neural network called `CLIP` which efficiently learns `visual concepts from natural language supervision`. CLIP can be applied to any visual classification benchmark by simply providing the names of the visual categories to be recognized, similar to the “zero-shot” capabilities of GPT-2 and GPT-3.

> CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a *dog*” and predict the class of the caption CLIP estimates best pairs with a given image.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006141807218.png)

### 5. [sis](https://github.com/matsui528/sis)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211006142200903.png)

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
```

```python
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
```







---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/imageretrieval/  

