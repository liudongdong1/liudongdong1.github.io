# CircleGan


> Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." *Proceedings of the IEEE international conference on computer vision*. 2017.  cite 10600  [[pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1703.10593.pdf)] [[code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

------

# Paper: CircleGAN

<div align=center>
<br/>
<b>Unpaired image-to-image translation using cycle-consistent adversarial networks</b>
</div>

#### Summary

1. present a method that can learn to do the same, `capturing special characteristics `of one image collection and figureing out how these characteristics could be `translated into the other image collection`.

#### Research Objective

  - **Application Area**: 
      - `Collection style transfer`: learns to mimic the style of an entire collection of artworks, rather than transferring the style of a single selseted piece of art.
      - `Object transfiguration:`  trained to tranlate one object class form ImageNet to another, or translate one object into another object of the same category, or translate between two visually similar categories.
      - `Season transfer:` 
      - `Photo generation from paintings:` 
      - `Photo enhancement:` generate photos with shallower depth of field.
- **Purpose**: 
  -  to lear the mapping between an input image and an output image using a training set of aligned image pairs.
  - to translate an image from a source domain X to a target domain Y in the absence of paired examples.

#### Relative work:

- **Generative Adversarial Networks(GANs):**   like image generation, image editing, representation learing, text2image, image inpainting, future prediction.   `the adversarial loss that forces the generated images to be`
- **Image-to-Image Translation**: use a dataset of input-output examples to learn a parametric translation function using CNNs,  using pix2pix framework, a conditional generative adversarial network to learn a mapping from input to output images, like `generating photographs from sketches or from attribute and semantic layouts`
- **Unpaired Image-to-Image Translation:** use a weight-sharing strategy to learn a common representation across domains.
- **Cycle Consistency:** using transitivity as a way to regularize structured data.
- **Neural Style Transfer**: synthesizes a novel image by combining the` content of one image with the style of another image` based on matching the `Gram matrix statistics of pre-trained deep features`. `to capture correspondences between higher-level appearance structures`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120123435526.png)

#### Methods

- **Problem Formulation**:

![image-20211120161816552](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120161816552.png)

- **system overview**:
  - given any two unordered image collections X, Y, our algorithm learns to automatically translate an image from one into the other and vice versa.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120160938235.png)

- idt loss的定义在论文的application之中，防止input 与out put之间的color compostion过多
- **Adversarial loss**:  尽可能让生成器生成的数据分布接近于真实的数据分布

![adversarial loss](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120163349319.png)

- **Cycle Consistency Loss:**  due to the factor that the adversarial losses alone cannot guarantee that the learned function can map an individual input x to a desired output y.  `for each image x from domain x, the image translation cycle should be able to bring x back to the original image.`

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120164015832.png)

- 生成器G的loss： self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
- 判别器D的loss：![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211121125715426.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120163705404.png)

- **Full Objective**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120164258463.png)

#### Evaluation

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120122855683.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211120164439195.png)

#### Code

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211121124249044.png)

```python
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.Alignment_net = Alignment()
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        #inter,final = self.Alignment_net(self.real_A,self.rec_B)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
```

**level**: CCF_A
**author**: Phillip Isola;  Jun-Yan Zhu; Tinghui Zhou; Alexei A. Efros
**date**: 2018
**keyword**:

- GANs, 

------

# Paper: Pix2pix

<div align=center>
<br/>
<b>Image-to-Image Translation with Conditional Adversarial Networks</b>
</div>



#### Summary

1. investigate conditional adversarial networks as general-purpose solution to image-to-image translation problem, that the net learn the mapping from input image to output image, and learn a loss function to train the mapping;

#### Research Objective

  - **Application Area**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018123156008.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018142258797.png)

#### Previous work:

- **Structured losses for image modeling:**  image -to-image translation problems as per-pixel classification or regression that each output pixel is considered conditionally independent from all others given the input image; conditional GANs instead learn a structured loss which penalize the joint configuration of the output.![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211121115016608.png)
- **Conditional GANs**: used for discrete labels, text, image prediction from a normal map, future frame prediction, product photo generation, image generation from sparse annotations, inpainting, future state prediction, image manipulation guided by user constraints, style transfer, superresolution.

#### Methods

- **Challenge:**

  - output is high-dimensional, structured object;
    - Use a deep net, D, to analyze output!
  - uncertainty in mapping; many plausible outputs;
    - D only cares about “plausibility”, doesn’t hedge; like **MAD-GAN; BiCycleGAN**

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018131631479.png)

- **Problem Formulation**:

> For GANs, learning a mapping from random noise vector z to output image  $y$, $G: z->y$;
>
> For conditional GANs, learning a mapping from observed image x and random noise vector z to $y$, $G:\{x,z\}-->y$;

- **cGAN vs Pix2pix**

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20211121115129571.png)

- **Loss Function Object**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018121240333.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018140505080.png)

```python
   def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
```

> restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness, term a PatchGAN that only penalizes structure at the scale of patches, like tries to classify if each N*N patch in an image is real or fake.

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018121401145.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018131416705.png)

#### Evaluation

  - **Environment**:   

    - Semantic labels↔photo, trained on the Cityscapes dataset [12]. 
    - Architectural labels→photo, trained on CMP Facades [45]. 
    - Map↔aerial photo, trained on data scraped from Google Maps. 
    - BW→color photos, trained on [51]. 
    - Edges→photo, trained on data from [65] and [60]; binary edges generated using the HED edge detector [58] plus postprocessing. 
    - Sketch→photo: tests edges→photo models on humandrawn sketches from [19]. 
    - Day→night, trained on [33]. 
    - Thermal→color photos, trained on data from [27].
    - Photo with missing pixels→inpainted photo, trained on Paris StreetView from [14].

- **Evaluation metrics:**

  - run “real vs. fake” perceptual studies on Amazon Mechanical Turk (AMT).
  - measure whether or not our synthesized cityscapes are realistic enough that off-the-shelf recognition system can recognize the objects in them.

- **Results**

  - Different Loss ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018122611607.png)![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018122709256.png)

  - Different Generator architectures:![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018122741654.png)

  - Different receptive field:![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018122825191.png)


![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018122931674.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201018123033271.png)

#### Notes <font color=orange>去加强了解</font>

  - 运行论文代码

**level**:  2021 ICRL
**author**: Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., & Cohen-Or, D.
**date**: 2020
**keyword**:

- GAN, styletransfer; latent space encoding

> Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., & Cohen-Or, D. (2020). Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation. *arXiv preprint arXiv:2008.00951*.

------

# Paper: StyleGAN(pSp)

<div align=center>
<br/>
<b>Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
</b>
</div>



#### Summary

1. pSp framework is based on an encoder network that directly generates a series of style vectors which are fed into a pretrained style-GAN generator, forming the extended W+ latent space;
2. introduce a dedicated identity loss which is shown to achieve improved performance in the reconstruction of an input image.
3. a novel styleGan encoder able to directly encode real face images into the W+ latent domain.
4. makes our model operate globally instead of locally, without requiring pixel-to-pixel correspondence.

#### Research Objective

  - **Application Area**: stylegan inversion; frontalization; inpainting; face generation from segmentation; super resolution; face interpolation for real images;![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019191118769.png)



- **Purpose**:   retrieval of the latent vector that generates a desired, not necessarily known image;

#### Proble Statement

- how to controlling stylegan's latent space and performing meaningful manipulations in W;
- how to accelerate the encoding process?

previous work:

- StyleGAN 23,24:   a disentangled latent space W, obtained from the initial space Z via a multi-layer perceptron mapping network, which offer control and editing capabilities.
- **Latent Space Embedding:** inversion methods 
  - directly optimize the latent vector to minimize the error for the given image;
  - train an encoder to map the given image to the latent space;
  - use a hybrid approach combining both;
- **Latent-Space Manipulation:**  first, inverting  a given image into the latent space, then editing the inverted latent code **in a semantically meaningful manner** to obtain a new code used by the unconditional GAN to generate the output image;
  - finding linear directions that correspond to changes in a given binary labeled attribute;
  - utilize a pretrained 3DMM to learn semantic face edits in the latent space;
  - finding useful paths in a completely unsupervised manner by using PCA;
- the input image must be invertible, like exits a latent code that reconstructs the image;
- the input image domain must typically be the same domain the GAN was trained on.
- the latent space does not contain rich semantics for an unknown data damain;
- GAN inversion remains difficult.

#### Methods

- **Problem Formulation**:

- **system overview**:

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019200512316.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019200645869.png)

【Loss Function】

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019200940361.png)

![image-20201019201025121](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201025121.png)

- preserve identity between the input and output images: ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201156223.png)

  ![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201248655.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201651904.png)

#### Evaluation

  - **Environment**:   
    - Dataset: CelebA-HQ dataset [20], which contains 30,000 high quality images

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201837985.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201019201918457.png)

#### Notes <font color=orange>去加强了解</font>

  - 学习一下相关代码



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/circlegan/  

