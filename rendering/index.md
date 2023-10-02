# Rendering


> Rückert D, Franke L, Stamminger M. ADOP: Approximate Differentiable One-Pixel Point Rendering[J]. arXiv preprint arXiv:2110.06635, 2021. [pdf](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2110.06635.pdf)   [code  star 538](https://github.com/darglein/ADOP)
>

------

### Paper: ADOP

<div align=center>
<br/>
<b>ADOP: Approximate Differentiable One-Pixel Point Rendering</b>
</div>

#### Summary

1. present a novel point-based, differentiable neural rendering pipeline for scene refinement and novel view synthesis.
2. the point cloud rendering is performed by a differentiable renderer using multi-resolution one-pixel point rasterization.
3. after rendering , the neural image pyramid is passed through a deep neural network for shading calculations and hole-filling.

previous work:

- Point-Based rendering: 
  - rendering as points as oriented discs, which are usually called splats or surfels, with the radius of each disc being precomputed from the point cloud density. The discs are rendered with a Gaussian alpha-mask and then combined by a normalizing blend funciton. `[Auto Splats]`;
  - point sample rendering, where points are rendered as one-pixel splats generating a sparse image of the scene. using iterative or pyramid-based, hole filling approaches.
- Novel view synthesis: 
  - image-based rendering(IBR): relies on the basic principle of warping colors from one frame to another.
  - use a triangle-mesh proxy to directly warp the image colors to a novel view.
  - learning-based approaches to create multi plane image representation or directly estimate the required warp-field.
- Inverse Rendering:
  - Traditional triangle rasterization with depth-testing has no analytically correct spatial derivative, avalaible systems eigher approximate the gradient or approximate the rendering itself using alpha blending along edges.

#### Methods

- **Problem Formulation**:
- **system overview**:

> Given a set of RGB images and an initial 3D reconstruction, this inverse rendering approach is able to synthesize novel frames and optimize the scene's parameters, which includes structural parameters like point position and camera pose as well as image settings such as exposure time and white balance.
>
> - an initial estimate of the point cloud, and the camera parameters.
> - the output  are synthesized images from arbitrary camera poses.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025084206355.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211025090611976.png)





---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/rendering/  

