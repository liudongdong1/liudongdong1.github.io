# NoiseRelated


### 1. Perlin Noise

> Perlin Noise is an extremely powerful algorithm that is used often in procedural content generation. It is especially useful for games and other visual media such as movies.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201011200726085.png)

1. Input x, y, z coordinates, and [x,y,z]%1 to find the coordinate's location within the cube.                                                             ![image-20201011200936597](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201011200936597.png)

2. on each of the 4 unit coordinates(for 2d), generate a pseudorandom gradient vector, and define a positive direction( in the direction that it points to). The reasoning behind these specific gradient vectors is described in [Ken Perlin's SIGGRAPH 2002 paper: *Improving Noise*](http://mrl.nyu.edu/~perlin/paper445.pdf).

   ![Gradient Vector](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201011201315531.png)

3. calculate the 4 vectors from given point to the 4 surrounding points on the grid.![distance vector](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201011202243159.png)

4. take the [dot product](http://en.wikipedia.org/wiki/Dot_product) between the two vectors (the gradient vector and the distance vector). This gives us our final *influence* values: grad.x * dist.x + grad.y * dist.y + grad.z * dist.z

5.  interpolate between these 4 values so that we get a sort of weighted average in between the 4 grid points (8 in 3D). he fade function for the improved perlin noise implementation is this:$6*t*5-15*t*4+10*t*3![image-20201011205309630](C:/Users/dell/AppData/Roaming/Typora/typora-user-images/image-20201011205309630.png)$![image-20201011205309630](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201011205309630.png)

###    2. [Filters](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

[**Chapter 1: The g-h Filter**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/01-g-h-filter.ipynb)

Intuitive introduction to the g-h filter, also known as the αα-ββ Filter, which is a family of filters that includes the Kalman filter. Once you understand this chapter you will understand the concepts behind the Kalman filter.

[**Chapter 2: The Discrete Bayes Filter**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/02-Discrete-Bayes.ipynb)

Introduces the discrete Bayes filter. From this you will learn the probabilistic (Bayesian) reasoning that underpins the Kalman filter in an easy to digest form.

[**Chapter 3: Probabilities, Gaussians, and Bayes' Theorem**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/03-Gaussians.ipynb)

Introduces using Gaussians to represent beliefs in the Bayesian sense. Gaussians allow us to implement the algorithms used in the discrete Bayes filter to work in continuous domains.

[**Chapter 4: One Dimensional Kalman Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb)

Implements a Kalman filter by modifying the discrete Bayes filter to use Gaussians. This is a full featured Kalman filter, albeit only useful for 1D problems.

[**Chapter 5: Multivariate Gaussians**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/05-Multivariate-Gaussians.ipynb)

Extends Gaussians to multiple dimensions, and demonstrates how 'triangulation' and hidden variables can vastly improve estimates.

[**Chapter 6: Multivariate Kalman Filter**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb)

We extend the Kalman filter developed in the univariate chapter to the full, generalized filter for linear problems. After reading this you will understand how a Kalman filter works and how to design and implement one for a (linear) problem of your choice.

[**Chapter 7: Kalman Filter Math**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb)

We gotten about as far as we can without forming a strong mathematical foundation. This chapter is optional, especially the first time, but if you intend to write robust, numerically stable filters, or to read the literature, you will need to know the material in this chapter. Some sections will be required to understand the later chapters on nonlinear filtering.

[**Chapter 8: Designing Kalman Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb)

Building on material in Chapters 5 and 6, walks you through the design of several Kalman filters. Only by seeing several different examples can you really grasp all of the theory. Examples are chosen to be realistic, not 'toy' problems to give you a start towards implementing your own filters. Discusses, but does not solve issues like numerical stability.

[**Chapter 9: Nonlinear Filtering**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/09-Nonlinear-Filtering.ipynb)

Kalman filters as covered only work for linear problems. Yet the world is nonlinear. Here I introduce the problems that nonlinear systems pose to the filter, and briefly discuss the various algorithms that we will be learning in subsequent chapters.

[**Chapter 10: Unscented Kalman Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb)

Unscented Kalman filters (UKF) are a recent development in Kalman filter theory. They allow you to filter nonlinear problems without requiring a closed form solution like the Extended Kalman filter requires.

This topic is typically either not mentioned, or glossed over in existing texts, with Extended Kalman filters receiving the bulk of discussion. I put it first because the UKF is much simpler to understand, implement, and the filtering performance is usually as good as or better then the Extended Kalman filter. I always try to implement the UKF first for real world problems, and you should also.

[**Chapter 11: Extended Kalman Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb)

Extended Kalman filters (EKF) are the most common approach to linearizing non-linear problems. A majority of real world Kalman filters are EKFs, so will need to understand this material to understand existing code, papers, talks, etc.

[**Chapter 12: Particle Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb)

Particle filters uses Monte Carlo techniques to filter data. They easily handle highly nonlinear and non-Gaussian systems, as well as multimodal distributions (tracking multiple objects simultaneously) at the cost of high computational requirements.

[**Chapter 13: Smoothing**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/13-Smoothing.ipynb)

Kalman filters are recursive, and thus very suitable for real time filtering. However, they work extremely well for post-processing data. After all, Kalman filters are predictor-correctors, and it is easier to predict the past than the future! We discuss some common approaches.

[**Chapter 14: Adaptive Filtering**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb)

Kalman filters assume a single process model, but manuevering targets typically need to be described by several different process models. Adaptive filtering uses several techniques to allow the Kalman filter to adapt to the changing behavior of the target.

[**Appendix A: Installation, Python, NumPy, and FilterPy**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-A-Installation.ipynb)

Brief introduction of Python and how it is used in this book. Description of the companion library FilterPy.

[**Appendix B: Symbols and Notations**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-B-Symbols-and-Notations.ipynb)

Most books opt to use different notations and variable names for identical concepts. This is a large barrier to understanding when you are starting out. I have collected the symbols and notations used in this book, and built tables showing what notation and names are used by the major books in the field.

*Still just a collection of notes at this point.*

[**Appendix D: H-Infinity Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-D-HInfinity-Filters.ipynb)

Describes the H∞H∞ filter.

*I have code that implements the filter, but no supporting text yet.*

[**Appendix E: Ensemble Kalman Filters**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-E-Ensemble-Kalman-Filters.ipynb)

Discusses the ensemble Kalman Filter, which uses a Monte Carlo approach to deal with very large Kalman filter states in nonlinear systems.

[**Appendix F: FilterPy Source Code**](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-F-Filterpy-Code.ipynb)


---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/noiserelative/  

