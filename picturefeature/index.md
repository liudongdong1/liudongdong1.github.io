# PictureFeature


> Suárez, Iago, et al. "BEBLID: Boosted efficient binary local image descriptor." *Pattern Recognition Letters* 133 (2020): 366-372.

------

# Paper: BEBLID:

<div align=center>
<br/>
<b>BEBLID: Boosted efficient binary local image descriptor.
</b>
</div>

#### Summary

1. BEBLID, an efficient learned binary image descriptor;
2. achieves an accuracy close to SIFT and better computational efficiency than ORB.
3. BEBLID details:
   1. use the integral image to efficiently compute the difference between the mean gray values in a pair of square image regions.
   2. use the Boosted SCC algorithm to discriminatively select a set of features and combine them to produce a strong description.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210222223342668.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210222223651396.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210222223753793.png)

#### Relative 

- SIFT: requiring the use of a GPU to achieve real-time performance in certain contexts.
- ORB: the fastest binary approaches, use features based on the comparison of pairs of image pixels, use limited number of comparisons selected to be uncorrelated with unsupervised approach.
- DNN Relative:

#### [code](https://github1s.com/iago-suarez/beblid-opencv-demo/blob/main/demo.py)

> ORB检测到强角，在不同的尺度上比较它们，并使用FAST或Harris响应来挑选最好的。它还使用局部patch的一阶矩来寻找每个角点的方向。

```shell
#pip install "opencv-contrib-python>=4.5.1"
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from math import sqrt

parser = argparse.ArgumentParser(description='Code from AKAZE local features matching tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='graf1.png')
parser.add_argument('--input2', help='Path to input image 2.', default='graf3.png')
parser.add_argument('--homography', help='Path to the homography matrix.', default='H1to3p.xml')
args = parser.parse_args()

img1 = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

fs = cv.FileStorage(args.homography, cv.FILE_STORAGE_READ)
homography = fs.getFirstTopLevelNode().mat()

detector = cv.ORB_create(10000)   #使用orb方法
# descriptor = cv.ORB_create()
descriptor = cv.xfeatures2d.BEBLID_create(0.75)  # 使用beblid 方法。
kpts1 = detector.detect(img1, None)
kpts2 = detector.detect(img2, None)
kpts1, desc1 = descriptor.compute(img1, kpts1)
kpts2, desc2 = descriptor.compute(img2, kpts2)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)
matched1 = []
matched2 = []
nn_match_ratio = 0.8  # Nearest neighbor matching ratio
for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])
inliers1 = []
inliers2 = []
good_matches = []
inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
for i, m in enumerate(matched1):
    # Create the homogeneous point
    col = np.ones((3, 1), dtype=np.float64)
    col[0:2, 0] = m.pt
    # Project from image 1 to image 2
    col = np.dot(homography, col)
    col /= col[2, 0]
    # Calculate euclidean distance
    dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                pow(col[1, 0] - matched2[i].pt[1], 2))
    if dist < inlier_threshold:
        good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
        inliers1.append(matched1[i])
        inliers2.append(matched2[i])
res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

cv.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
cv.imwrite("matching_result.png", res)
inlier_ratio = len(inliers1) / float(len(matched1))
print('Matching Results')
print('*******************************')
print('# Keypoints 1:                        \t', len(kpts1))
print('# Keypoints 2:                        \t', len(kpts2))
print('# Matches:                            \t', len(matched1))
print('# Inliers:                            \t', len(inliers1))
print('# Inliers Ratio:                      \t', inlier_ratio)
cv.imshow('result', res)
cv.waitKey()

```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/picturefeature/  

