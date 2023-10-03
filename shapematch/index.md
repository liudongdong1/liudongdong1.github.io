# ShapeMatch


### 1. Image Moments Calculation

#### 1.1. raw moments

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815082821286.png)



#### 1.2. central moments

> the central moments are `translation invariant`, no matter where the blob is in the image, if the shape is the same, the moment will be the same.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815082947566.png)

#### 1.3. Normalized central moments

> both `translation and scale invariant`.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815083108353.png)

### 2. Hu Moments

> Hu Moments are a set of 7 numbers calculated using central moments that are invariant to image transformations, the first 6 moments have been proved to be invariant to translation,scale and rotation and reflection, while the 7th moments's sign changes for image reflection.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815085725522.png)

#### 2.1  calculation using Opencv

```python
# Read image as grayscale image
im=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
_,im=cv2.threshold(im,128,255,cv2.THRESH_BINARY)
#calculate Moments
mements=cv2.moments(im)
# calculate Hu Moments
huMoments=cv2.HuMoments(moments)
# log scale hu moments
for i in range(0,7):
	huMoments[i]=-1*copysign(1.0,huMoments[i])*log10(abs(huMoments[i]))
#shape matching using Hu Moments
d1=cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I1,0)
d2=cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)
d3=cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I3,0)
```

> The Hu Moments obtained in the previous step have a large range. For example, the 7 Hu Moments of K ( K0.png ) shown above.
>
> h[0] = 0.00162663
> h[1] = 3.11619e-07
> h[2] = 3.61005e-10
> h[3] = 1.44485e-10
> h[4] = -2.55279e-20
> h[5] = -7.57625e-14
> h[6] = 2.09098e-20
>
> Note that hu[0] is not comparable in magnitude as hu[6]. We can use use a log transform given below to bring them in the same range
>
> <font color=red>using log transform to translate date to the same range</font>.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815090912913.png)

> opencv provides an easy to use a utility function called matchShapes that takes in two images(or contours) and finds the distance between them using Hu Moments,

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815094311197.png)

### 3. Blob

> a blob is a group of connected pixels in an image that shares some commn property(eg. grayscale value).

To find the center of the blob, we will perform the following steps:-

1. Convert the Image to grayscale.

2. Perform Binarization on the Image.

3. Find the center of the image after calculating the moments.

#### 3.1. Center of a single blob

```c++
# convert image to grayscale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
```

#### 3.2. Center of Multiple blobs

```python
# read image through command line
img = cv2.imread(args["ipimage"])
# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)
# find contours in the binary image
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
   # calculate moments for each contour
   M = cv2.moments(c)
   if M["m00"] != 0:
	 cX = int(M["m10"] / M["m00"])
	 cY = int(M["m01"] / M["m00"])
   else:
	 cX, cY = 0, 0
   # calculate x,y coordinate of center
   cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
   cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

   # display the image
   cv2.imshow("Image", img)
   cv2.waitKey(0)
```

### 4. [matchShape methods](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=matchshapes)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200815100001467.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/shapematch/  

