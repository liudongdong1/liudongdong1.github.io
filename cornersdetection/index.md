# OpencvBaseOp


## 0. Image Processing

### 0.1. **Scaling an Image** 

```python
import cv2 
import numpy as np 
FILE_NAME = 'volleyball.jpg'
try: 
    # Read image from disk. 
    img = cv2.imread(FILE_NAME) 
    # Get number of pixel horizontally and vertically. 
    (height, width) = img.shape[:2] 
    # Specify the size of image along with interploation methods. 
    # cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC 
    # is used for zooming. 
    res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC) 
    # Write image back to disk. 
    cv2.imwrite('result.jpg', res) 
except IOError: 
    print ('Error while reading files !!!') 
```

### 0.2. **Rotating an image**

```python
import cv2 
import numpy as np 
FILE_NAME = 'volleyball.jpg'
try: 
    # Read image from the disk. 
    img = cv2.imread(FILE_NAME) 
    # Shape of image in terms of pixels. 
    (rows, cols) = img.shape[:2]  
    # getRotationMatrix2D creates a matrix needed for transformation. 
    # We want matrix for rotation w.r.t center to 45 degree without scaling. 
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
    res = cv2.warpAffine(img, M, (cols, rows)) 
    # Write image back to disk. 
    cv2.imwrite('result.jpg', res) 
except IOError: 
    print ('Error while reading files !!!') 
```

### 0.3. **Translating**

```python
import cv2 
import numpy as np 
FILE_NAME = 'volleyball.jpg'
# Create translation matrix. 
# If the shift is (x, y) then matrix would be 
# M = [1 0 x] 
#     [0 1 y] 
# Let's shift by (100, 50). 
M = np.float32([[1, 0, 100], [0, 1, 50]])  
try: 
    # Read image from disk. 
    img = cv2.imread(FILE_NAME) 
    (rows, cols) = img.shape[:2] 
    # warpAffine does appropriate shifting given the 
    # translation matrix. 
    res = cv2.warpAffine(img, M, (cols, rows)) 
    # Write image back to disk. 
    cv2.imwrite('result.jpg', res) 
except IOError: 
    print ('Error while reading files !!!') 
```

## 1. FaceDetection

```python
# Creating database 
# It captures images and stores them in datasets  
# folder under the folder name of sub_data 
import cv2, sys, numpy, os 
haar_file = 'haarcascade_frontalface_default.xml'
  
# All the faces data will be 
#  present this folder 
datasets = 'datasets'  
  
  
# These are sub data sets of folder,  
# for my faces I've used my name you can  
# change the label here 
sub_data = 'vivek'     
  
path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
    os.mkdir(path) 
  
# defining the size of images  
(width, height) = (130, 100)     
  
#'0' is used for my webcam,  
# if you've any other camera 
#  attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0)  
  
# The program loops until it has 30 images of the face. 
count = 1
while count < 30:  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
        cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
    count += 1
      
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break
```

## 2. Face Features

```python
# We import the necessary packages 
from imutils import face_utils 
import numpy as np 
import argparse 
import imutils 
import dlib 
import cv2 
  
# We construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape-predictor", required = True, 
    help ="path to facial landmark predictor") 
ap.add_argument("-i", "--image", required = True, 
    help ="path to input image") 
args = vars(ap.parse_args()) 
  
# We are initializing the  dlib's face detector (HOG-based) and then  
# creation of the facial landmark predictor 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"]) 
  
# We then load the input image, resize it, and convert it to grayscale 
images = cv2.imread(args["image"]) 
images = imutils.resize(images, width = 500) 
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
  
# We then detect faces in the grayscale image 
rects = detector(gray, 1) 
  
# Now, job is to loop over the face detections 
for (i, rect) in enumerate(rects): 
    # We will determine the facial landmarks for the face region, then 
    # can convert the facial landmark (x, y)-coordinates to a NumPy array 
    shape = predictor(gray, rect) 
    shape = face_utils.shape_to_np(shape) 
  
    # We then convert dlib's rectangle to a OpenCV-style bounding box 
    # [i.e., (x, y, w, h)], then can draw the face bounding box 
    (x, y, w, h) = face_utils.rect_to_bb(rect) 
    cv2.rectangle(images, (x, y), (x + w, y + h), (255, 255, 0), 2) 
  
    # We then show the face number  
    cv2.putText(images, 'Face % {}'.format(i + 1), (x - 10, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
  
    # We then loop over the (x, y)-coordinates for the facial landmarks  
    # and draw them on the image 
    for (x, y) in shape: 
        cv2.circle(images, (x, y), 1, (0, 0, 255), -1) 
  
# Now show the output image with the face detections as well as  
# facial landmarks 
cv2.imshow("Output", images) 
cv2.waitKey(0) 
```

## 3. Edge Deteciton

### 3.1. Canny Edge

```python
# OpenCV program to perform Edge detection in real time 
# import libraries of python OpenCV  
# where its functionality resides 
import cv2  
# np is an alias pointing to numpy library 
import numpy as np 
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
# loop runs if capturing has been initialized 
while(1): 
    # reads frames from a camera 
    ret, frame = cap.read() 
    # converting BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    # define range of red color in HSV 
    lower_red = np.array([30,150,50]) 
    upper_red = np.array([255,255,180])  
    # create a red HSV colour boundary and  
    # threshold HSV image 
    mask = cv2.inRange(hsv, lower_red, upper_red) 
    # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
    # Display an original image 
    cv2.imshow('Original',frame) 
    # finds edges in the input image image and 
    # marks them in the output map edges 
    edges = cv2.Canny(frame,100,200)
    # Display edges in a frame 
    cv2.imshow('Edges',edges)  
    # Wait for Esc key to stop 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
# Close the window 
cap.release() 
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  
```

### 3.2  Shi-Tomasi Corner

> Shi-Tomasi Corner Detection was published by J.Shi and C.Tomasi in their paper ‘*Good Features to Track*‘. Here the basic intuition is that corners can be detected by looking for significant change in all direction.

```python
# Python program to illustrate  
# corner detection with  
# Shi-Tomasi Detection Method 
    
# organizing imports  
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline   
# path to input image specified and   
# image is loaded with imread command 
img = cv2.imread('chess.png')  
# convert image to grayscale 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Shi-Tomasi corner detection function 
# We are detecting only 100 best corners here 
# You can change the number to get desired result. 
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10) 
# convert corners values to integer 
# So that we will be able to draw circles on them 
corners = np.int0(corners) 
# draw red color circles on all corners 
for i in corners: 
    x, y = i.ravel() 
    cv2.circle(img, (x, y), 3, (255, 0, 0), -1) 
# resulting image 
plt.imshow(img)  
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows() 
```

### 3.3. Feature of Contours

> an **image moment** is a certain particular weighted average ([moment](https://en.wikipedia.org/wiki/Moment_(mathematics))) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. Image moments are useful to describe objects after [segmentation](https://en.wikipedia.org/wiki/Image_segmentation). [Simple properties of the image](https://en.wikipedia.org/wiki/Image_moment#Examples) which are found *via* image moments include area (or total intensity), its [centroid](https://en.wikipedia.org/wiki/Centroid), and [information about its orientation](https://en.wikipedia.org/wiki/Image_moment#Examples_2).

```python
import numpy as np
import cv2 as cv
img = cv.imread('star.jpg',0)
ret,thresh = cv.threshold(img,127,255,0)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print( M )
#计算质心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#计算轮廓周长
perimeter = cv.arcLength(cnt,True)
#轮廓逼近
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
# 绘制边框   直角边框
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# 绘制边框   旋转边框
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)
```

## 4. Count black dots on a white surface

```python
import cv2 
path ="white dot.png"
# reading the image in grayscale mode 
gray = cv2.imread(path, 0) 
# threshold 
th, threshed = cv2.threshold(gray, 100, 255,  cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
# findcontours 
cnts = cv2.findContours(threshed, cv2.RETR_LIST,                 cv2.CHAIN_APPROX_SIMPLE)[-2] 
# filter by area 
s1 = 3
s2 = 20
xcnts = [] 
for cnt in cnts: 
    if s1<cv2.contourArea(cnt) <s2: 
        xcnts.append(cnt) 
# printing output 
print("\nDots number: {}".format(len(xcnts))) 
```

## 5. Pedestrian

```python
import cv2 
import imutils 
   
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
cap = cv2.VideoCapture('vid.mp4') 
   
while cap.isOpened(): 
    # Reading the video stream 
    ret, image = cap.read() 
    if ret: 
        image = imutils.resize(image,  
                               width=min(400, image.shape[1])) 
   
        # Detecting all the regions  
        # in the Image that has a  
        # pedestrians inside it 
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05) 
   
        # Drawing the regions in the  
        # Image 
        for (x, y, w, h) in regions: 
            cv2.rectangle(image, (x, y), 
                          (x + w, y + h),  
                          (0, 0, 255), 2) 
   
        # Showing the output Image 
        cv2.imshow("Image", image) 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break
  
cap.release() 
cv2.destroyAllWindows() 
```

## 6. Smile Detection

```python
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') 
faces  = face_cascade.detectMultiScale(gray, 1.3, 5) 
def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame 
video_capture = cv2.VideoCapture(0) 
while True: 
   # Captures video_capture frame by frame 
    _, frame = video_capture.read()   
    # To capture image in monochrome              
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
    # calls the detect() function     
    canvas = detect(gray, frame)    
    # Displays the result on camera feed                      
    cv2.imshow('Video', canvas)  
    # The control breaks once q key is pressed                         
    if cv2.waitKey(1) & 0xff == ord('q'):                
        break
# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
```

## 7. Circle Detection

> iris detection to white blood cell segmentation;

- **Initializing the Accumulator Matrix:** Initialize a matrix of dimensions rows * cols * maxRadius with zeros.

- **Pre-processing the image:** Apply blurring, grayscale and an edge detector on the image. This is done to ensure the circles show as darkened image edges.

- **Looping through the points:** Pick a point ![x_i](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2d981c0ca950adeb20a721d62b690d5f_l3.png) on the image.

- Fixing r and looping through a and b: 

  Use a double nested loop to find a value of r, varying a and b in the given ranges.

- **Voting:** Pick the points in the accumulator matrix with the maximum value. These are strong points which indicate the existence of a circle with a, b and r parameters. This gives us the Hough space of circles.

- **Finding Circles:** Finally, using the above circles as candidate circles, vote according to the image. The maximum voted circle in the accumulator matrix gives us the circle.

> **Detection Method:** *OpenCV has an advanced implementation, HOUGH_GRADIENT, which uses gradient of the edges instead of filling up the entire 3D accumulator matrix, thereby speeding up the process.*
> **dp:** *This is the ratio of the resolution of original image to the accumulator matrix.*
> **minDist:** *This parameter controls the minimum distance between detected circles.*
> **Param1:** *Canny edge detection requires two parameters — minVal and maxVal. Param1 is the higher threshold of the two. The second one is set as Param1/2.*
> **Param2:** *This is the accumulator threshold for the candidate detected circles. By increasing this threshold value, we can ensure that only the best circles, corresponding to larger accumulator values, are returned.*
> **minRadius:** *Minimum circle radius.*
> **maxRadius:** *Maximum circle radius.*

```python
import cv2 
import numpy as np 
# Read image. 
img = cv2.imread('eyes.jpg', cv2.IMREAD_COLOR) 
# Convert to grayscale. 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Blur using 3 * 3 kernel. 
gray_blurred = cv2.blur(gray, (3, 3)) 
# Apply Hough transform on the blurred image. 
detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 40) 
  
# Draw circles that are detected. 
if detected_circles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
        cv2.imshow("Detected Circle", img) 
        cv2.waitKey(0) 
```

## 8. [Line Detection](https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/)

> - First it creates a 2D array or accumulator (to hold values of two parameters) and it is set to zero initially.
> - Let rows denote the r and columns denote the (θ)theta.
> - Size of array depends on the accuracy you need. Suppose you want the accuracy of angles to be 1 degree, you need 180 columns(Maximum degree for a straight line is 180).
> - For r, the maximum distance possible is the diagonal length of the image. So taking one pixel accuracy, number of rows can be diagonal length of the image.

```python
# Python program to illustrate HoughLine 
# method for line detection 
import cv2 
import numpy as np 
# Reading the required image in  
# which operations are to be done.  
# Make sure that the image is in the same  
# directory in which this python program is 
img = cv2.imread('image.jpg') 
# Convert the img to grayscale 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
# Apply edge detection method on the image 
edges = cv2.Canny(gray,50,150,apertureSize = 3) 
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 200) 
# The below for loop runs till r and theta values  
# are in the range of the 2d array 
for r,theta in lines[0]: 
    # Stores the value of cos(theta) in a 
    a = np.cos(theta) 
    # Stores the value of sin(theta) in b 
    b = np.sin(theta) 
    # x0 stores the value rcos(theta) 
    x0 = a*r 
    # y0 stores the value rsin(theta) 
    y0 = b*r    
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
    x1 = int(x0 + 1000*(-b)) 
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
    y1 = int(y0 + 1000*(a)) 
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
    x2 = int(x0 - 1000*(-b)) 
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
    y2 = int(y0 - 1000*(a)) 
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
    # (0,0,255) denotes the colour of the line to be  
    #drawn. In this case, it is red.  
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)  
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 
cv2.imwrite('linesDetected.jpg', img) 
```

## 9. Gun Detection

```python
import numpy as np 
import cv2 
import imutils 
import datetime 
gun_cascade = cv2.CascadeClassifier('cascade.xml') 
camera = cv2.VideoCapture(0) 
firstFrame = None
gun_exist = False 
while True:     
    ret, frame = camera.read() 
    frame = imutils.resize(frame, width = 500) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    gun = gun_cascade.detectMultiScale(gray, 
                                       1.3, 5, 
                                       minSize = (100, 100))      
    if len(gun) > 0: 
        gun_exist = True          
    for (x, y, w, h) in gun:       
        frame = cv2.rectangle(frame, 
                              (x, y), 
                              (x + w, y + h), 
                              (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w]     
    if firstFrame is None: 
        firstFrame = gray 
        continue
    # print(datetime.date(2019)) 
    # draw the text and timestamp on the frame 
    cv2.putText(frame, datetime.datetime.now().strftime("% A % d % B % Y % I:% M:% S % p"), 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.35, (0, 0, 255), 1) 
    cv2.imshow("Security Feed", frame) 
    key = cv2.waitKey(1) & 0xFF   
    if key == ord('q'): 
        break
        if gun_exist: 
    print("guns detected") 
else: 
    print("guns NOT detected") 
camera.release() 
cv2.destroyAllWindows() 
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/cornersdetection/  

