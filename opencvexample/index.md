# OpencvExample


## 1. 车牌识别

```python
import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('D://skoda1.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (600,400) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#使用双边滤波（模糊）会从图像中删除不需要的细节。
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200) 
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
# Masking the part other than the number plate  裁剪图片
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2. [Color Tracking](https://github.com/gaborvecsei/Color-Tracker)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200830160521.png)

## 3. .颜色分割或阈值分割 

```python
import cv2 as cv import matplotlib.pyplot as pltfrom PIL import Image 
!wget -nv https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19 
img = Image.open('./bird.png')

```

## 4. 手掌检测与计数

```python
import numpy as np
import cv2 as cv

def skinmask(img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    ret, thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    return thresh

def getcnthull(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    return contours, hull

def getdefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

cap = cv.VideoCapture("your_video_path") # '0' for webcam
while cap.isOpened():
    _, img = cap.read()
    try:
        mask_img = skinmask(img)
        contours, hull = getcnthull(mask_img)
        cv.drawContours(img, [contours], -1, (255,255,0), 2)
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
        defects = getdefects(contours)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv.circle(img, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
            cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
        cv.imshow("img", img)
    except:
        pass
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917182916225.png)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20200917183149461.png)

> Side：*a，b，c*和angle：*gamma。*现在，该伽马始终小于90度，因此可以说：如果*伽马*小于90度或*pi / 2*，则将其视为手指。

## 5. VolumnControl

```python
import cv2
import mediapipe as mp
import time
 
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
 
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
 
        return lmList
 
 
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()
```

```python
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

from pynput.keyboard import Key,Controller
keyboard = Controller()

################################
wCam, hCam = 1280, 720
################################
 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
 
detector = htm.handDetector(detectionCon=0.7)


last_angle=None
last_length=None



minAngle = 0
maxAngle = 180
angle    = 0
angleBar = 400
angleDeg = 0
minHand  = 50 #50
maxHand  = 300 #300
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
 
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
 
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
 
        # Hand range 50 - 300
    
        angle  = np.interp(length, [minHand, maxHand], [minAngle, maxAngle])
        angleBar = np.interp(length, [minHand, maxHand], [400, 150])
        angleDeg = np.interp(length, [minHand, maxHand], [0, 180])   # degree angle 0 - 180
       

        if last_length:
            if length>last_length:
                keyboard.press(Key.media_volume_up)
                keyboard.release(Key.media_volume_up)
                print("VOL UP")
            elif length<last_length:
                keyboard.press(Key.media_volume_down)
                keyboard.release(Key.media_volume_down)
                print("VOL DOWN")
        
        last_angle=angle
        last_length=length

        # print(int(length), angle)
 
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
 
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(angleBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(angleDeg)} deg', (40, 90), cv2.FONT_HERSHEY_COMPLEX,
                2, (0, 9, 255), 3)
 
    cv2.imshow("Img", img)
    cv2.waitKey(1)
```



1. Drowsiness Detector 睡意检测

https://github.com/misbah4064/drowsinessDetector

2. Object Tracking 目标跟踪

https://github.com/misbah4064/object_tracking

3. Lane Detection 车道线检测

https://github.com/misbah4064/lane_detection

4. Face Landmark Detection 人脸特征点检测https://github.com/misbah4064/faceLandmarks模型文件：https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2

5. Hand Pose Detection 手部姿态检测

https://github.com/misbah4064/hand_pose_detection

6. Yolo Object Detection and Tracking CPU端YOLO目标检测与跟踪

https://github.com/misbah4064/yolo_objectDetection_imagesCPU

7. License Plate Detection and Recognition 车牌检测与识别

https://github.com/misbah4064/licensePlateReader

8. Text Detection and Recognition 文本检测与识别

https://github.com/misbah4064/textDetection-Recognition

9. Background Removal and replace with video 视频背景去除与替换

https://github.com/misbah4064/backgroundRemoval

10. Car Detection 车辆检测

https://github.com/misbah4064/car_detector_haarcascades

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/opencvexample/  

