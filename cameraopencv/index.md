# CameraOpencv


```python
import cv2
from PyQt5.QtCore import QTimer

class Camera(object):
    def __init__(self,timesInterval):
        self.device = 0
        self.timesInterval =timesInterval  #ms
        self.cap = cv2.VideoCapture()
        self.timer = QTimer()       #A single-shot timer fires only once, non-single-shot timers fire every interval milliseconds.

    def stop(self):
        self.timer.stop()
        self.cap.release()
        return True

    def pause(self):
        self.timer.stop()

    def begin(self):
        self.timer.start(self.timesInterval)

    def start(self, device):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.start(self.timesInterval)
        self.cap.open(device)
        self.device = device
        return True

    def restart(self):
        self.start(self.device)

    @property
    def is_pause(self):
        return self.cap.isOpened() and not self.timer.isActive()

    @property
    def is_open(self):
        return self.cap.isOpened()

    @property
    def frame(self):
        if self.is_open and not self.is_pause:
            return self.cap.read()[1]

    @property
    def frame_count(self):
        if self.is_open:
            return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def frame_pos(self):
        if self.is_open:
            return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    @frame_pos.setter
    def frame_pos(self, value):
        if self.is_open:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)

    @property
    def resolution(self):
        if self.is_open:
            return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


def showGestureType():
    #Todo 
	pass
def openimg():
	"""displays predefined gesture images at right most window"""
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL )
	image = cv2.imread(tempplatefile)
	cv2.imshow("Image",image)
	cv2.setWindowProperty("Image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.resizeWindow("Image",298,430)
	cv2.moveWindow("Image", 1052,214)



def capture_images(self,cam,saveimg):
	"""Saves the images for custom gestures if button is pressed in custom gesture generationn through gui"""
	cam.release()
	cv2.destroyAllWindows()
	if not os.path.exists(tempdatafile+'SampleGestures'):
		os.mkdir(tempdatafile+'SampleGestures')

	gesname=saveimg[-1]
	if(len(gesname)>=1):
		img_name = "./SampleGestures/"+"{}.png".format(str(gesname))
		save_img = cv2.resize(mask, (image_x, image_y))
		cv2.imwrite(img_name, save_img)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/cameraopencv/  

