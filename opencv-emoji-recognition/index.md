# OpenCV Emoji Recognition


## 1. 表情识别模型

> 使用OpenVINO模型库中的**emotions-recognition-retail-0003**人脸表情模型，该模型是基于全卷积神经网络训练完成，使用ResNet中Block结构构建卷积神经网络。数据集使用了AffectNet表情数据集，支持五种表情识别，分别是：('neutral', 'happy', 'sad', 'surprise', 'anger')。
>
> 输入格式：NCHW=1x3x64x64
> 输出格式：1x5x1x1

```python
import cv2 as cv
import numpy as np
from openvino.inference_engine import IENetwork, IECore
weight_pb = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector_uint8.pb"
config_text = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector.pbtxt"
model_xml = "emotions-recognition-retail-0003.xml"
model_bin = "emotions-recognition-retail-0003.bin"
labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']
emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sad","surprise"]
emotion_net = IENetwork(model=model_xml, weights=model_bin)
ie = IECore()
versions = ie.get_versions("CPU")
input_blob = next(iter(emotion_net.inputs))
n, c, h, w = emotion_net.inputs[input_blob].shape
print(emotion_net.inputs[input_blob].shape)
output_info = emotion_net.outputs[next(iter(emotion_net.outputs.keys()))]
output_info.precision = "FP32"
exec_net = ie.load_network(network=emotion_net, device_name="CPU")
root_dir = "D:/facedb/emotion_dataset/"

def emotion_detect(frame):
    net = cv.dnn.readNetFromTensorflow(weight_pb, config=config_text)
    h, w, c = frame.shape
    blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)    
    net.setInput(blobImage)
    cvOut = net.forward()
    # 绘制检测矩形
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            # roi and detect landmark
            y1 = np.int32(top) if np.int32(top) > 0 else 0
            y2 = np.int32(bottom) if np.int32(bottom) < h else h-1
            x1 = np.int32(left) if np.int32(left) > 0 else 0
            x2 = np.int32(right) if np.int32(right) < w else w-1
            roi = frame[y1:y2,x1:x2,:]
            image = cv.resize(roi, (64, 64))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            res = exec_net.infer(inputs={input_blob: [image]})
            prob_emotion = res['prob_emotion']
            probs = np.reshape(prob_emotion, (5))
            txt = labels[np.argmax(probs)]
            cv.putText(frame, txt, (np.int32(left), np.int32(top)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv.rectangle(frame, (np.int32(left), np.int32(top)),
                         (np.int32(right), np.int32(bottom)), (0, 0, 255), 2, 8, 0)

if __name__ == "__main__":
    capture = cv.VideoCapture("D:/images/video/Boogie_Up.mp4")
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        emotion_detect(frame)
        cv.imshow("emotion-detect-demo", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
```

原文链接：https://mp.weixin.qq.com/s/C5jDkoxztNwv6mTvp3QmLQ

## 2. UNet 人像分割

> 人像分割的相关应用非常广，例如基于人像分割可以实现背景的替换做出各种非常酷炫的效果。我们将训练数据扩充到人体分割，那么我们就是对人体做美颜特效处理，同时对背景做其他的特效处理，这样整张画面就会变得更加有趣，更加提高颜值了，这里我们对人体前景做美颜调色处理，对背景做了以下特效：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200708093528787.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200708093555801.png)

> UNet网络，类型于一个U字母：首先进行Conv（两次）+Pooling下采样；然后Deconv反卷积进行上采样（部分采用resize+线性插值上采样），crop之前的低层feature map，进行融合；然后再次上采样。重复这个过程，直到获得输出388x388x2的feature map，最后经过softmax获得output segment map。
>
> 代码地址: https://github.com/milesial/Pytorch-UNet;  https://github.com/leijue222/portrait-matting-unet-flask

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200708093934749.png)

```python
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/opencv-emoji-recognition/  

