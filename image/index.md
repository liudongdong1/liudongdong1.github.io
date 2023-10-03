# image


#### 1. imageHangle

```python
# WARNING: you are on the master branch, please refer to the examples on the branch that matches your `cortex version`

import cv2
import numpy as np


def resize_image(image, desired_width):
    current_width = image.shape[1]
    scale_percent = desired_width / current_width
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def compress_image(image, grayscale=True, desired_width=416, top_crop_percent=0.45):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = resize_image(image, desired_width)
    height = image.shape[0]
    if top_crop_percent:
        image[: int(height * top_crop_percent)] = 128

    return image


def image_from_bytes(byte_im):
    nparr = np.frombuffer(byte_im, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def image_to_jpeg_nparray(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    is_success, im_buf_arr = cv2.imencode(".jpg", image, quality)
    return im_buf_arr


def image_to_jpeg_bytes(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    buf = image_to_jpeg_nparray(image, quality)
    byte_im = buf.tobytes()
    return byte_im

```

#### 2. ColorList

```python
# WARNING: you are on the master branch, please refer to the examples on the branch that matches your `cortex version`


def get_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    code originally from https://github.com/fizyr/keras-retinanet/
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
    """
    if label < len(colors):
        return colors[label]
    else:
        print("Label {} has no color, returning default.".format(label))
        return (0, 255, 0)


colors = [
    [31, 0, 255],
    [0, 159, 255],
    [255, 95, 0],
    [255, 19, 0],
    [255, 0, 0],
    [255, 38, 0],
    [0, 255, 25],
    [255, 0, 133],
    [255, 172, 0],
    [108, 0, 255],
    [0, 82, 255],
    [0, 255, 6],
    [255, 0, 152],
    [223, 0, 255],
    [12, 0, 255],
    [0, 255, 178],
    [108, 255, 0],
    [184, 0, 255],
    [255, 0, 76],
    [146, 255, 0],
    [51, 0, 255],
    [0, 197, 255],
    [255, 248, 0],
    [255, 0, 19],
    [255, 0, 38],
    [89, 255, 0],
    [127, 255, 0],
    [255, 153, 0],
    [0, 255, 255],
    [0, 255, 216],
    [0, 255, 121],
    [255, 0, 248],
    [70, 0, 255],
    [0, 255, 159],
    [0, 216, 255],
    [0, 6, 255],
    [0, 63, 255],
    [31, 255, 0],
    [255, 57, 0],
    [255, 0, 210],
    [0, 255, 102],
    [242, 255, 0],
    [255, 191, 0],
    [0, 255, 63],
    [255, 0, 95],
    [146, 0, 255],
    [184, 255, 0],
    [255, 114, 0],
    [0, 255, 235],
    [255, 229, 0],
    [0, 178, 255],
    [255, 0, 114],
    [255, 0, 57],
    [0, 140, 255],
    [0, 121, 255],
    [12, 255, 0],
    [255, 210, 0],
    [0, 255, 44],
    [165, 255, 0],
    [0, 25, 255],
    [0, 255, 140],
    [0, 101, 255],
    [0, 255, 82],
    [223, 255, 0],
    [242, 0, 255],
    [89, 0, 255],
    [165, 0, 255],
    [70, 255, 0],
    [255, 0, 172],
    [255, 76, 0],
    [203, 255, 0],
    [204, 0, 255],
    [255, 0, 229],
    [255, 133, 0],
    [127, 0, 255],
    [0, 235, 255],
    [0, 255, 197],
    [255, 0, 191],
    [0, 44, 255],
    [50, 255, 0],
]
```

#### 3. BoundingBox

```python
# WARNING: you are on the master branch, please refer to the examples on the branch that matches your `cortex version`

import numpy as np
import cv2
from .colors import get_color


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def draw_boxes(image, boxes, overlay_text, labels, obj_thresh, quiet=True):
    for box, overlay in zip(boxes, overlay_text):
        label_str = ""
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != "":
                    label_str += ", "
                label_str += labels[i] + " " + str(round(box.get_score() * 100, 2)) + "%"
                label = i
            if not quiet:
                print(label_str)

        if label >= 0:
            if len(overlay) > 0:
                text = label_str + ": [" + " ".join(overlay) + "]"
            else:
                text = label_str
            text = text.upper()
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array(
                [
                    [box.xmin - 3, box.ymin],
                    [box.xmin - 3, box.ymin - height - 26],
                    [box.xmin + width + 13, box.ymin - height - 26],
                    [box.xmin + width + 13, box.ymin],
                ],
                dtype="int32",
            )

            # cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            rec = (box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin)
            rec = tuple(int(i) for i in rec)
            cv2.rectangle(img=image, rec=rec, color=get_color(label), thickness=3)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(
                img=image,
                text=text,
                org=(box.xmin + 13, box.ymin - 13),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1e-3 * image.shape[0],
                color=(0, 0, 0),
                thickness=1,
            )

    return image

```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/image/  

