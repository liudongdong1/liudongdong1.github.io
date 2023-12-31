# TemplateMatching


### 1. 识别空的货架空间

> 如果使用模板匹配，`轻微倾斜/移动`，就很难找到这种方法。我们需要多个不同尺寸的模板来捕获这张图片中的所有空货架区域。

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210528092130181.png)

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("/content/drive/MyDrive/Computer Vision/new_shelf.jpg")
plt.figure(figsize = (20,15))
plt.imshow(img)
#创建特定模板的代码
template_1 = img[60:270, 1890:2010]
plt.imshow(template_1)
template_2 = img[300:500, 1825:1905]
plt.imshow(template_2)

*******************************************************************/
* Title: template_defenition.py
* Author: Jean Rovani
* Date: 2020
* Code version: 3rd revision
* Availability: https://gist.github.com/jrovani/012f0c6e66647b4e7b844797fa6ded22#file-template_definition-py
*******************************************************************/
DEFAULT_TEMPLATE_MATCHING_THRESHOLD = 0.85
class Template:
    def __init__(self, label, template, color, matching_threshold=DEFAULT_TEMPLATE_MATCHING_THRESHOLD):
        self.label = label
        self.color = color
        self.template = template
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold
image = cv2.imread("/content/drive/MyDrive/Computer Vision/shelf_new.jpg")
templates = [
    Template(label="1", template = template_1, color=(0, 0, 255)),
    Template(label="2", template = template_2, color=(0, 255, 0))
]

*******************************************************************/
* Title: plot_bounding_boxes.py
* Author: Jean Rovani
* Date: 2020
* Code version: 6th revision
* Availability: https://gist.github.com/jrovani/099f80a5ee75657ff7aa6ed491568f04#file-plot_bounding_boxes-py
*******************************************************************/
detections_1 = []
detections_2 = []
for template in templates:
    template_matching = cv2.matchTemplate(
        template.template, image, cv2.TM_CCOEFF_NORMED
    )
match_locations = np.where(template_matching >= template.matching_threshold)
for (x, y) in zip(match_locations[1], match_locations[0]):
        match = {
            "TOP_LEFT_X": x,
            "TOP_LEFT_Y": y,
            "BOTTOM_RIGHT_X": x + template.template_width,
            "BOTTOM_RIGHT_Y": y + template.template_height,
            "MATCH_VALUE": template_matching[y, x],
            "LABEL": template.label,
            "COLOR": template.color
        }
        if match['LABEL'] == '1':
          detections_1.append(match)
        else:
          detections_2.append(match)
        
        
#效果查看
image_with_detections = image.copy()
for temp_d in [detections_1, detections_2]:
  for detection in temp_d:
      cv2.rectangle(
          image_with_detections,
          (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
          (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
          detection["COLOR"],
          2,
      )
plt.figure(figsize = (20,15))
plt.imshow(image_with_detections)


#Sorting detections by BOTTOM_RIGHT_X coordinate
detections_1 = sorted(detections_1, key = lambda i: i['BOTTOM_RIGHT_X'])
detections_2 = sorted(detections_2, key = lambda i: i['BOTTOM_RIGHT_X'])
det_wo_dupl_1 = [detections_1[0]]
det_wo_dupl_2 = [detections_2[0]]
check = 1
min_x_1 = templates[0].template.shape[1]
min_x_2 = templates[1].template.shape[1]
for d in range(1, len(detections_1)):
  min_x_check = detections_1[d]["BOTTOM_RIGHT_X"] - detections_1[d-check]["BOTTOM_RIGHT_X"]
  if min_x_check > min_x_1:
    det_wo_dupl_1.append(detections_1[d])
    check = 1
  else:
    check += 1
check = 1
for d in range(1, len(detections_2)):
  min_x_check = detections_2[d]["BOTTOM_RIGHT_X"] - detections_2[d-check]["BOTTOM_RIGHT_X"]
  if min_x_check > min_x_2:
    det_wo_dupl_2.append(detections_2[d])
    check = 1
  else:
    check += 1
det_wo_dupl = det_wo_dupl_1 + det_wo_dupl_2
print(len(det_wo_dupl))


#过滤处理
image_with_detections = image.copy()
min_x = templates[0].template.shape[1]
for detection in det_wo_dupl:
    cv2.rectangle(
        image_with_detections,
        (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
        (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
        detection["COLOR"],
        20,
    )
plt.figure(figsize = (20,15))
plt.imshow(image_with_detections)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/templatematching/  

