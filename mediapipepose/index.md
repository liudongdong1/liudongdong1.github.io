# MediaPipePose


> Human pose estimation from video plays a critical role in various applications such as [quantifying physical exercises](https://google.github.io/mediapipe/solutions/pose_classification.html), `sign language recognition`, and `full-body gesture control`. For example, it can form the basis for` yoga, dance, and fitness applications`. It can also enable the overlay of digital content and information on top of the physical world in augmented reality.
>
> MediaPipe Pose is a ML solution for high-fidelity body pose tracking, inferring `33 3D landmarks `on the whole body from RGB video frames utilizing our [BlazePose](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html) research that also powers the [ML Kit Pose Detection API](https://developers.google.com/ml-kit/vision/pose-detection). Current state-of-the-art approaches rely primarily on powerful desktop environments for inference, whereas our method achieves real-time performance on most modern [mobile phones](https://google.github.io/mediapipe/solutions/pose.html#mobile), [desktops/laptops](https://google.github.io/mediapipe/solutions/pose.html#desktop), in [python](https://google.github.io/mediapipe/solutions/pose.html#python-solution-api) and even on the [web](https://google.github.io/mediapipe/solutions/pose.html#javascript-solution-api).

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/pose_tracking_full_body_landmarks.png)

A list of pose landmarks. Each landmark consists of the following:

- `x` and `y`: Landmark coordinates normalized to `[0.0, 1.0]` by the image width and height respectively.
- `z`: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of `z` uses roughly the same scale as `x`.
- `visibility`: A value in `[0.0, 1.0]` indicating the likelihood of the landmark being visible (present and not occluded) in the image.

### 1. Pose Draw

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 2. LandMark

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 3. curl Counter

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 4. 2D&3D 检测

> 注意mediapipe 版本的问题，plot3d

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(
            image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(
            image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img


def handleImage2D(filename):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.imread(filename)
        image_height, image_width, _ = image.shape
        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not results.pose_landmarks:
            print("没有检测到关键点")
            return
        landmarks = results.pose_landmarks.landmark
        print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x*image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*image_height,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z/landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x]
        shoulderRight = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*image_height,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z/landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x]
    
        print("shoulder Left:",shoulderLeft,"\t shoulder Right:",shoulderRight)
        annotated_image = image.copy()
        cv2.circle(annotated_image,(int(shoulderLeft[0]),int(shoulderLeft[1])),5,(255,255,0))
        cv2.circle(annotated_image,(int(shoulderRight[0]),int(shoulderRight[1])),5,(255,255,0))
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('annotated_image' +
                    str(1) + '.png', annotated_image)


def handleImage3D(filename):
    # Run MediaPipe Pose and plot 3d pose world landmarks.
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        image = cv2.imread(filename)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print the real-world 3D coordinates of nose in meters with the origin at
        # the center between hips.
        print('Nose world landmark:'),
        print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.Right_SHOULDER])
        
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

handleImage2D('./temp.png')
handleImage3D("temp.png")
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210806210106163.png)

### 5. HandDraw

- https://github.com/handtracking-io/yoha

![](https://user-images.githubusercontent.com/7460509/136609045-827ec14b-a94a-4477-b064-b1c9488754f6.gif)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mediapipepose/  

