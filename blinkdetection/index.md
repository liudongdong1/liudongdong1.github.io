# BlinkDetection


### 1. 眨眼检测

- 使用dlib库 眨眼检测

```python
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


threshold = 0.27
shape_predictor = os.path.join(
    'backend', "shape_predictor_68_face_landmarks.dat")


class Eye_Detector():
    def __init__(self):
        EYE_AR_THRESH = threshold
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.index = [lStart, lEnd, rStart, rEnd]

    def calculate_ear(self, frame):
        lStart, lEnd, rStart, rEnd = self.index
        detector, predictor = self.detector, self.predictor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        ear = 0
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
        return ear

```

- 使用socketio进行通信

```python
#!/usr/bin/env python3
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from PIL import Image
import base64
import io
import cv2
import numpy as np
import random
import time
from flask_sslify import SSLify
from backend import Eye_Detector

# ==================
# Initialize
# ==================

app = Flask(__name__)
socketio = SocketIO(app)
player_list = []
detector = Eye_Detector()
SSLify(app)

# prevent cached responses(https://stackoverflow.com/questions/47376744/how-to-prevent-cached-response-flask-server-using-chrome)
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

# ==================
# Utils Function
# ==================


def find_by_id(_id):
    """
    Find element by id in the player_list.
    Args:
      _id: ID of this player.
    Return:
      index: The index of this player in the player list.
    """
    filt = [i for (i, item) in enumerate(player_list) if item['id'] == _id]
    index = None if len(filt) == 0 else filt[0]
    return index


def find_random_waiting(_id):
    """
    Randomly choose a 'waiting' player from the list.
    But it can not choose itself(_id).
    Args:
      _id: ID of this player.
    Return:
      index: The index of the chosen player in the player list.
    """
    filt = [i for (i, item) in enumerate(player_list) if (
        item['status'] == 'waiting' and item['id'] != _id)]
    if len(filt):
        index = random.choice(filt)
    else:
        index = None
    return index

# ==================
# Socket-io
# ==================


@socketio.on('connect')
def on_connect():
    """
    Called when the client just connected.
    Create a player and push it to the list.
    Player:
      id: Indentifier of the player.
      status: 'idle', 'waiting', 'playing'.
      ear: Eye aspect ratio.
      rival: The rival's id.
      startTime: The start time of the game.
      end: Whether the game has ended. '1' means end, '0' means playing.
    """
    client_id = request.args['id']
    index = find_by_id(client_id)
    if not index:
        player_list.append({'id': client_id, 'status': 'idle',
                            'ear': 0, 'rival': None, 'startTime': 0, 'end': 0})
    print(client_id, 'connected !')


@socketio.on('_disconnect')
def on_disconnect(message):
    """
    Called when the client just disconnected.
    Delete the player from the list.
    """
    client_id = message['id']
    index = find_by_id(client_id)
    del player_list[index]
    print(client_id, 'disconnected !')


@socketio.on('set_player_wait')
def set_player_wait(message):
    """
    Set the player's status to 'waiting'.
    And then Randomly match a player whose status is also 'waiting'.
    Args:
      id: ID of the player.
    Emit:
      get_rival: Tell the player its rival's id.
    """
    client_id = message['id']
    curr_index = find_by_id(client_id)
    player_list[curr_index]['status'] = 'waiting'
    print(client_id, 'set to waiting...')
    # look for rival
    rival_index = find_random_waiting(client_id)
    if rival_index != None:
        rival_id = player_list[rival_index]['id']
        player_list[curr_index]['rival'] = rival_id
        player_list[curr_index]['status'] = 'playing'

        player_list[rival_index]['rival'] = client_id
        player_list[rival_index]['status'] = 'playing'

        emit('get_rival', {'id': rival_id})
        print(client_id, 'find rival', rival_id)


@socketio.on('set_player_startTime')
def set_player_startTime(message):
    """
    Set the player's startTime.
    Args:
      id: ID of the player.
    """
    client_id = message['id']
    curr_index = find_by_id(client_id)
    player_list[curr_index]['startTime'] = time.time()
    print('startTime', player_list[curr_index])


@socketio.on('send_image')
def send_image(message):
    """
    1. Convert base64 string from client to np image.
    2. Run 'Detector engine' to calculate its value.
    3. Emit new player data to the player.
    Args:
        uri: Base64 string from client.
        id: ID of the player.
    Emit:
        get_arena_data: send arena status to the client including 
          EAR1: Eye aspect ratio of this player.
          EAR2: Eye aspect ratio of its rival.
          elapsed: Elapsed time since start.
          end: Whether the game has ended.
    """
    # ===============
    # Process Base64
    # ===============

    uri, _id = message['uri'], message['id']
    # split header and body
    img_data = uri.split(',')[1]
    img_data = base64.b64decode(img_data)
    image = Image.open(io.BytesIO(img_data))
    # bgr
    array = np.array(image)
    REJECT = 0

    # ===============
    # Set EAR
    # ===============

    index = find_by_id(_id)
    EAR1 = player_list[index]['ear']

    # If the game has ended, we should not update the player.
    if player_list[index]['end'] == 0:
        EAR1 = detector.calculate_ear(array)
        player_list[index]['ear'] = EAR1
        player_list[index]['uri'] = uri

    # Find rival's EAR.
    rival_id = player_list[index]['rival']
    rival_index = find_by_id(rival_id)
    EAR2 = player_list[rival_index]['ear']

    # Calculat elapsed time
    elapsed = time.time() - player_list[index]['startTime']

    # EAR < threshold is determined as 'blink'.
    threshold = 0.20
    # The first 3 second is not counted.
    prepare_time = 3
    if ((EAR1 < threshold and EAR1 > 0) or (EAR2 < threshold and EAR2 > 0)) and elapsed > prepare_time:
        # The game end. Set player's status back to 'idle'.
        player_list[index]['end'] = 1
        player_list[index]['status'] = 'idle'
        player_list[rival_index]['end'] = 1
        player_list[rival_index]['status'] = 'idle'
        emit('get_arena_data', {'EAR1': EAR1, 'EAR2': EAR2, 'elapsed': elapsed, 'end': 1,
                                'uri1': player_list[index]['uri'], 'uri2': player_list[rival_index]['uri']})
    else:
        emit('get_arena_data', {'EAR1': EAR1,
                                'EAR2': EAR2, 'elapsed': elapsed, 'end': 0})

# ==================
# Run flask
# ==================


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=3000,
                 keyfile=os.path.join('backend', 'server.key'), certfile=os.path.join('backend', 'server.crt'))

```



---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/blinkdetection/  

