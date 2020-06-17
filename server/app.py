from flask import Flask, Response, render_template, request, jsonify
import logging
from local_landmark import FaceMask
import threading
from config import *
import json
import cv2
import numpy as numpy
import requests
from concurrent.futures import ThreadPoolExecutor


lock = threading.Lock()
app = Flask(__name__)

faceMask = FaceMask()

# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
CUR_MASK = SURGICAL_MASK
curFrame = None
rectImg = None
curEmotion = None
landmark = None
gpu_emotion_api_url = 'http://localhost:7007/emotion'

@app.route('/')
def index():
    return render_template('index.html')

def get_frame(maskType=CUR_MASK, showMask=SHOWMASK):
    global lock
    global faceMask 
    global curFrame
    i = 0
    while True:
        with lock:
            #get camera frame
            faceMask.update_frame()
            curFrame = faceMask.show_frame(maskType, SHOWMASK)
            if showMask:
                rectImg = faceMask.detector.get_rectImg()
                landmark = faceMask.detector.get_rectLandmark()
                if rectImg is not None:
                    _, img_encoded = cv2.imencode('.jpg', rectImg)
                    payload = {"landmark": landmark}
                    img_file = {'file': ('image.jpg', img_encoded.tostring(), 'image/jpeg', {'Expires': '0'}),
                                'json': (None, json.dumps(payload), 'application/json'),}
                    # send http request with image and receive response
                    response = requests.post(gpu_emotion_api_url, files=img_file)
                    print(response.text)

                    # print(rectImg.shape)
                    # print(landmark)
                    # i = i + 1
                    # n = str(i)
                    # cv2.imwrite("image_processed_" + n + ".png", rectImg)

            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + curFrame + b'\r\n\r\n')


@app.route('/stream', methods=['GET'])
def stream():
    return Response(get_frame(maskType=CUR_MASK),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/showMask', methods=['POST'])
def showMask():
    global SHOWMASK
    received = request.form.to_dict()
    print(received)
    if 'showMask' not in received:
        print("Inappropriate post request")
        exit()
    if received['showMask'] == 'false':
        print("Inside false")
        SHOWMASK = False
    elif received['showMask'] == 'true':
        print("inside true")
        SHOWMASK = True
    else:
        print("Inappropriate post request")
        print(received)
        exit()   
    print("Showmask is %r" % SHOWMASK)
    response = jsonify({'result': 'success', 'maskStatus': received})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response






def send_image():
    global curFrame
    global curEmotion
    if curFrame is not None:
        return True

executor = ThreadPoolExecutor(max_workers=1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)