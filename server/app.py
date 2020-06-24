from flask import Flask, Response, render_template, request, jsonify
import logging
from local_landmark import FaceMask
import threading
from config import *
import json
import cv2
import numpy as numpy
import requests
import time
from concurrent.futures import ThreadPoolExecutor


lock = threading.Lock()
app = Flask(__name__)

faceMask = FaceMask()

# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
FUNMODE = True
CUR_MASK = SURGICAL_MASK
curFrame = None
rectImg = None
curEmotion = None
landmark = None
failTime = 0
lastPostSuccess = True
gpu_emotion_api_url = 'http://localhost:7007/emotion'

@app.route('/')
def index():
    return render_template('index.html')

def get_frame(maskType=CUR_MASK):
    global lock
    global faceMask 
    global curFrame
    global failTime
    global lastPostSuccess
    i = 0
    while True:
        with lock:
            #get camera frame
            faceMask.update_frame()
            curFrame = faceMask.show_frame(maskType, showMask=SHOWMASK, funMode=FUNMODE)
            if showMask:
                rectImg = faceMask.detector.get_rectImg()
                landmark = faceMask.detector.get_rectLandmark()
                if rectImg is not None:
                    _, img_encoded = cv2.imencode('.jpg', rectImg)
                    payload = {"landmark": landmark}
                    img_file = {'file': ('image.jpg', img_encoded.tostring(), 'image/jpeg', {'Expires': '0'}),
                                'json': (None, json.dumps(payload), 'application/json'),}
                    # send http request with image and receive response
                    if lastPostSuccess or (time.perf_counter() - failTime) > 1.0:
                        try:
                            response = requests.post(gpu_emotion_api_url, files=img_file)
                            print(response.text)
                            lastPostSuccess = True
                        except requests.exceptions.HTTPError as errh:
                            print ("Http Error:",errh)
                            failTime = time.perf_counter()
                            lastPostSuccess = False
                        except requests.exceptions.ConnectionError as errc:
                            print ("Error Connecting:",errc)
                            failTime = time.perf_counter()
                            lastPostSuccess = False
                        except requests.exceptions.Timeout as errt:
                            print ("Timeout Error:",errt)
                            failTime = time.perf_counter()
                            lastPostSuccess = False
                        except requests.exceptions.RequestException as err:
                            print ("OOps: Something Else",err)
                            failTime = time.perf_counter()
                            lastPostSuccess = False

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


@app.route('/userButton', methods=['POST'])
def handleUserButton():
    global SHOWMASK
    global FUNMODE
    received = request.form.to_dict()
    if 'showMask' in received:
      if received['showMask'] == 'false':
          print("Inside false of showmask")
          SHOWMASK = False
      elif received['showMask'] == 'true':
          print("inside true of showmask")
          SHOWMASK = True
      else:
        print("Inappropriate post request")
        print(received)
        exit()  

    if 'funMode' in received:
      if received['funMode'] == 'false':
          print("Inside false of funmode")
          FUNMODE = False
      elif received['funMode'] == 'true':
          print("inside true of funmode")
          FUNMODE = True
      else:
        print("Inappropriate post request")
        print(received)
        exit()  
  
    print(received)
    print("Showmask is %r" % SHOWMASK)
    print("Funmode is %r" % FUNMODE)
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