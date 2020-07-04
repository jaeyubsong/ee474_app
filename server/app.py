from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import logging
from local_landmark import FaceMask
import threading
from config import *
from util import checkIfInt
import json
import cv2
import numpy as np
import requests
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
import threading
import pyscreenshot as ImageGrab


# # grab fullscreen
# im = ImageGrab.grab(bbox=(10, 10, 2000, 1000))  # X1,Y1,X2,Y2

# # save image file
# im.save('box.png')
# exit()



lock = threading.Lock()
app = Flask(__name__)

faceMask = FaceMask()
faceMask.start()


# Global variabble
CAM_ON = False
SHOWMASK = False
FUNMODE = False
CUR_MASK = SURGICALMASK
cur_img_byte = None
rectImg = None
curEmotion = 0
landmark = None
failTime = 0
lastPostSuccess = True
emotion_stat = [0, 0, 0, 0, 0]
gpu_emotion_api_url = 'http://localhost:7007/emotion'
gpu_audienceInfo_api_url = 'http://localhost:7008/audienceInfo'

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    global lock
    global faceMask 
    # global cur_img_byte
    global CAM_ON
    global failTime
    global lastPostSuccess
    global curEmotion
    i = 0
    while True:
        print("get frame called")
        time.sleep(0.05)
        with lock:
            # print("Update frame")
            #get camera frame
            curFrame_byte = faceMask.show_frame(maskType=CUR_MASK, showMask=SHOWMASK, funMode=FUNMODE)
            # cur_img_byte = copy.deepcopy(tmp_byte)
            if curFrame_byte is None:
                print("curFrame_byte of none is returned")
                return
            CAM_ON = True
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + curFrame_byte + b'\r\n\r\n')


@app.route('/stream', methods=['GET'])
def stream():
    print("ASDSADD")
    return Response(stream_with_context(get_frame()),
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
    global SHOWMASK, FUNMODE, CUR_MASK
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

    if 'maskType' in received:
        if not checkIfInt(received['maskType']):
            print("Inappropriate data value")
            print(received)
            exit()
        else:
            mask_num = int(received['maskType'])
            CUR_MASK = mask_num
  
    print(received)
    print("Showmask is %r" % SHOWMASK)
    print("Funmode is %r" % FUNMODE)
    print("Curmask is %r" % CUR_MASK)
    response = jsonify({'result': 'success', 'maskStatus': received})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response


@app.route('/myEmotion', methods=['POST'])
def getMyEmotion():
    # global curEmotion
    response = jsonify({'result': 'success', 'myEmotion': curEmotion})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response


@app.route('/getServerData', methods=['POST'])
def getServerData():
    global CAM_ON
    app.logger.info("get server data called")
    print("get server data called")
    # global curEmotion
    response = jsonify({'result': 'success', 'myEmotion': curEmotion, 'astonished': emotion_stat[0], 
                        'unsatisfied': emotion_stat[1], 'joyful': emotion_stat[2], 
                        'neutral': emotion_stat[3], 'sad': emotion_stat[4]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    CAM_ON = True
    return response



def get_emotion():
    global cur_img_byte
    global curEmotion
    global failTime
    global lastPostSuccess
    while True:
        print("get_emotion()")
        time.sleep(0.1)
        fail_elapsed = time.perf_counter() - failTime
        if CAM_ON == True:# and (lastPostSuccess or fail_elapsed > 5):
            print("Inside get emotion")
            rectImg = faceMask.detector.get_rectImg()
            if rectImg is None:
                continue
            _, cur_img_encoded = cv2.imencode('.jpg', rectImg)
            img_file = {'file': ('image.jpg', cur_img_encoded.tostring(), 'image/jpeg', {'Expires': '0'})}
            try:
                response = requests.post(gpu_emotion_api_url, files=img_file)
                print(response.text)
                json_response = json.loads(response.text)
                print(json_response)
                lastPostSuccess = True
                curEmotion = json_response["emotion"]
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
        else:
            print("Cam is off")


def get_audienceInfo():
    global cur_img_byte
    global curEmotion
    global failTime
    global lastPostSuccess
    global emotion_stat
    while True:
        print("get_audienceInfo()")
        # part of the screen
        screenShot = ImageGrab.grab(bbox=(10, 10, 2000, 1000))  # X1,Y1,X2,Y2
        screenShot.save('box.png')

        screenShot_cv = np.array(screenShot.getdata(), dtype = 
'uint8').reshape((screenShot.size[1], screenShot.size[0], 3))
        # save image file
        # time.sleep(0.)
        fail_elapsed = time.perf_counter() - failTime
        _, cur_img_encoded = cv2.imencode('.jpg', screenShot_cv)
        img_file = {'file': ('image.jpg', cur_img_encoded.tostring(), 'image/jpeg', {'Expires': '0'})}
        try:
            response = requests.post(gpu_audienceInfo_api_url, files=img_file)
            print(response.text)
            json_response = json.loads(response.text)
            print(json_response)
            lastPostSuccess = True
            emotion_stat[0] = json_response["astonished"]
            emotion_stat[1] = json_response["unsatisfied"]
            emotion_stat[2] = json_response["joyful"]
            emotion_stat[3] = json_response["neutral"]
            emotion_stat[4] = json_response["sad"]
            print("Emotion stat is as follows")
            print(emotion_stat)
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

# executor = ThreadPoolExecutor(max_workers=2)
# a = executor.submit(get_emotion)

if __name__ == '__main__':
    app.debug = False
    threading.Thread(target=get_emotion).start()
    threading.Thread(target=get_audienceInfo).start()
    # threading.Thread(target=app.run, kwargs=dict(host='0.0.0.0', port=5007, debug=False, use_reloader=False, threaded=True)).start()
    app.run(host='0.0.0.0', port=5007, debug=False, use_reloader=False, threaded=True)