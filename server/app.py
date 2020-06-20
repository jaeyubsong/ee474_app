from flask import Flask, Response, render_template, request, jsonify
import logging
from local_landmark import FaceMask
from screenshot import ScreenCapture
import threading

lock = threading.Lock()
app = Flask(__name__)

faceMask = FaceMask()
capture = ScreenCapture()

# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
CUR_MASK = SURGICAL_MASK

@app.route('/')
def index():
    return render_template('index.html')

def get_frame(maskType=CUR_MASK, showMask=SHOWMASK):
    global lock
    global faceMask 
    while True:
        with lock:
            #get camera frame
            faceMask.update_frame()
            frame = faceMask.show_frame(maskType, SHOWMASK)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_screen():
    global lock
    global capture
    while "Screen Capturing":
        with lock:
            capture.update_frame()
            frame = capture.show_frame()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/stream', methods=['GET'])
def stream():
    return Response(get_frame(maskType=CUR_MASK),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recorder', methods=['GET'])
def recorder():
    return Response(get_screen(), mimetype='multipart/x-mixed-replace; boundary=frame')


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, threaded=True)