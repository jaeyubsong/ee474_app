from flask import Flask, Response, render_template
from local_landmark import FaceMask
app = Flask(__name__)

faceMask = FaceMask()

# Global variabble
SURGICAL_MASK = 1

@app.route('/')
def index():
    return render_template('index.html')

def get_frame(maskType):
    global faceMask 
    while True:
        #get camera frame
        faceMask.update_frame()
        frame = faceMask.show_frame(maskType)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(get_frame(maskType=SURGICAL_MASK),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)