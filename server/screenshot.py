import numpy as np
import cv2
from mss import mss
from PIL import Image
import dlib
import time
import json
from io import BytesIO
import pyscreenshot as ImageGrab
import threading
import requests

sct = mss()
num_monitors = len(sct.monitors)

# capture screen of the last monitor
# in this case, monitor 2
bounding_box = sct.monitors[-1]

# hyperparameter
resize_size = 800.

# Initiate dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# mode
DEBUGMODE = 1
NONDEBUG = 2

MODE = DEBUGMODE

# Class Declaration
class ScreenAnalyzer:
    def __init__(self):
        self.feature = [] # saves face rectangles and landmarks
        # self.org_feature = [] # feature elements transformed to fit original window
        # self.org_nosePoint = [] # coordinates for noses
        # self.transMat = []
        # self.rotMat = []

    def resetInternals(self):
        self.feature = []
        # self.org_feature = []
        # self.org_nosePoint = []
        # self.transMat = []
        # self.rotMat = [] 

    # Detect landmark and headpose
    def detect(self, frame):
        self.resetInternals()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 1)
        print("rects ", len(rects))
        for i, rect in enumerate(rects):
            t = rect.top()
            b = rect.bottom()
            l = rect.left()
            r = rect.right()
            # org_t = int(t/ratio)
            # org_b = int(b/ratio)
            # org_l = int(l/ratio)
            # org_r = int(r/ratio)

            self.feature.append({})
            self.feature[i]['rect'] = [t, b, l ,r]
            self.feature[i]['landmark'] = []

            # self.org_feature.append({})
            # self.org_feature[i]['rect'] = [org_t, org_b, org_l ,org_r]
            # self.org_feature[i]['landmark'] = []
            
            # Detect landmark
            shape = predictor(gray, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                # org_x, org_y = int(x/ratio), int(y/ratio)
                self.feature[i]['landmark'].append([x, y])
                # self.org_feature[i]['landmark'].append([org_x, org_y])
                if MODE == DEBUGMODE:
                    cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)

            # save results for debugging
            if MODE == DEBUGMODE and len(self.feature) > 0:
                cv2.rectangle(gray, (l, t), (r, b), (0, 255, 0), 2)
                #gray = cv2.flip(gray, 1)
                ratio = resize_size / gray.shape[1]
                dim = (int(resize_size), int(gray.shape[0] * ratio))
                resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

                cv2.namedWindow("feature")
                cv2.moveWindow("feature", 40,30)
                cv2.imshow('feature', resized)
                # cv2.imwrite('feature.png', gray)
                
    
    def get_feature(self):
        return self.feature
    
class ScreenSender(threading.Thread):
	def __init__(self, image, addr):
		self.image = image
		self.addr = addr

	def send(self):
		# create url
		test_url = addr + '/screen'

		# prepare headers for http request
		content_type = 'image/jpeg'
		headers = {'content-type': content_type}

		_, img_encoded = cv2.imencode('.jpg', self.image)

		img_file = {'file': ('screen.jpg', img_encoded.tostring(), 'image/jpeg', {'Expires': '0'})}

		# send http request with image and receive response
		response = requests.post(test_url, files = img_file)

		# decode response
		print(response.text)



# class FeatureSender(threading.Thread):

class Recorder():
    def __init__(self):
        self.sct = mss()
        self.bounding_box = self.sct.monitors[-1]
        self.sct_img = np.array(self.sct.grab(self.bounding_box))

    def capture_screen(self):
        self.sct_img = np.array(self.sct.grab(self.bounding_box))

    def get_captured_screen(self):
        return self.sct_img

class ScreenCapture():
    def __init__(self):
        self.recorder = Recorder()

    def update_frame(self):
        self.recorder.capture_screen()

    def show_frame(self):
        curFrame = self.recorder.get_captured_screen()
        ret, jpeg = cv2.imencode('.jpg', curFrame)
        return jpeg.tobytes()

def capture_image():

    # Get raw pixels from the screen
    sct_img = sct.grab(bounding_box)    
    return np.array(sct_img)



if __name__ == '__main__':
    features = ScreenAnalyzer()
    while "Screen capturing":
        img = capture_image()
        if MODE == DEBUGMODE:
        	cv2.imwrite('screen.png', img)
        features.detect(img)

        # send image features
        

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break