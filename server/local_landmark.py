import sys
import cv2
import numpy as np
import dlib
from random import randint
import argparse
from threading import Thread
import time
from copy import copy, deepcopy
from config import *
from face_alignment import get_angle, rotate_opencv


# Hyperparameter
resize_size = 150.
send_size = 48

# global variable
texture_id = 0
input_vid_path = None

global_rect = []


# Initiate dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D model points.
modelPoints = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
size = [resize_size, resize_size]
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camMat = np.array(
[[focal_length, 0, center[0]],
[0, focal_length, center[1]],
[0, 0, 1]], dtype = "double"
)


dist_coef = np.zeros((4,1)) # Assuming no lens distortion

# If debug is true, opencv.show activates
DEBUG = False

LANDMARK_MODE = 1
MASK_MODE = 2


HAPPY_EMOJI = 11

CAM = 1
VIDEO = 2

# debug parameters
MODE = MASK_MODE
INPUT = CAM


blindFold = cv2.imread('./res/mask/blindFold.png', cv2.IMREAD_UNCHANGED)
bunny = cv2.imread('./res/mask/bunny.png', cv2.IMREAD_UNCHANGED)
darthVadar = cv2.imread('./res/mask/darthVadar.png', cv2.IMREAD_UNCHANGED)
grouchoGlasses = cv2.imread('./res/mask/grouchoGlasses.png', cv2.IMREAD_UNCHANGED)
guyFawkes = cv2.imread('./res/mask/guyFawkes.png', cv2.IMREAD_UNCHANGED)
halloween = cv2.imread('./res/mask/halloween.png', cv2.IMREAD_UNCHANGED)
surgicalMask = cv2.imread('./res/mask/surgicalMask.png', cv2.IMREAD_UNCHANGED)


happy_emoji = cv2.imread('./res/emoji/e3.png', cv2.IMREAD_UNCHANGED)
rows, cols, channels = surgicalMask.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4V')
out = None
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))


# function from https://stackoverflow.com/a/54058766
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def clip(input_val, max_val):
    return max(0, min(input_val, max_val))


def resize_image(inputImg, width=None, height=None):
    org_width, org_height, _ = inputImg.shape
    if width is None and height is None:
        print("Error as both width and height is NONE")
        exit(-1)
    elif width is None:
        ratio = 1.0 * height / org_height
        width = int(ratio * org_width)
    elif height is None:
        ratio = 1.0 * width / org_width
        height = int(ratio * org_height)
    dim = (int(width), int(height))
    print(dim)
    resized = cv2.resize(inputImg, dim, interpolation=cv2.INTER_AREA)
    return resized




def put_mask(inputImg, landmark, maskType):
    global blindFold, bunny, darthVadar, grouchoGlasses, guyFawkes, halloween, surgicalMask
    print(inputImg.shape)
    max_y, max_x, _ = inputImg.shape
    print("MAx is")
    print(max_x)
    print(max_y)
    if maskType == BLINDFOLD:
        left_jaw = clip(landmark[0][0] - 15, max_x)
        right_jaw = clip(landmark[16][0] + 15, max_x)
        low_nose_y = landmark[33][1]
        right_eyebrow_y = clip(landmark[23][1] - 15, max_y)
        width = right_jaw - left_jaw
        height = (low_nose_y - right_eyebrow_y)
        mask = blindFold.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == BUNNY:
        left_jaw = clip(landmark[0][0] - 15, max_x)
        right_jaw = clip(landmark[16][0] + 15, max_x)
        chin_y = landmark[8][1]
        right_eyebrow_y = clip(landmark[24][1] - 300, max_y)
        width = right_jaw - left_jaw
        height = (chin_y - right_eyebrow_y)
        mask = bunny.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == DARTHVADAR:
        left_jaw = clip(landmark[0][0] - 160, max_x)
        right_jaw = clip(landmark[16][0] + 160, max_x)
        chin_y = clip(landmark[8][1] + 150, max_y)
        right_eyebrow_y = clip(landmark[24][1] - 250, max_y)
        width = right_jaw - left_jaw
        height = (chin_y - right_eyebrow_y)
        mask = darthVadar.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == GROUCHOGLASSES:
        left_jaw = clip(landmark[0][0], max_x)
        right_jaw = clip(landmark[16][0], max_x)
        outer_edge_lip_y = clip(landmark[51][1], max_y)
        right_eyebrow_y = clip(landmark[24][1], max_y)
        width = right_jaw - left_jaw
        height = (outer_edge_lip_y - right_eyebrow_y)
        mask = grouchoGlasses.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == GUYFAWKES:
        left_jaw = clip(landmark[0][0] - 15, max_x)
        right_jaw = clip(landmark[16][0] + 15, max_x)
        chin_y = landmark[8][1]
        right_eyebrow_y = clip(landmark[24][1] - 40, max_y)
        width = right_jaw - left_jaw
        height = (chin_y - right_eyebrow_y)
        mask = guyFawkes.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == HALLOWEEN:
        left_jaw = clip(landmark[0][0]-40, max_x)
        right_jaw = clip(landmark[16][0]+40, max_x)
        right_eyebrow_y = clip(landmark[24][1]-90, max_y)
        lower_nose_y = clip(landmark[33][1]+20, max_y)
        width = right_jaw - left_jaw
        height = (lower_nose_y - right_eyebrow_y)
        mask = halloween.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = right_eyebrow_y
    elif maskType == SURGICALMASK:
        left_jaw = clip(landmark[0][0] - 15, max_x)
        right_jaw = clip(landmark[16][0] + 15, max_x)
        left_eye_y = clip(landmark[38][1] - 20, max_y)
        chin_y = clip(landmark[8][1] + 70, max_y)
        width = right_jaw - left_jaw
        height = chin_y - left_eye_y
        mask = surgicalMask.copy()
        mask = resize_image(mask, width=width, height=height)
        x_val = left_jaw
        y_val = left_eye_y

    h, w = mask.shape[:2]
    angle=get_angle(landmark[48], landmark[54])
    mask = rotate_opencv(mask, (w/2, h/2), -angle)
    # mask = resize_image(mask, width=width)
    newImg = overlay_transparent(inputImg, mask, x_val, y_val)
    return newImg


def put_bg_effect(inputImg, landmark, bgType):
    global happy_emoji
    if bgType == HAPPY_EMOJI:
        emoji_size = 50
        magic_num = 10
        left_cheek_x = landmark[2][0] - magic_num
        right_cheek_x = landmark[14][0] + magic_num
        upper_lip_y = landmark[27][1]
        width = right_cheek_x - left_cheek_x
        mask = happy_emoji.copy()
        mask = resize_image(mask, width=emoji_size)
        print(inputImg.shape)
        imgWidth = inputImg.shape[1]
        imgHeight = inputImg.shape[0]
        howMany = randint(5, 20)
        for i in range(howMany):
            x_pos = randint(emoji_size, imgWidth-emoji_size)
            y_pos = randint(emoji_size, imgHeight-emoji_size)
            newImg = overlay_transparent(inputImg, mask, x_pos, y_pos)
    else:
        newImg = inputImgq
    return newImg



class Cam:
    def __init__(self):
        if INPUT == CAM:
            self.capture = cv2.VideoCapture(0)
        elif INPUT == VIDEO:
            self.capture = cv2.VideoCapture(input_vid_path)
        # self.curFrame = self.capture.read()[1]
        self.curFrame = None
        self.curSmallFrame = None
    

    def update_frame(self):
        # if self.curFrame:
            self.curFrame = self.capture.read()[1]

        # else:
        #     out.release()
        #     cv2.destroyAllWindows()
    
    def get_curFrame(self):
        return self.curFrame

landmark_per_rect = 5
cur_ld = 0
rects = []

class Detector:
    def __init__(self):
        self.feature = [] # saves face rectangles and landmarks
        self.org_feature = [] # feature elements transformed to fit original window
        self.org_nosePoint = [] # coordinates for noses
        self.transMat = []
        self.rotMat = []
        self.rectImg = None
        self.rectLandmark = []
        self.rects = None

    def resetInternals(self):
        self.feature = []
        self.org_feature = []
        self.org_nosePoint = []
        self.transMat = []
        self.rotMat = [] 
        self.rectImg = None
        self.rectLandmark = []


    # Detect landmark and headpose
    def detect(self, frame):
        global rects
        global cur_ld, landmark_per_rect
        # self.resetInternals()
        if frame is None:
            return
        gray = frame
        ratio = resize_size / gray.shape[1]
        dim = (int(resize_size), int(gray.shape[0] * ratio))
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        # rects = detector(resized, 3)
        if cur_ld < landmark_per_rect:
            rects = detector(resized, 3)
            # self.rects = copy.deepcopy(rects)
            cur_ld = cur_ld + 1
        else:
            cur_ld = 0
        if len(rects) == 0:
            print("Returned as there is no rect")
            return
        # else:
        #     self.resetInternals()
        print(rects)
        # copied_rect = copy.deepcopy(rects)
        # print(copied_rect)
        # past_feature = copy.deepcopy(self.feature)
        # past_org_feature = copy.deepcopy(self.org_feature)
        # past_transMat = copy.deepcopy(self.transMat)
        # past_rotMat = copy.deepcopy(self.rotMat)
        # past_rectImg = self.rectImg
        # past_rectLandmark = copy.deepcopy(self.rectLandmark)

        for i, rect in enumerate(rects):
            t = rect.top()
            b = rect.bottom()
            l = rect.left()
            r = rect.right()
            org_t = int(t/ratio)
            org_b = int(b/ratio)
            org_l = int(l/ratio)
            org_r = int(r/ratio)
            shape = predictor(resized, rect)
            if shape is None:
                print("No landmark detected")
                return
            else:
                self.resetInternals()

            self.feature.append({})
            self.feature[i]['rect'] = [t, b, l ,r]
            self.feature[i]['landmark'] = []

            self.org_feature.append({})
            self.org_feature[i]['rect'] = [org_t, org_b, org_l ,org_r]
            self.org_feature[i]['landmark'] = []

            # Make rectangle as image
            rect_ratio_x = 1.0 * send_size / (r-l)
            rect_ratio_y = 1.0 * send_size / (b-t)

            # print("rect raxio x: %f, y: %f" % (rect_ratio_x, rect_ratio_y))
            rect_dim = (send_size, send_size)
            rect_img = resized[t:b, l:r]
            rect_img = cv2.resize(rect_img, rect_dim, interpolation = cv2.INTER_AREA)
            # self.rectImg = rect_img


            # Detect landmark                

            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                org_x, org_y = int(x/ratio), int(y/ratio)
                # print("x-l is %d" % (x-l))
                rect_x, rect_y = int((x - l) * rect_ratio_x), int(1.0 * (y - t) * rect_ratio_y)
                # cv2.circle(rect_img, (rect_x, rect_y), 1, (0, 0, 255), -1)

                self.feature[i]['landmark'].append([x, y])
                self.org_feature[i]['landmark'].append([org_x, org_y])
                self.rectLandmark.append([rect_x, rect_y])
                if MODE == LANDMARK_MODE:
                    cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
            self.rectImg = rect_img
            # Detect headpose
            nose_tip = self.feature[i]['landmark'][33]
            chin = self.feature[i]['landmark'][8]
            # left_eye_left_corner = self.feature[i]['landmark'][45]
            # right_eye_right_corner = self.feature[i]['landmark'][36]
            # left_mouth_corner = self.feature[i]['landmark'][54]
            # right_mouth_corner = self.feature[i]['landmark'][48]
            right_eye_right_corner = self.feature[i]['landmark'][45]
            left_eye_left_corner = self.feature[i]['landmark'][36]
            right_mouth_corner = self.feature[i]['landmark'][54]
            left_mouth_corner = self.feature[i]['landmark'][48]
            image_points = np.array([
                (nose_tip[0], nose_tip[1]),
                (chin[0], chin[1]),
                (left_eye_left_corner[0], left_eye_left_corner[1]),
                (right_eye_right_corner[0], right_eye_right_corner[1]),
                (left_mouth_corner[0], left_mouth_corner[1]),
                (right_mouth_corner[0], right_mouth_corner[1])
            ], dtype="double")

            _, self.rotMat, self.transMat = cv2.solvePnP(modelPoints, image_points, camMat, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
            nosePoints, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), self.rotMat, self.transMat, camMat, dist_coef)

            # print("Transmat:")
            # print(self.transMat)
            # print("rotmat:")
            # print(self.rotMat)
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nosePoints[0][0][0]), int(nosePoints[0][0][1]))
            org_p1 = (int(p1[0] / ratio), int(p1[1] / ratio))
            org_p2 = (int(p2[0] / ratio), int(p2[1] / ratio))
            self.org_nosePoint.append([])
            self.org_nosePoint[-1].append(org_p1)
            self.org_nosePoint[-1].append(org_p2)

            if MODE == LANDMARK_MODE:
                cv2.line(resized, p1, p2, (255,0,0), 2)


            if DEBUG and len(self.feature) > 0:
                cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
                resized = cv2.flip(resized, 1)
                cv2.imshow('resized', resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def get_feature(self):
        return self.feature
    
    def get_org_feature(self):
        return self.org_feature
    
    def get_org_nosePoint(self):
        return self.org_nosePoint
    
    def get_rectImg(self):
        return self.rectImg
    
    def get_rectLandmark(self):
        return self.rectLandmark


ii = 0

class FaceMask:
    def __init__(self, cam_fps=30, detect_fps=10):
        self.cam = Cam()
        self.detector = Detector()
        self.cam_fps = cam_fps
        self.detect_fps = detect_fps
        self.ms_between_cam = 1./self.cam_fps * 1000
        self.ms_between_detect = 1./self.detect_fps * 1000
        self.last_update = 0
    
    def start(self):
        self.last_update = time.perf_counter_ns()
        Thread(target=self.update_cam, args=()).start()
        Thread(target=self.update_landmark, args=()).start()
        # cur_frame = self.cam.get_curFrame()
        # self.detector.detect(cur_frame)


    def update_cam(self):
        while True:
            cur_time = time.perf_counter_ns()
            elapsed_time_ms = (cur_time - self.last_update) / 1000000
            if elapsed_time_ms > self.ms_between_cam:
                print("Update cur frame")
                self.cam.update_frame()

    def update_landmark(self):
        while True:
            cur_time = time.perf_counter_ns()
            elapsed_time_ms = (cur_time - self.last_update) / 1000000
            if elapsed_time_ms > self.ms_between_detect:
                print("Update landmark (elapsed time: %d" % elapsed_time_ms)
                self.detector.detect(self.cam.get_curFrame())
                self.last_update = cur_time


    def update_frame(self):
        self.cam.update_frame()
        cur_frame = self.cam.get_curFrame()
        # if cur_frame is None:
        #     out.release()
        #     exit()
        self.detector.detect(cur_frame)
    
    def show_frame(self, maskType=None, showMask=True, funMode=True, effectType=HAPPY_EMOJI):
        print("Showframe called with maskType: %d, showMask: %r, funMode: %r" % (maskType, showMask, funMode))
        global surgicalMask
        global happy_emoji
        curFrame = self.cam.get_curFrame()
        if curFrame is None:
            return
        landmarks = self.detector.get_org_feature()


        for i in range(len(landmarks)):
            landmark = landmarks[i]['landmark']
            if MODE == LANDMARK_MODE:
                # print(landmarks)
                for i in range(len(landmarks)):
                    for j in range(68):
                        cv2.circle(curFrame, (landmarks[i]['landmark'][j][0], landmarks[i]['landmark'][j][1]), 1, (0, 0, 255), -1)
                    t = landmarks[i]['rect'][0]
                    b = landmarks[i]['rect'][1]
                    l = landmarks[i]['rect'][2]
                    r = landmarks[i]['rect'][3]
                    cv2.rectangle(curFrame, (l, t), (r, b), (0, 255, 0), 2)
                    point = self.detector.get_org_nosePoint()
                    cv2.line(curFrame, point[i][0], point[i][1], (255,0,0), 2)
            if MODE == MASK_MODE and showMask == True:
                # cv2.circle(curFrame, (landmark[0][0], landmark[0][1]), 10, (0, 0, 255), -1)
                # t = landmarks[i]['rect'][0]
                # b = landmarks[i]['rect'][1]
                # l = landmarks[i]['rect'][2]
                # r = landmarks[i]['rect'][3]
                # cv2.rectangle(curFrame, (l, t), (r, b), (0, 255, 0), 2)
                if len(landmark) > 0:
                    curFrame = put_mask(inputImg=curFrame, landmark=landmark, maskType=maskType)

            if funMode == True:
                if effectType == HAPPY_EMOJI and len(landmark) > 0:
                    curFrame = put_bg_effect(inputImg=curFrame, landmark=landmark, bgType=HAPPY_EMOJI)

            
        # elif MODE == MASK_MODE and showMask == False:
        #     curFrame = cv2.flip(curFrame, 1)
        #     ret, jpeg = cv2.imencode('.jpg', curFrame)
        

        # if funMode == True:
        #     if effectType == HAPPY_EMOJI:
        #         curFrame = put_emoji_effect(inputImg=curFrame, landmark=landmark)

        curFrame = cv2.flip(curFrame, 1)
        ret, jpeg = cv2.imencode('.jpg', curFrame)
        if DEBUG:
            cv2.imshow('original', curFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        if out is not None:
            out.write(curFrame)
        print("Returning processed jpeg")
        return jpeg.tobytes()

    def main(self):
        while True:
            self.update_frame()
            self.show_frame(maskType=SURGICALMASK, showMask=False, funMode=True)

 
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=False, type=int, default=2, help="Landmark mode: 1, Mask mode: 2")
    ap.add_argument("-i", "--input", required=False, type=str, default=None, help="input video")
    ap.add_argument("-o", "--output", required=False, type=str, default=None, help="output video")

    args = vars(ap.parse_args())
    if args['mode'] == LANDMARK_MODE:
        print("Landmark mode")
    elif args['mode'] == MASK_MODE:
        print("Mask mode")
    DEBUG = True
    MODE = args['mode']
    if args['input'] is not None:
        print("Arg is not none")
        input_vid_path = args['input']
        INPUT = VIDEO
    if args['output'] is not None:
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
    faceMask = FaceMask()
    faceMask.main()