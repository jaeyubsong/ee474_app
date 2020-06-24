import sys
import cv2
import numpy as np
import dlib
from random import randint
import argparse
from copy import copy, deepcopy

# Hyperparameter
resize_size = 200.
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

SURGICAL_MASK = 1
HAPPY_EMOJI = 2

CAM = 1
VIDEO = 2

# debug parameters
MODE = MASK_MODE
INPUT = CAM


surgical_mask = cv2.imread('./res/mask/surgical_mask.png', cv2.IMREAD_UNCHANGED)
happy_emoji = cv2.imread('./res/emoji/e3.png', cv2.IMREAD_UNCHANGED)
rows, cols, channels = surgical_mask.shape

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


def resize_image(inputImg, width=None, height=None):
    org_width, org_height, _ = inputImg.shape
    if width is None and height is None:
        print("Error as both width and height is NONE")
        exit(-1)
    elif width is None:
        ratio = 1.0 * height / org_height
        width = int(ratio * org_width)
        dim = (width, height)
    else:
        ratio = 1.0 * width / org_width
        height = int(ratio * org_height)
        dim = (width, height)
    resized = cv2.resize(inputImg, dim, interpolation=cv2.INTER_AREA)
    return resized


def put_mask(inputImg, landmark, maskType):
    global surgical_mask
    if maskType == SURGICAL_MASK:
        magic_num = 10
        left_cheek_x = landmark[2][0] - magic_num
        right_cheek_x = landmark[14][0] + magic_num
        upper_lip_y = landmark[27][1]
        width = right_cheek_x - left_cheek_x
        mask = surgical_mask.copy()
        mask = resize_image(mask, width=width)
    newImg = overlay_transparent(inputImg, mask, left_cheek_x + magic_num, upper_lip_y)
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


class Detector:
    def __init__(self):
        self.feature = [] # saves face rectangles and landmarks
        self.org_feature = [] # feature elements transformed to fit original window
        self.org_nosePoint = [] # coordinates for noses
        self.transMat = []
        self.rotMat = []
        self.rectImg = None
        self.rectLandmark = []

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
        # self.resetInternals()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = resize_size / gray.shape[1]
        dim = (int(resize_size), int(gray.shape[0] * ratio))
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        rects = detector(resized, 1)
        if len(rects) == 0:
            return
        else:
            self.resetInternals()
        for i, rect in enumerate(rects):
            t = rect.top()
            b = rect.bottom()
            l = rect.left()
            r = rect.right()
            org_t = int(t/ratio)
            org_b = int(b/ratio)
            org_l = int(l/ratio)
            org_r = int(r/ratio)

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
            shape = predictor(resized, rect)
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


class FaceMask:
    def __init__(self):
        self.cam = Cam()
        self.detector = Detector()

    def update_frame(self):
        self.cam.update_frame()
        cur_frame = self.cam.get_curFrame()
        if cur_frame is None:
            out.release()
            exit()
        self.detector.detect(cur_frame)
    
    def show_frame(self, maskType=None, showMask=True, funMode=True, effectType=HAPPY_EMOJI):
        global surgical_mask
        global happy_emoji
        curFrame = self.cam.get_curFrame()
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
                if maskType == SURGICAL_MASK:
                    curFrame = put_mask(inputImg=curFrame, landmark=landmark, maskType=SURGICAL_MASK)
                elif maskType == HAPPY_EMOJI:
                    magic_num = 10
                    left_cheek_x = landmark[2][0] - magic_num
                    right_cheek_x = landmark[14][0] + magic_num
                    upper_lip_y = landmark[27][1]
                    width = right_cheek_x - left_cheek_x
                    mask = happy_emoji.copy()
                    mask = resize_image(mask, width=width)
                    curFrame = overlay_transparent(curFrame, mask, left_cheek_x + magic_num, upper_lip_y)
            if funMode == True:
                if effectType == HAPPY_EMOJI:
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
        return jpeg.tobytes()

    def main(self):
        while True:
            self.update_frame()
            self.show_frame(maskType=SURGICAL_MASK, showMask=False, funMode=True)

 
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