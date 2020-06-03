import cv2
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Hyperparameter
resize_size = 400.

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = resize_size / gray.shape[1]
    dim = (int(resize_size), int(gray.shape[0] * ratio))    
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    rects = detector(resized, 1)
    print("rects is as follows")
    rects_and_landmarks = []
    for i, rect in enumerate(rects):
        t = rect.top()
        b = rect.bottom()
        l = rect.left()
        r = rect.right()
        rects_and_landmarks.append({})
        rects_and_landmarks[i]['rect'] = [t, b, l ,r]
        rects_and_landmarks[i]['landmark'] = []
        shape = predictor(resized, rect)
        for j in range(68):
            x, y = shape.part(j).x, shape.part(j).y
            rects_and_landmarks[i]['landmark'].append([x,y])
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
        cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
        cv2.imshow('resized', resized)
        print(rects_and_landmarks)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # Landmark on original frame
    for item in rects_and_landmarks:
        t = int(item['rect'][0] / ratio)
        b = int(item['rect'][1] / ratio)
        l = int(item['rect'][2] / ratio)
        r = int(item['rect'][3] / ratio)
        for j in range(68):
            x, y = int(item['landmark'][j][0] / ratio), int(item['landmark'][j][1] / ratio)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

    flipped = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow('frame',flipped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()