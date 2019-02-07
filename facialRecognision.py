import dlib
import cv2 as cv
import numpy as np
from imutils import face_utils

# Training model
p = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

# feed our training model to dlib
predictor = dlib.shape_predictor(p)

camera = cv.VideoCapture(0)

while(True):
    ret, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)


    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()

print('-------------- Finished ------------------')
