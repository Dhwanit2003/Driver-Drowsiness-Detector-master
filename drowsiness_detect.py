
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2

pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

EYE_ASPECT_RATIO_THRESHOLD = 0.3

EYE_ASPECT_RATIO_CONSEC_FRAMES = 40

COUNTER = 0



face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_capture = cv2.VideoCapture(0)

time.sleep(2)
total_closed_time = 0
music_playing = False

fps = video_capture.get(cv2.CAP_PROP_FPS)

while(True):
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            total_closed_time += 1 

            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Are Closed", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                cv2.putText(frame, f"Total Closed Time: {total_closed_time // fps} seconds", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if total_closed_time // fps >= 3 and not music_playing:
                    pygame.mixer.music.play(-1)
                    music_playing = True
        else:
            pygame.mixer.music.stop()
            music_playing = False
            COUNTER = 0
            total_closed_time = 0 
            cv2.putText(frame, "Eyes Are Open", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == 27: 
        break

video_capture.release()
cv2.destroyAllWindows()
