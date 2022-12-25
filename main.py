import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pyag
from pynput.mouse import Button, Controller
import dlib
import imutils
from turtle import delay
from imutils import face_utils
import autopy
from utils import *

mp_face_mesh = mp.solutions.face_mesh #  gtetting face mesh example that contains the 478 landmarks adn creates a uniuqe face mesh

LEFTEYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHTEYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]


cap = cv.VideoCapture(1)
resolutionWeight = 1280
resolutionHeight = 1024

camWidth = 640
camHeight = 480

mappedWidth= resolutionWeight / camWidth
mappedHeight = resolutionHeight / camHeight




with mp_face_mesh.FaceMesh( max_num_faces = 1, refine_landmarks= True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5 ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            imgH, imgW = frame.shape[:2] # height and width of the frame
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
               mesh_points = np.array([np.multiply([p.x, p.y], [imgW, imgH]).astype(int) for p in results.multi_face_landmarks[0].landmark])
               #print(mesh_points)
              # cv.polylines(frame, [mesh_points[LEFTEYE]], True, (0,255,0), 1 , cv.LINE_AA)
               #cv.polylines(frame, [mesh_points[RIGHTEYE]], True, (0, 255, 0), 1, cv.LINE_AA)

               (left_CX, left_CY), l_Radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
               (right_CX, right_CY), r_Radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
               center_left = np.array([left_CX, left_CY], dtype=np.int32 )
               center_right = np.array([right_CX, right_CY],dtype=np.int32)

               cv.circle(frame, center_right,int(l_Radius), (0,255,0),1,cv.LINE_AA )
               cv.circle(frame, center_left, int(r_Radius), (0,255,0),1,cv.LINE_AA)


               pyag.moveTo(left_CX+imgW,left_CY)



            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
cap.release()
cv.destroyAllWindow()