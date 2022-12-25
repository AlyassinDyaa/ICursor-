import cv2 as cv2
import numpy as np
import mediapipe as mp
import pyautogui
import pyautogui as pyag
from pynput.mouse import Button, Controller
import dlib
import imutils
from turtle import delay
from imutils import face_utils
import autopy
from utils import *
import autopy


# Thresholds and consecutive frame length for triggering the mouse action.
MAT = 0.6  # used for mouth threshold
MACF = 15
EyeThresh = 0.19
EACF = 15
wink_thresh = 0.04
winkCThresh = 0.19
wf = 5

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
mc = 0  # MOUTH_COUNTER
ec = 0  # EYE_COUNTER
wc = 0  # WINK_COUNTER

im = False  # INPUT_MODE
eyeClicking = False  # EYE_CLICK
leftWink = False  # LEFT_WINK
rightWink = False  # RIGHT_WINK
sm = False  # SCROLL_MODE
anchor = (0, 0)  # ANCHOR_POINT

# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()  # Face detector in the DLIB library that detects the face
predictor = dlib.shape_predictor(shape_predictor)  # adds important facial landmarks on the face

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(noseStartPosition, noseEndPosition) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mouthStartPosition, mouthEndPosition) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

mp_face_mesh = mp.solutions.face_mesh #  gtetting face mesh example that contains the 478 landmarks adn creates a uniuqe face mesh

LEFTEYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHTEYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]


cap = cv2.VideoCapture(1)
resolutionWeight = 1280
resolutionHeight = 1024

camWidth = 640
camHeight = 480

mappedWidth= resolutionWeight / camWidth
mappedHeight = resolutionHeight / camHeight




with mp_face_mesh.FaceMesh( max_num_faces = 1, refine_landmarks= True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5 ) as face_mesh:
    while True:
        # Grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=camWidth, height=camHeight)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # print("1")
        # **********************************************************************************
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgH, imgW = frame.shape[:2]  # height and width of the frame
        results = face_mesh.process(rgb)  # gets the output of the landmarks
        if results.multi_face_landmarks:  # check if landmarks desired are available or not
            # print( results.multi_face_landmarks[0].landmark) # prints all the landmarks
            mesh_points = np.array([np.multiply([p.x, p.y], [imgW, imgH]).astype(int) for p in
                                    results.multi_face_landmarks[
                                        0].landmark])  # multiply x and w by w and h to get the pixel coordinate of each landmark. After multplying the values we get a floadting point but for accuary we convert it to integre
            # cv.polylines(frame, [mesh_points[LEFTEYE]], q, (0,255,0), 1 , cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHTEYE]], True, (0, 255, 0), 1, cv.LINE_AA)

            (left_CX, left_CY), l_Radius = cv2.minEnclosingCircle(mesh_points[
                                                                      LEFT_IRIS])  # function in openCV that returns the center points and radius so that we can draw a circle around it
            (right_CX, right_CY), r_Radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([left_CX, left_CY], dtype=np.int32)
            center_right = np.array([right_CX, right_CY], dtype=np.int32)

            cv2.circle(frame, center_right, int(l_Radius), (0, 0, 255), 1, cv2.LINE_AA)  # drawing circle on right eye
            cv2.circle(frame, center_left, int(r_Radius), (0, 0, 255), 1, cv2.LINE_AA)  # drawing circle on left eye


        # **********************************************************************************

        # print("2")
        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        if len(rects) > 0:
            rect = rects[0]
        else:
            # cv2.imshow("Frame", frame)
            # print("3")
            key = cv2.waitKey(1) & 0xFF
            continue

        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        mouth = shape[mouthStartPosition: mouthEndPosition]
        leftEye = shape[leftEyeStart: leftEyeEnd]
        rightEye = shape[rightEyeStart: rightEyeEnd]
        nose = shape[noseStartPosition: noseEndPosition]

        # Because I flipped the frame, left is right, right is left. SWAP
        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        mouseController = Controller()
        # print("4")
        # Average the mouth aspect ratio together for both eyes
        mouthAspectRatio = mouth_aspect_ratio(mouth)  # mouth open or closed
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)  # left eye open or closed
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)  # right eye open or closed
        ear = (leftEyeAspectRatio + rightEyeAspectRatio) / 2.0

        bothEyesAspectRatio = np.abs(leftEyeAspectRatio - rightEyeAspectRatio)  # both eyes aspect ratio
        # print("5")
        nose_point = (nose[3, 0], nose[3, 1])

        # Compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        mHullValues = cv2.convexHull(mouth)
        lEHullValues = cv2.convexHull(leftEye)
        rEHullValues = cv2.convexHull(rightEye)

        #cv2.drawContours(frame, [mHullValues], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [lEHullValues], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rEHullValues], -1, (0, 255, 0), 1)

        # for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        # cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
        # Check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter

        # ********************** EYES CLICKING ********************************
        if bothEyesAspectRatio > wink_thresh:

            if leftEyeAspectRatio < rightEyeAspectRatio:
                if leftEyeAspectRatio < EyeThresh:

                    wc += 1  # wink counter

                    if wc > wf:
                        print("Left Click")
                        mouseController.click(Button.left, 2)
                        wc = 0  # wink counter


            elif leftEyeAspectRatio > rightEyeAspectRatio:

                if rightEyeAspectRatio < EyeThresh:

                    wc += 1
                    if wc > wf:
                        print("Right Click")
                        mouseController.click(Button.right, 1)

                        wc = 0  # wink counter
            else:
                wc = 0  # wink counter


        # **********************************************************

        # ********************** MOUTH ******************************

        if mouthAspectRatio > MAT:
            mc += 1

            if mc >= MACF:
                # if the alarm is not on, turn it on
                im = not im
                # SCROLL_MODE = not SCROLL_MODE
                mc = 0
                anchor = nose_point

        else:
            mc = 0

        # **********************************************************

        if im:

            cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            x, y = anchor
            nx, ny = nose_point
            w, h = 60, 35
            multiple = 1

            dir = direction(nose_point, anchor, w, h)
            cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            drag = 18
            if dir == 'right':
                pyag.moveRel(drag, 0)
            elif dir == 'left':
                pyag.moveRel(-drag, 0)
            elif dir == 'up':
                if sm:
                    pyag.scroll(40)
                else:
                    pyag.moveRel(0, -drag)
            elif dir == 'down':
                if sm:
                    pyag.scroll(-40)
                else:
                    pyag.moveRel(0, drag)

        if sm:
            cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `Esc` key was pressed, break from the loop
        if key == 27:
            break

    # Do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()
