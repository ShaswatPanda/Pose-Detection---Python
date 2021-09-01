# Libraries
import mediapipe as mp
import cv2
import matplotlib as plt
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from playsound import playsound
import sys
import logging
from logging.config import dictConfig
from matplotlib.backends.backend_pdf import PdfPages



# DEFINITIONS
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_pose = mp.solutions.pose
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# LOG FILE
logging.basicConfig(filename="LogFiles/anglesLog.log", level=logging.INFO)
logging.FileHandler('LogFiles/anglesLog.log', "w")

# FEED CAPTURE
cap = cv2.VideoCapture(0)                           # Webcam feed
#cap = cv2.VideoCapture('Videos/poseTesting.mp4')   # Local video feed

pTime = 0

# VIDEO PROCESSING

# Resolution scaling

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

make_1080p()

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

#make_480p()

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#change_res(540,960)

# Upscaling/Downscaling

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

"""
while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent=75)
    cv2.imshow('frame75', frame75)
    frame150 = rescale_frame(frame, percent=150)
    cv2.imshow('frame150', frame150)

"""

# GRAPH TO PDF

def pdf_save(fig_list, file_name, path_name='/path_name/'):

  pp=PdfPages(path_name+file_name)
  for fig in fig_list:
    pp.savefig(figure=fig)

  pp.close()

  print ('PDF saved to %s' % path_name+file_name)
  return


# Read logo and resize
logo = cv2.imread('Images/manastik_logo.png')
size = 150
logo = cv2.resize(logo, (size, size))

# Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

# GRAPH INIT
# x axis values
x = []
# corresponding y axis values
y1 = []
y2 = []


index = count()
plt.style.use('fivethirtyeight')


# CURL COUNTER AND FLAGS
counter = 0
stage1 = None   # Left arm
stage2 = None   # Right arm

backStage1 = None
backStage2 = None
backStage = "Straight"

audioFlag0 = 1 # Instruction
audioFlag1 = 1 # Back straight
audioFlag4 = 1 # Back crooked
audioFlag2 = 1 # Right action
audioFlag3 = 1 # Wrong leg

# DEFAULT COLOR VALUES
green = (0,255,0)
red = (0,0,255)
white = (255,255,255)
color1 = (0,255,0)
color2 = (0,255,0)
backColor = (255,255,255)



# INITIALIZATION
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Getting feed
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Flip the frame
            #frame = cv2.flip(frame, 1)

            # Region of Image (ROI), where we want to insert logo
           # hwc = frame.shape
            roi = frame[-size-10:-10, -size-10:-10]

            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

            #frame = cv2.flip(frame, 1)

        # Recolor formatting
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect
        results = pose.process(image)
        #hresults = holistic.process(image)
        #print("Face landmarks: \n", results.face_landmarks)
        #print("Pose landmarks: \n", results.pose_landmarks)

        # Recolor format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, channels = image.shape
        #print(width)


        if ret:
            # Flip the frame
            frame = cv2.flip(frame, 1)

            # Region of Image (ROI), where we want to insert logo
           # hwc = frame.shape
            roi = frame[-size-10:-10, -size-10:-10]

            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

       


        # ANGLE CALCULATIONS
        
        # X-Y PLANE
        def calculate_angle(a, b, c):
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
        # Y-Z PLANE
        def calculate_angle_yz(a, b, c):
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[2]-b[2], c[1]-b[1]) - np.arctan2(a[2]-b[2], a[1]-b[1])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
        # X-Z PLANE
        def calculate_angle_xz(a, b, c):
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # End

            radians = np.arctan2(c[2]-b[2], c[0]-b[0]) - np.arctan2(a[2]-b[2], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle>180:
                angle = 360 - angle

            return angle
        
        
        
        

        # FONT SCALING FUNCTION
        def get_optimal_font_scale(text, width):
            for scale in reversed(range(0, 60, 1)):
                textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
                new_width = textSize[0][0]
                #print(new_width)
                if (new_width <= width):
                    return scale/10
            return 1

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            wlandmarks = results.pose_world_landmarks.landmark
            #print(landmarks)

            # JOINTS

            lshoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            lelbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z
            lwrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z
            rshoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            relbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
            rwrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
            lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
            rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
            lknee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
            rknee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
            rankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
            lankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z

            # ANGLE CALCULATIONS

            lElbowAngle = "Left Elbow:" + str(int(calculate_angle(lshoulder, lelbow, lwrist)))
            rElbowAngle = "Right Elbow:" + str(int(calculate_angle(rshoulder, relbow, rwrist)))
            lShoulderAngle = "Left Shoulder:" + str(int(calculate_angle(lhip, lshoulder, lelbow)))
            rShoulderAngle = "Right Shoulder:" + str(int(calculate_angle(rhip, rshoulder, relbow)))
            rKneeAngle = "Right Knee:" + str(180 - int(calculate_angle(rhip, rknee, rankle)))
            lKneeAngle = "Left Knee:" + str(180 - int(calculate_angle(lhip, lknee, lankle)))
            rHipAngle = "Right Hip:" + str(180 - int(calculate_angle(rshoulder, rhip, rknee)))
            lHipAngle = "Left Hip:" + str(180 - int(calculate_angle(lshoulder, lhip, lknee)))
            
            # Z-AXIS ANGLES
            
            lElbowAngleXZ = "XZ: " + str(int(calculate_angle_xz(lshoulder, lelbow, lwrist)))
            rElbowAngleXZ = "XZ: " + str(int(calculate_angle_xz(rshoulder, relbow, rwrist)))
            lShoulderAngleXZ = "XZ:" + str(int(calculate_angle_xz(lhip, lshoulder, lelbow)))
            rShoulderAngleXZ = "XZ:" + str(int(calculate_angle_xz(rhip, rshoulder, relbow)))


            lElbowAngleYZ = "YZ: " + str(int(calculate_angle_yz(lshoulder, lelbow, lwrist)))
            rElbowAngleYZ = "YZ: " + str(int(calculate_angle_yz(rshoulder, relbow, rwrist)))
            lShoulderAngleYZ = "YZ:" + str(int(calculate_angle_yz(lhip, lshoulder, lelbow)))
            rShoulderAngleYZ = "YZ:" + str(int(calculate_angle_yz(rhip, rshoulder, relbow)))
            
            

            # LOGGING

            #"""
            



            logging.info(lElbowAngle)
            logging.info(rElbowAngle)
            logging.info(lShoulderAngle)
            logging.info(rShoulderAngle)
            logging.info(rKneeAngle)
            logging.info(rHipAngle )
            logging.info(lHipAngle)


            #logging.error('your text goes here')
            #logging.debug('your text goes here')
            
            #"""

            # CURL COUNTER
            angle1 = int(calculate_angle(lshoulder, lelbow, lwrist))
            angle2 = int(calculate_angle(rshoulder, relbow, rwrist))

            backAngle1 = int(calculate_angle(lshoulder, lhip, lknee))
            backAngle2 = int(calculate_angle(rshoulder, rhip, rknee))

            if angle1 > 160:
                stage1 = "Stretched"
            if angle1 < 30 and stage1 == 'Stretched':
                stage1 = "Folded"
                counter += 1
                #print(counter)

            if angle2 > 160:
                stage2 = "Stretched"
                color2 = green
            if angle2 < 30 and stage2 == 'Stretched':
                stage2 = "Folded"
                color2 = red
                #counter += 1


            # Back straight condition

            if backAngle1 > 174 and backAngle2 > 174:
                backStage1 = "Straight"
                backStage2 = "Straight"
                backColor = white
                backStage = "Straight"
            if backAngle1 < 174 and backAngle2 < 174 and backStage1 == backStage2 =='Straight':
                backStage1 = "Crooked"
                backStage2 = "Crooked"
                backColor = red
                backStage = "Crooked"
                #print(counter)


            # AUDIO PROMPTS

            # Instruction prompt
            if audioFlag0 == 1:
                playsound('Audio/Correct Arm.mp3')
                audioFlag0 = 0

            # Condition 1: Standing Straight

            if backStage1 == backStage2 == 'Straight':
                if audioFlag1==1:
                    playsound('Audio/Standing Correctly.mp3')
                audioFlag1 = 0
                #audioFlag2 = 1
                #audioFlag3 = 1
                audioFlag4 = 1

            if backStage1 == backStage2 == 'Crooked':
                if audioFlag4==1:
                    playsound('Audio/Stand Straight.mp3')
                audioFlag1 = 1
                audioFlag2 = 1
                #audioFlag3 = 1
                audioFlag4 = 0

            # Condition 2: Left arm movement correct

            if stage1 == 'Folded' and stage2 != 'Folded':
                if audioFlag2==1:
                    playsound('Audio/Correct.mp3')
                audioFlag2 = 0
                audioFlag1 = 0
                audioFlag3 = 1

            # Condition 3: Wrong arm movement

            if stage2 == 'Folded' and stage1 != 'Folded':
                if audioFlag3==1:
                    playsound('Audio/Wrong hand.mp3')
                audioFlag2 = 1
                audioFlag1 = 0
                audioFlag3 = 0

        except:
            pass


        #print(landmarks)
        #for lndmrk in mp_pose.PoseLandmark:
         #   print("")
        #print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        
        # x axis values
        #x = [1,2,3]
        # corresponding y axis values
        #y = [2,4,1]
        
        y1.append(int(calculate_angle(lshoulder, lelbow, lwrist)))
        y2.append(int(calculate_angle(rshoulder, relbow, rwrist)))
        x.append(int(next(index)))
          
        plt.cla()
        
        plt.plot(x, y1, label = 'Left Elbow')
        plt.plot(x, y2, label = 'Right Elbow')
        
        plt.legend()
        
        
        
        """
        # plotting the points 
        
          
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
          
        # giving a title to my graph
        plt.title('Angle Detection - L Elbow')
          
        plt.tight_layout()
        
        # function to show the plot
        plt.show()
        
        #plt.savefig('Graphs/output1.png', facecolor='', bbox_inches="tight",
        #    pad_inches=0.3, transparent=True)
        
        
        #"""

        # LANDMARK DRAWINGS

        """
         # 1. Face landmark drawing
        mp_drawing.draw_landmarks(image, hresults.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2))
        

        # 2. Right Hand
        mp_drawing.draw_landmarks(image, hresults.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, hresults.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        
        """
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(246,66,230), thickness=2, circle_radius=2))




        # CALCULATE FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        #scale = 1 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
        #fontScale = 3*min(width,height)/(25/scale)

        # DISPLAY FPS
        cv2.rectangle(image, (40, 0), (160, 70), (0,0,0), -1)
        cv2.putText(image, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        # DISPLAY REPS AND STAGES
        cv2.rectangle(image, (180, 0), (400, 120), (0,0,0), -1)
        cv2.putText(image, "Reps:", (200,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        cv2.putText(image, str(counter), (200,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        #cv2.putText(image, str(stage1), (420,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        #cv2.putText(image, str(stage2), (600,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv2.putText(image, str(stage1), tuple(np.multiply([lelbow[0], lelbow[1]], [width, height]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 2, color1, 3)
        cv2.putText(image, str(stage2), tuple(np.multiply([relbow[0],relbow[1]], [width, height]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 2, color2, 3)


        # DISPLAY ANGLES

        cv2.putText(image, str(lElbowAngle),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #"""
        cv2.putText(image, str(lElbowAngleXZ),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(lElbowAngleYZ),
                    tuple(np.multiply([lelbow[0], lelbow[1]], [width+50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        #"""
        #cv2.putText(image, str((lelbow[2])),
        #            tuple(np.multiply([lelbow[0], lelbow[1]], [width-50, height-50]).astype(int)),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        
        cv2.putText(image, str(rElbowAngle),
                    tuple(np.multiply([relbow[0],relbow[1]], [width, height+50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        #cv2.putText(image, str((lwrist[2])),
        #            tuple(np.multiply([lwrist[0], lwrist[1]], [width, height]).astype(int)),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        cv2.putText(image, str(lShoulderAngle),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(lShoulderAngleXZ),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width-50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, str(lShoulderAngleYZ),
                    tuple(np.multiply([lshoulder[0], lshoulder[1]], [width+50, height-50]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
        #cv2.putText(image, str(lshoulder[2]),
        #            tuple(np.multiply([lshoulder[0], lshoulder[1]], [width-50, height-50]).astype(int)),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
       
        cv2.putText(image, str(rShoulderAngle),
                    tuple(np.multiply([rshoulder[0], rshoulder[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, str(lHipAngle),
                    tuple(np.multiply([lhip[0], lhip[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, backColor, 2, cv2.LINE_AA)
        
        cv2.putText(image, str(rHipAngle),
                    tuple(np.multiply([rhip[0], rhip[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, backColor, 2, cv2.LINE_AA)

        cv2.putText(image, str(lKneeAngle),
                    tuple(np.multiply([lknee[0], lknee[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, backColor, 2, cv2.LINE_AA)
        
        cv2.putText(image, str(rKneeAngle),
                    tuple(np.multiply([rknee[0], rknee[1]], [width, height]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, backColor, 2, cv2.LINE_AA)


        cv2.putText(image, str(backStage),
                    (420,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, backColor, 2, cv2.LINE_AA)



        """"

        # Left elbow
        cv2.rectangle(image, (180, 0), (810, 70), (0,0,0), -1)
        cv2.putText(image, str(lElbowAngle), (200,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        # Right elbow
        cv2.rectangle(image, (180, 80), (810, 150), (0,0,0), -1)
        cv2.putText(image, str(rElbowAngle), (200,130), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        # Left shoulder
        cv2.rectangle(image, (830, 0), (1550, 70), (0,0,0), -1)
        cv2.putText(image, str(lShoulderAngle), (850,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)
        # Right shoulder
        cv2.rectangle(image, (830, 80), (1550, 150), (0,0,0), -1)
        cv2.putText(image, str(rShoulderAngle), (850,130), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

        """

        # WINDOW NORMALISER
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resize(image, (640, 480))
        cv2.imshow('Image', image)



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


#"""
# plotting the points 
#plt.plot(x, y)
 
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')



# giving a title to my graph
plt.title('Angle Detection - L Elbow')

plt.tight_layout()

plt.savefig('Graphs/output1.png', facecolor='', bbox_inches="tight",
            pad_inches=0.3, transparent=True)


#pdf_save([f1,f2,3], 'test_pdf_save.pdf')

#plt.show()
plt.close()


#"""

#Deinitialize
cap.release()
cv2.destroyAllWindows()


# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:06:27 2021

@author: Shaswat
"""

