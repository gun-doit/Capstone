import cv2
import time
import math #####추가 라이브러리
import numpy as np
import pandas as pd
import mediapipe as mp
import pyrealsense2 as rs
from multiprocessing import Process
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh



## 프레임 크기 ##
WIDTH = 640
HEIGHT = 480

## 눈 가로 ##
E_LEYE_x = 0
F_LEYE_x = 0
E_REYE_x = 0
F_REYE_x = 0
E_LEYE_y = 0
F_LEYE_y = 0
E_REYE_y = 0
F_REYE_y = 0

## 눈 세로 ##
U_REYE_x = 0
D_REYE_x = 0
U_LEYE_x = 0
D_LEYE_x = 0
U_REYE_y = 0
D_REYE_y = 0
U_LEYE_y = 0
D_LEYE_y = 0

## 눈 가로세로 길이 ##
hei_REYE = 0
wid_REYE = 0
hei_LEYE = 0
wid_LEYE = 0


pipeline = rs.pipeline()
config = rs.config()
setWidth = 640
setHeight = 480

inputScale = 1.0/255

config.enable_stream(rs.stream.depth, setWidth, setHeight, rs.format.z16, 30)
config.enable_stream(rs.stream.color, setWidth, setHeight, rs.format.bgr8, 30)
pipeline.start(config)

def Face(image, results):
    global D_REYE_y,U_REYE_y,D_LEYE_y,U_LEYE_y,E_LEYE_x,F_LEYE_x,F_REYE_x,E_REYE_x
    
    ## 얼굴의 좌표를 받아옴 ##
    if(results_face.multi_face_landmarks != None):
        face_landmarks = results_face.multi_face_landmarks[0]  # 첫 번째 얼굴 랜드마크만 사용

        ## 왼쪽눈 눈꼬리 ##
        Leye_end_landmark = face_landmarks.landmark[263]
        E_LEYE_x = int(Leye_end_landmark.x * WIDTH)
        E_LEYE_y = int(Leye_end_landmark.y * HEIGHT)
        ## 왼쪽눈 눈앞머리 ##
        Leye_front_landmark = face_landmarks.landmark[362]
        F_LEYE_x = int(Leye_front_landmark.x * WIDTH)
        F_LEYE_y = int(Leye_front_landmark.y * HEIGHT)
        ## 오른쪽눈 눈꼬리 ##
        Reye_end_landmark = face_landmarks.landmark[33]
        E_REYE_x = int(Reye_end_landmark.x * WIDTH)
        E_REYE_y = int(Reye_end_landmark.y * HEIGHT)
        ## 오른쪽눈 눈앞머리 ##
        Reye_front_landmark = face_landmarks.landmark[133]
        F_REYE_x = int(Reye_front_landmark.x * WIDTH)
        F_REYE_y = int(Reye_front_landmark.y * HEIGHT)
       
        ## 왼쪽눈 눈위 ##
        Leye_up_landmark = face_landmarks.landmark[386]
        U_LEYE_x = int(Leye_up_landmark.x * WIDTH)
        U_LEYE_y = int(Leye_up_landmark.y * HEIGHT)
        ## 왼쪽눈 눈아래 ##
        Leye_down_landmark = face_landmarks.landmark[374]
        D_LEYE_x = int(Leye_down_landmark.x * WIDTH)
        D_LEYE_y = int(Leye_down_landmark.y * HEIGHT)
        ## 오른쪽눈 눈위 ##
        Reye_up_landmark = face_landmarks.landmark[159]
        U_REYE_x = int(Reye_up_landmark.x * WIDTH)
        U_REYE_y = int(Reye_up_landmark.y * HEIGHT)
        ## 오른쪽눈 눈아래 ##
        Reye_down_landmark = face_landmarks.landmark[145]
        D_REYE_x = int(Reye_down_landmark.x * WIDTH)
        D_REYE_y = int(Reye_down_landmark.y * HEIGHT)

## 눈 가로세로 길이 측정 함수 ##
def EYE_LENGTH(D_REYE_y,U_REYE_y,D_LEYE_y,U_LEYE_y,E_LEYE_x,F_LEYE_x,F_REYE_x,E_REYE_x):
  hei_REYE = (D_REYE_y - U_REYE_y)
  hei_LEYE = (D_LEYE_y - U_LEYE_y)
  
  wid_LEYE = (E_LEYE_x - F_LEYE_x)
  wid_REYE = (F_REYE_x - E_REYE_x)
  
  print("leye height :" + hei_LEYE)
  print("reye height :" +hei_REYE)
  print("leye width :" +wid_LEYE)
  print("leye width :" +wid_REYE)
       
#가이드라인 이미지 불러오고 GraySCale, Edge 검출
img = cv2.imread('image/face_guideline_head.png')
edges = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
pipeline = rs.pipeline()
config = rs.config()
setWidth = 640
setHeight = 480

inputScale = 1.0/255
    
config.enable_stream(rs.stream.depth, setWidth, setHeight, rs.format.z16, 30)
config.enable_stream(rs.stream.color, setWidth, setHeight, rs.format.bgr8, 30)
pipeline.start(config)
    
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    i = 0
    
    while cv2.waitKey(1) < 0:
        i+=1
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
           
        if not frame:
            continue
                   
        resize_edges = np.repeat(edges[:,:,np.newaxis],3,-1)
        frame = np.asanyarray(frame.get_data())
        image_height, image_width, _ = frame.shape
            
        # 엣지 추가
        frame = cv2.bitwise_and(frame, resize_edges)
            
        frame.flags.writeable = False
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(frame) 
        results_face = face_mesh.process(frame) 

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        Face(frame, results)
        EYE_LENGTH(D_REYE_y,U_REYE_y,D_LEYE_y,U_LEYE_y,E_LEYE_x,F_LEYE_x,F_REYE_x,E_REYE_x)
        
        if i == 5:
            Face(frame, results)
            i = 0

        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        cv2.imshow('MediaPipe Holistic', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
