################################ 라이브러리 설정 ################################
#-*- coding: utf-8 -*-
import cv2
import tkinter
from PIL import Image, ImageTk
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import math
import time
import mediapipe as mp
import pyrealsense2 as rs
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Process
import threading
import os
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

## 프레임 크기 ##
WIDTH = 640
HEIGHT = 480

L_ANKLE_x = 0
L_ANKLE_y = 0
R_ANKLE_x = 0
R_ANKLE_y = 0

L_KNEE_x = 0
L_KNEE_y = 0
R_KNEE_x = 0
R_KNEE_y = 0

L_HIP_x = 0
L_HIP_y = 0
R_HIP_x = 0
R_HIP_y = 0

img = cv2.imread('image/Guideline.png')
edges = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

pipeline = rs.pipeline()
config = rs.config()
setWidth = 640
setHeight = 480

inputScale = 1.0/255

config.enable_stream(rs.stream.depth, setWidth, setHeight, rs.format.z16, 30)
config.enable_stream(rs.stream.color, setWidth, setHeight, rs.format.bgr8, 30)
pipeline.start(config)


def LEG(image, Landmarks):
  global L_ANKLE_x,L_ANKLE_y,R_ANKLE_x,R_ANKLE_y,L_HIP_x,L_HIP_y,R_HIP_x,R_HIP_y,L_KNEE_x,L_KNEE_y,R_KNEE_x,R_KNEE_y
  
  if(results.pose_landmarks != None):
    L_ANKLE_x, L_ANKLE_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * HEIGHT
    R_ANKLE_x, R_ANKLE_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * HEIGHT
    L_KNEEx, L_KNEE_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * HEIGHT
    R_KNEE_x, R_KNEE_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * HEIGHT
    L_HIP_x, L_HIP_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * HEIGHT
    R_HIP_x, R_HIP_y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * HEIGHT


def Leg_Angle(lax,lay,rax,ray,lhx,lhy,rhx,rhy,lkx,lky,rkx,rky):
  
  la = np.array([lax,lay])
  ra = np.array([rax,ray])
  lh = np.array([lhx,lhy])
  rh = np.array([rhx,rhy])
  lk = np.array([lkx,lky])
  rk = np.array([rkx,rky])
  
  R_thigh = rk - rh # 오른쪽 허벅지
  L_thigh = lk - lh # 왼쪽 허벅지
  R_calf = ra - rk # 오른쪽 종아리
  L_calf = la - lk # 왼쪽 종아리
  
  norm_R_thigh = np.linalg.norm(R_thigh)
  norm_L_thigh = np.linalg.norm(L_thigh)
  norm_R_calf = np.linalg.norm(R_calf)
  norm_L_calf = np.linalg.norm(L_calf)
  
  dot_R_KNEE = np.dot(R_thigh,R_calf)
  dot_L_KNEE = np.dot(L_thigh,L_calf)

  R_KNEE_cos_th = dot_R_KNEE / (norm_R_thigh * norm_R_calf)
  L_KNEE_cos_th = dot_L_KNEE / (norm_L_thigh * norm_L_calf)
  
  R_KNEE_rad = math.acos(R_KNEE_cos_th)
  R_KNEE_deg = math.degrees(R_KNEE_rad)
  L_KNEE_rad = math.acos(L_KNEE_cos_th)
  L_KNEE_deg = math.degrees(L_KNEE_rad)
  
  print("left knee degree : " + str(L_KNEE_deg))
  print("right knee degree : " + str(R_KNEE_deg))


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
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
        
        results = pose.process(frame) 
        results_face = pose.process(frame) 

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if i == 5:
          LEG(frame, results)
          i = 0
        
        LEG(frame,results)
        Leg_Angle(L_ANKLE_x,L_ANKLE_y,R_ANKLE_x,R_ANKLE_y,L_HIP_x,L_HIP_y,R_HIP_x,R_HIP_y,L_KNEE_x,L_KNEE_y,R_KNEE_x,R_KNEE_y)
        
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
          frame,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())         

        cv2.imshow('MediaPipe knee angle', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break