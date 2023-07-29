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
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# 생성한 라이브러리
# from Library import *
# from guideline import *
# from save_media import *


################################ 전역 변수 설정 ################################
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINE = cv2.imread(ROOT_DIR + '/image/GuideLine.png')
FACE_GUIDELINE = cv2.imread(ROOT_DIR + '/image/face_guideline_head.png')
WIDTH = 640
HEIGHT = 480
COUNTOUT = 60
N_SECONDS = 2
PI = math.pi

#카메라 설정 부분
PIPELINE = rs.pipeline()
CONFIG = rs.config()
CONFIG.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)    #컬러
CONFIG.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)     #깊이
ALIGN_TO = rs.stream.depth
ALIGN = rs.align(ALIGN_TO)
lock = threading.Lock()

#컬러
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)

#폰트 로드
FONTPATH = ROOT_DIR + '/NanumGothicBold.ttf'
FONT = ImageFont.truetype(FONTPATH,20)

@dataclass
class Str:
    guide:str = " "

class Int:
    guide:int = 0
    
class Pos:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

L_SHOULDER = Pos()
R_SHOULDER = Pos()
L_EAR = Pos()
R_EAR = Pos()
NOSE = Pos()
Y_DIS = Pos()

CHIN = Pos()
FORHEAD = Pos()
LEYE_END = Pos()
LEYE_FRONT = Pos()
REYE_END = Pos()
REYE_FRONT = Pos()
LLIP = Pos()
RLIP = Pos()
UPPERLIP = Pos()
NOSE_TIP = Pos()
GLABELLA = Pos()
MIDDLE = Pos()

SUM_LS = Pos()
SUM_RS = Pos()

EYE_LIP_DEG = Int()
EYE_DEG = Int()
LIP_DEG = Int()
FACE_DEG = Int()

CHIN_DIS = Int()
    
STR = Str()
R_TEXT = Str()
R_TY_TEXT = Str()
F_TEXT = Str()
FA_TEXT = Str()
FC_LR_TEXT = Str()
FC_CENTER_TEXT = Str()

DISTANCE = Int()
MIDDLE_LR_S = Pos()
MIDDLE_LR_F = Pos()
MIDDLE_PEL = Pos()
MIDDLE_KNEE = Pos()

S_SCORE = Str()
F_SCORE_CENTER = Str()
F_SCORE_LR = Str()

S_SCORE_I = Int()
F_SCORE_CENTER_I = Int()
F_SCORE_LR_I = Int()

L_PEL = Pos()
R_PEL = Pos()
L_KNEE = Pos()
R_KNEE = Pos()
L_FOOT = Pos()
R_FOOT = Pos()
PEL_DIS = Pos()
P_SCORE = Str()
P_SCORE_I = Str()

SUM_LPEL = Pos()
SUM_RPEL = Pos()

FC_LEFT_END = Pos()
FC_RIGHT_END = Pos()

L_KNEE_DEG = Int()
R_KNEE_DEG = Int()

RGB_COLOR = Str()

R_KNEE_LIST=[]
L_KNEE_LIST=[]

DATA = [('Face_lr', F_SCORE_LR_I.guide),('Face_center', F_SCORE_CENTER_I.guide), ('Shoulder', S_SCORE_I.guide), ('test1', 67), ('test2', 22)]

FIRST_T = Pos()
CUT_OFF_T = Pos()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
