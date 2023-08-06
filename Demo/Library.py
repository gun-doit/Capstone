################################ 라이브러리 설정 ################################
#-*- coding: utf-8 -*-

# TKinter
from tkinter import *
from tkinter import ttk
import tkinter as tk

# 이미지
import cv2
from PIL import Image, ImageTk,ImageFont, ImageDraw
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import norm 
from scipy import stats as ss

#비디오
import mediapipe as mp
import pyrealsense2 as rs

# 데이터
from dataclasses import dataclass
from pandas import DataFrame
import numpy as np

# 부가적
import math
import time
from pathlib import Path
from multiprocessing import Process
import threading
import os, sys


################################ 전역 변수 설정 ################################
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINE = cv2.imread(ROOT_DIR + '/image/GuideLine.png')
FACE_GUIDELINE = cv2.imread(ROOT_DIR + '/image/face_guideline_head.png')
IMAGE_DIR = ROOT_DIR + '/assets/images/'
BASE_BG = '#E1E9F6'
WIDTH = 640
HEIGHT = 480
COUNTOUT = 60
N_SECONDS = 2
PI = math.pi

# 리얼센스 카메라 설정 부분
PIPELINE = rs.pipeline()
CONFIG = rs.config()
CONFIG.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)    #컬러
CONFIG.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)     #깊이
ALIGN_TO = rs.stream.depth
ALIGN = rs.align(ALIGN_TO)
lock = threading.Lock()

# 미디어 파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

#컬러
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)

#폰트 로드
FONTPATH = ROOT_DIR + '/assets/fonts/NanumGothicBold.ttf'
FONT = ImageFont.truetype(FONTPATH,20)

# 리스트 
KEY_LIST = [
    ['1','2','3'],
    ['4','5','6'],
    ['7','8','9'],
    ['Back','0','Login']
]
DISEASE_LIST = [
    ['두통', '심장병'],
    ['당뇨', '아토피'],
    ['비만', '치매'],
    ['고혈압', '카페인'],
    ['음주', '흡연'],
]
CHARACTER_LIST = [
    ['토끼', '강아지', '곰', '기린'],
    ['사자', '여우', '쥐', '코끼리']
]

#구조체 및 전역변수 설정
@dataclass
class Str:
    guide:str = " "

class Int:
    guide:int = 0
    
class Pos:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class data:
    gender: str = ''
    phone: str = ''
    height: int = 0
    weight: int = 0
    birth: str = ''

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

## 눈 가로 ##
E_LEYE = Pos()
F_LEYE = Pos()
E_REYE = Pos()
F_REYE = Pos()

## 눈 세로 ##
U_REYE = Pos()
D_REYE = Pos()
U_LEYE = Pos()
D_LEYE = Pos()

## 눈 가로세로 길이 ##
HEI_REYE = Int()
WID_REYE = Int()
HEI_LEYE = Int()
WID_LEYE = Int()

HEI_REYE_TEXT = Str()
WID_REYE_TEXT = Str()
HEI_LEYE_TEXT = Str()
WID_LEYE_TEXT = Str()

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

L_KNEE_DEG = Int()
R_KNEE_DEG = Int()


FC_LEFT_END = Pos()
FC_RIGHT_END = Pos()

RGB_COLOR1 = Str()
RGB_COLOR2 = Str

R_KNEE_LIST=[]
L_KNEE_LIST=[]

DATA = [('Face_lr', F_SCORE_LR_I.guide),('Face_center', F_SCORE_CENTER_I.guide), ('Shoulder', S_SCORE_I.guide), ('test1', 67), ('test2', 22)]

FIRST_T_L = Pos()
CUT_OFF_T_L = Pos()
FIRST_T_R = Pos()
CUT_OFF_T_R = Pos()

LEFT_EYE_AREA = Str()
RIGHT_EYE_AREA = Str()

