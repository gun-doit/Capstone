from Library import *

#* 파라미터 : 미디어파이프의 랜드마크, 깊이정보 프레임
#* 반환값 : 문자열
#* 가이드라인의 좌표를 지정:
#* 어깨 : x 148,496 | y 256,480
#* 귀 : x 256,374 | y 109,183


def GuideText(frame):
    #텍스트 위치 계산
    text_x, text_y = int((HEIGHT - len(STR.guide)*7) / 2), 40 #가로 중앙으로 설정

    frame = Image.fromarray(frame)
    ImageDraw.Draw(frame).text((text_x,text_y), STR.guide, font=FONT, fill=GREEN)

    frame = np.array(frame)
    return frame
##############어깨##############

#어깨 깊이
def Shoulder_Depth():
    #두 어깨의 깊이 차이가 0.1 이상일 때 => 카메라와 평행으로 서주세요
    if(abs(L_SHOULDER.z - R_SHOULDER.z) > 0.07):
        STR.guide = '카메라와 평행으로 서주세요'
        
    #*아닐 떄 출력
    #else:
        
    #코의 좌표와 어깨 중앙의 좌표를 기준으로 
    Middle_dist = (L_SHOULDER.z + R_SHOULDER.z) / 2
    if(abs(NOSE.z - Middle_dist) > 0.07):
        STR.guide = '허리를 펴주세요'
    #else:
       

#어깨 가이드라인
def INguideline(Depth,Landmarks):
    try :
        #어깨 좌표가 이미지 안에 들어와있는지 확인 및 좌표 가져오기
        L_SHOULDER.x, L_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * HEIGHT
        R_SHOULDER.x, R_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * HEIGHT
        L_EAR.x, L_EAR.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * HEIGHT
        R_EAR.x, R_EAR.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * HEIGHT
        NOSE.x, NOSE.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * HEIGHT
        
        if 0 <= int(L_SHOULDER.x) < WIDTH and 0 <= int(L_SHOULDER.y) < HEIGHT:
            if 0 <= int(R_SHOULDER.x) < WIDTH and 0 <= int(R_SHOULDER.y) < HEIGHT:
                if 0 <= int(NOSE.x) < WIDTH and 0 <= int(NOSE.y) < HEIGHT:
                    L_SHOULDER.z = Depth.get_distance(int(L_SHOULDER.x), int(L_SHOULDER.y))
                    R_SHOULDER.z = Depth.get_distance(int(R_SHOULDER.x), int(R_SHOULDER.y))
                    NOSE.z = Depth.get_distance(int(NOSE.x), int(NOSE.y))

        
        #가이드라인 안에 있는지 확인
        if(320 < L_SHOULDER.x < 501 and 225 < L_SHOULDER.y < 480 and 137 < R_SHOULDER.x < 320 and 225 < R_SHOULDER.y < 480):
            if(254 < L_EAR.x < 387 and 82 < L_EAR.y < 225 and 254 < R_EAR.x < 387 and 82 < R_EAR.y < 225): #가이드라인 안에서
                #측정 시작부분
                return True
        
        else:
            #가이드라인 안에 들어와주세요
            STR.guide = "가이드라인 안에 들어와주세요"
        
        Shoulder_Depth()
        
    #좌표를 가져오지 못함
    except:
        #카메라 안에 들어와주세요
        STR.guide = '카메라안으로 들어와주세요'
    
    return False


##############얼굴###############

#얼굴 어깨 수직선
def INFace_shoulder(Depth,Landmarks):
    try :
        #어깨 좌표가 이미지 안에 들어와있는지 확인 및 좌표 가져오기
        L_SHOULDER.x, L_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * HEIGHT
        R_SHOULDER.x, R_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * HEIGHT

        MIDDLE.x = (L_SHOULDER.x - R_SHOULDER.x)/2 + R_SHOULDER.x
        MIDDLE.y = (L_SHOULDER.y - R_SHOULDER.y)/2 + R_SHOULDER.y
    
    except:
        #카메라 안에 들어와주세요
        STR.guide = '카메라안으로 들어와주세요'
    
    return False

#얼굴 가이드라인
def INFace(Depth,Landmarks):
    try :
        face_landmarks = Landmarks.multi_face_landmarks[0] 

        ## 턱 끝 중앙 ##
        chin_landmark = face_landmarks.landmark[152]
        CHIN.x = int(chin_landmark.x * WIDTH)
        CHIN.y = int(chin_landmark.y * HEIGHT)
        ## 이마 끝 ##
        forhead_landmark = face_landmarks.landmark[10]
        FORHEAD.x = int(forhead_landmark.x * WIDTH)
        FORHEAD.y = int(forhead_landmark.y * HEIGHT)
        
        ## 왼쪽눈 눈꼬리 ##
        Leye_end_landmark = face_landmarks.landmark[263]
        LEYE_END.x = int(Leye_end_landmark.x * WIDTH)
        LEYE_END.y = int(Leye_end_landmark.y * HEIGHT)
        ## 왼쪽눈 눈앞머리 ##
        Leye_front_landmark = face_landmarks.landmark[362]
        LEYE_FRONT.x = int(Leye_front_landmark.x * WIDTH)
        LEYE_FRONT.y = int(Leye_front_landmark.y * HEIGHT)
        ## 오른쪽눈 눈꼬리 ##
        Reye_end_landmark = face_landmarks.landmark[33]
        REYE_END.x = int(Reye_end_landmark.x * WIDTH)
        REYE_END.y = int(Reye_end_landmark.y * HEIGHT)
        ## 오른쪽눈 눈앞머리 ##
        Reye_front_landmark = face_landmarks.landmark[133]
        REYE_FRONT.x = int(Reye_front_landmark.x * WIDTH)
        REYE_FRONT.y = int(Reye_front_landmark.y * HEIGHT)
        
        ## 왼쪽 입꼬리 ##
        Llip_landmark = face_landmarks.landmark[308]
        LLIP.x = (Llip_landmark.x * WIDTH)
        LLIP.y = (Llip_landmark.y * HEIGHT)
        ## 오른쪽 입꼬리 ##
        Rlip_landmark = face_landmarks.landmark[78]
        RLIP.x = (Rlip_landmark.x * WIDTH)
        RLIP.y = (Rlip_landmark.y * HEIGHT)
        ## 윗입술 중앙 ##
        upper_lip_landmark = face_landmarks.landmark[0]
        UPPERLIP.x = (upper_lip_landmark.x * WIDTH)
        UPPERLIP.y = (upper_lip_landmark.y * HEIGHT)
        
        ## 코끝 ##
        nose_tip_landmark = face_landmarks.landmark[1]
        NOSE_TIP.x = (nose_tip_landmark.x * WIDTH)
        NOSE_TIP.y = (nose_tip_landmark.y * HEIGHT)
        ## 미간 ##
        glabella_landmark = face_landmarks.landmark[168]
        GLABELLA.x = (glabella_landmark.x * WIDTH)
        GLABELLA.y = (glabella_landmark.y * HEIGHT)
        
        if(280 < CHIN.x < 340 and  250 < CHIN.y < 310 and 300 < FORHEAD.x < 400 and 53 < FORHEAD.y < 133):
            if(300 < GLABELLA.x < 330 and 130 < GLABELLA.y < 160):
                #측정 시작부분
                return True
            else:            
                STR.guide = "미간 점에 맞춰주세요"
        else:
            #가이드라인 안에 들어와주세요
            STR.guide = "가이드라인 안에 들어와주세요"
    
    #좌표를 가져오지 못함
    except:
        #카메라 안에 들어와주세요
        STR.guide = '카메라안으로 들어와주세요'
    
    return False
