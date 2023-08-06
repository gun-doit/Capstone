from Library import *


#############어깨 측정###############

def Shoulder_incline(lx, ly, rx, ry):
    #깊이
    Y_DIS.x = (870 / ((1/6) * MIDDLE_LR_S.z*1000))  * (ry-ly)
    
    # 0 : 평행
    # -1 : 오류
    # 2 : 왼쪽이 높음
    # 3 : 오른쪽이 높음
    if 0 <= (Y_DIS.x) <= 10: 
        R_TEXT.guide = '균형'
    elif lx - rx == 0:
        R_TEXT.guide = '인식불가'
    else:
        if (ry-ly) / (rx-lx) > 0:
            if abs(Y_DIS.x) > 20:
                R_TEXT.guide = '왼쪽어깨'
                R_TY_TEXT.guide = '고도 비대칭'
            elif abs(Y_DIS.x) > 15:
                R_TEXT.guide = '왼쪽어깨'
                R_TY_TEXT.guide = '중도 비대칭'
            else:
                R_TEXT.guide = '왼쪽어깨'
                R_TY_TEXT.guide = '경도 비대칭'
            
        else:
            if abs(Y_DIS.x) > 20:
                R_TEXT.guide = '오른쪽어깨'
                R_TY_TEXT.guide = '고도 비대칭' 
            elif abs(Y_DIS.x) > 15:
                R_TEXT.guide = '오른쪽어깨'
                R_TY_TEXT.guide = '중도 비대칭' 
            else:
                R_TEXT.guide = '오른쪽어깨'
                R_TY_TEXT.guide = '경도 비대칭'

#어깨 스코어링
def Shoulder_score():
    if abs(Y_DIS.x) < 1:
        S_SCORE.guide, S_SCORE_I.guide = 100, 100
    elif abs(Y_DIS.x) < 2:
        S_SCORE.guide, S_SCORE_I.guide = 99, 99
    elif abs(Y_DIS.x) <= 10:
        S_SCORE.guide, S_SCORE_I.guide = (100 - int(abs(Y_DIS.x) * 1)), (100 - int(abs(Y_DIS.x) * 1))
    elif 10 < abs(Y_DIS.x) < 100:
        S_SCORE.guide, S_SCORE_I.guide = (110 - int(abs(Y_DIS.x) * 2)), (110 - int(abs(Y_DIS.x) * 2))
    else:
        S_SCORE.guide = "잘못 측정"

#어깨 결과 측정
def Shoulder(Landmarks):
    L_SHOULDER.x, L_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * HEIGHT
    R_SHOULDER.x, R_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * HEIGHT
    
    SUM_LS.x += L_SHOULDER.x
    SUM_LS.y += L_SHOULDER.y
    SUM_RS.x += R_SHOULDER.x
    SUM_RS.y += R_SHOULDER.y
    
def Sum_Sholder(i):
    SUM_LS.x = SUM_LS.x / float(i-1)
    SUM_LS.y = SUM_LS.y / float(i-1)
    SUM_RS.x = SUM_RS.x / float(i-1)
    SUM_RS.y = SUM_RS.y / float(i-1) 
                
    Shoulder_incline(SUM_LS.x, SUM_LS.y, SUM_RS.x, SUM_RS.y)
                
def Video_result():   
    # 동영상 파일 경로
    video_path = ROOT_DIR + '/image/color_output.mp4'

    # 동영상 파일 로드
    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        i = 0
        while cap.isOpened():
            i+=1
            success, frame = cap.read()
            try:
                if not success or frame.shape is not None:
            
                    HEIGHT, WIDTH, _ = frame.shape

                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    MP_landmark = pose.process(frame)
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    Shoulder(MP_landmark)
                        
                    mp_drawing.draw_landmarks(
                        frame,
                        MP_landmark.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    cv2.imshow('Output', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                Sum_Sholder(i)
                Shoulder_score()
                
                cv2.destroyAllWindows()
                break

        cap.release()
  
        
###############얼굴 측정################

def F_Shoulder(Landmarks):
    L_SHOULDER.x, L_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * HEIGHT
    R_SHOULDER.x, R_SHOULDER.y = Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * WIDTH, Landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * HEIGHT
    
    MIDDLE.x = (L_SHOULDER.x - R_SHOULDER.x)/2 + R_SHOULDER.x
    MIDDLE.y = (L_SHOULDER.y - R_SHOULDER.y)/2 + R_SHOULDER.y

#눈,입과 수직선간의 각도 / 눈과입의 각도 / 얼굴 중심선과 수직선의 각도 측정
def Eye_lip_angle(lex,ley,rex,rey,llx,lly,rlx,rly):

    ll = np.array([llx,lly])
    rl = np.array([rlx,rly])
    le = np.array([lex,ley])
    re = np.array([rex,rey])
    
    l1 = np.array([MIDDLE.x,MIDDLE.y])
    l2 = np.array([MIDDLE.x,0])
    
    eye = le - re
    lip = ll - rl
    line = l1 - l2
    
    norm_eye = np.linalg.norm(eye)
    norm_lip = np.linalg.norm(lip)
    norm_line = np.linalg.norm(line)
    
    dot_eye_line = np.dot(eye,line)
    dot_lip_line = np.dot(lip,line)
    dot_eye_lip = np.dot(eye,lip)
    
    eye_cos_th = dot_eye_line / (norm_eye * norm_line)
    lip_cos_th = dot_lip_line / (norm_lip * norm_line)
    eye_lip_cos_th = dot_eye_lip / (norm_eye * norm_lip)
   
    eye_rad = math.acos(eye_cos_th)
    eye_deg = math.degrees(eye_rad)
    lip_rad = math.acos(lip_cos_th)
    lip_deg = math.degrees(lip_rad)
    eye_lip_rad = math.acos(eye_lip_cos_th)
    eye_lip_deg = math.degrees(eye_lip_rad)
 
    EYE_LIP_DEG.guide += eye_lip_deg
    EYE_DEG.guide += eye_deg
    LIP_DEG.guide += lip_deg

#얼굴 미간-윗입술과 수직선간의 각도 측정
def Face_angle(glx,gly,ulx,uly):

    gla = np.array([glx,gly]) 
    ulp = np.array([ulx,uly])
    l1 = np.array([MIDDLE.x,MIDDLE.y])
    l2 = np.array([MIDDLE.x,0])
    
    face = ulp - gla
    line = l1 - l2
    
    norm_face = np.linalg.norm(face)
    norm_line = np.linalg.norm(line)
    
    dot_face = np.dot(face,line)
    face_cos_th = dot_face / (norm_face * norm_line)
        
    face_rad = math.acos(face_cos_th)
    face_deg = math.degrees(face_rad)
    
    FACE_DEG.guide += face_deg
    
#코끝, 턱끝, 미간이 수직선과 얼마나 떨어져있는지 측정
def Nose_chin_glabelly():
    
    """chin_area = abs((MIDDLE.x - CHIN.x) * (0 - CHIN.y) - (MIDDLE.y - CHIN.y) * (MIDDLE.x - CHIN.x))
    line = ((MIDDLE.x - MIDDLE.x) **2 + (MIDDLE.y - 0) **2) **0.5
    CHIN_DIS.guide += chin_area/line"""
    CHIN_DIS.guide += (abs(MIDDLE.x - CHIN.x))
def Sum_Face(i):
    
    EYE_LIP_DEG.guide = EYE_LIP_DEG.guide / (i-1)
    EYE_DEG.guide = EYE_DEG.guide / (i-1)
    LIP_DEG.guide = LIP_DEG.guide / (i-1)
    FACE_DEG.guide = FACE_DEG.guide / (i-1)
    CHIN_DIS.guide = CHIN_DIS.guide / (i-1)

    right_left = Eye_lip_incline(EYE_LIP_DEG.guide, EYE_DEG.guide, LIP_DEG.guide)
    face = Face_incline(FACE_DEG.guide) 
    chin = Chin_measurement(CHIN_DIS.guide)
       
    FACE_TYPE(right_left, face, chin)
 
#Eye_lip_angle로 좌우 안면 비대칭 측정 함수
def Eye_lip_incline(eye_lip_results, eye_deg, lip_deg):
    
    if eye_lip_results < 1:
        right_left = '대칭'
    elif eye_lip_results < 2:
        right_left = '경도'
    elif eye_lip_results < 3:
        right_left = '중등도'
    else:
        right_left = '고도'
               
    return right_left
       
#Face_angle로 중앙 안면 비대칭 측정 함수
def Face_incline(face_results):
    
    if face_results < 1 :
        face = '대칭'
    elif face_results < 4:
        face = '경도'
    elif face_results < 8 :
        face = '중등도'
    else :
        face = '고도'
        
    return face

def Chin_measurement(chin_distance):

    chin_distance = (870 / ((1/6) * MIDDLE_LR_F.z*1000))  * chin_distance
    
    if chin_distance < 3:
        return 0 #근육형 or 대칭으로 분류
    else:
        return 1 #턱관절 비대칭으로 분류

    
def FACE_TYPE(right_left, face, chin):#최종 타입 정리
    
    if chin == 0:
        if (LIP_DEG.guide < 90 and LIP_DEG.guide >= 91):
            F_TEXT.guide = '근육형 불균형'
            FA_TEXT.guide = '입꼬리가 수직선으로부터 {}도 가량 틀어짐'.format(round(LIP_DEG.guide,2))
        else:
            F_TEXT.guide = '대칭'
    else:
        if (90 <= EYE_DEG.guide < 91):
            if LIP_DEG.guide < 90 :
                F_TEXT.guide = '하악 좌우변위 우측형'
                FA_TEXT.guide = '입꼬리가 수직선으로부터 {}도 가량 틀어짐'.format(round(LIP_DEG.guide,2))
            elif LIP_DEG.guide >= 91 :
                F_TEXT.guide = '하악 좌우변위 좌측형'
                FA_TEXT.guide = '입꼬리가 수직선으로부터 {}도 가량 틀어짐'.format(round(LIP_DEG.guide,2))
        else:
            if (EYE_DEG.guide < 90):
                if LIP_DEG.guide < 90 :
                    F_TEXT.guide = '접형골 우측형'
                    FA_TEXT.guide = '눈과 입의 각도가 {}도 틀어짐'.format(round(EYE_LIP_DEG.guide,2))
                    FC_LR_TEXT.guide = '{}'.format(right_left)
                    FC_CENTER_TEXT.guide = '{}'.format(face)
                elif LIP_DEG.guide >= 91 :
                    F_TEXT.guide = '측두골 좌측형'
                    FA_TEXT.guide = '눈과 입의 각도가 {}도 틀어짐'.format(round(EYE_LIP_DEG.guide,2))
                    FC_LR_TEXT.guide = '{}'.format(right_left)
                    FC_CENTER_TEXT.guide = '{}'.format(face)
            if(EYE_DEG.guide >= 91):
                if LIP_DEG.guide < 90 :
                    F_TEXT.guide = '측두골 우측형'
                    FA_TEXT.guide = '눈과 입의 각도가 {}도 틀어짐'.format(round(EYE_LIP_DEG.guide,2))
                    FC_LR_TEXT.guide = '{}'.format(right_left)
                    FC_CENTER_TEXT.guide = '{}'.format(face)
                elif LIP_DEG.guide >= 91:
                    F_TEXT.guide = '접형골 좌측형'
                    FA_TEXT.guide = '눈과 입의 각도가 {}도 틀어짐'.format(round(EYE_LIP_DEG.guide,2))
                    FC_LR_TEXT.guide = '{}'.format(right_left)
                    FC_CENTER_TEXT.guide = '{}'.format(face)

#중앙안면
def F_CENTER_SCORE(face_results):
    if face_results < 0.1:
        F_SCORE_CENTER.guide, F_SCORE_CENTER_I.guide = 100, 100
    elif face_results < 0.2:
        F_SCORE_CENTER.guide, F_SCORE_CENTER_I.guide = 99, 99
    elif face_results <= 1.0:
        F_SCORE_CENTER.guide, F_SCORE_CENTER_I.guide = (100 - int(face_results * 10)), (100 - int(face_results * 10))
    elif 1.0 < face_results < 100:
        F_SCORE_CENTER.guide, F_SCORE_CENTER_I.guide = (int(90 - ((face_results-1) // 0.3))), (int(90 - ((face_results-1) // 0.3)))
    else:
        F_SCORE_CENTER.guide = "잘못 측정"

#좌우안면
def F_LR_SCORE():
    if (round(EYE_LIP_DEG.guide,2)) < 0.1:
        F_SCORE_LR.guide, F_SCORE_LR_I.guide = 100, 100
    elif (round(EYE_LIP_DEG.guide,2)) < 0.2:
        F_SCORE_LR.guide, F_SCORE_LR_I.guide = 99, 99
    elif (round(EYE_LIP_DEG.guide,2)) < 100:
        F_SCORE_LR.guide, F_SCORE_LR_I.guide = (100 - int((round(EYE_LIP_DEG.guide,2)) * 10)), (100 - int((round(EYE_LIP_DEG.guide,2)) * 10))
    else:
        F_SCORE_LR.guide = "잘못 측정"
 
#얼굴 안내선
def Face_line(Landmarks, frame):
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
    #얼굴 중심 선
    frame = cv2.line(frame,(int(UPPERLIP.x),int(UPPERLIP.y)),(int(GLABELLA.x),int(GLABELLA.y)),(0,255,0),1)
    #수직선
    frame = cv2.line(frame,(int(MIDDLE.x),int(MIDDLE.y)),(int(MIDDLE.x),0),(255,255,255),1)
    #눈 직선긋기
    frame = cv2.line(frame,(int(LEYE_END.x),int(LEYE_END.y)),(int(REYE_END.x),int(REYE_END.y)),(255,0,255),2)
    #입꼬리 직선 긋기
    frame = cv2.line(frame,(int(LLIP.x),int(LLIP.y)),(int(RLIP.x),int(RLIP.y)),(255,0,255),2)
    #코 점
    frame = cv2.circle(frame,(int(NOSE_TIP.x),int(NOSE_TIP.y)),5,(255,0,0),-1)
    #미간 점
    frame = cv2.circle(frame,(int(GLABELLA.x),int(GLABELLA.y)),5,(255,0,0),-1)
    #턱끝 점
    frame = cv2.circle(frame,(int(CHIN.x),int(CHIN.y)),5,(255,0,0),-1)
    
    return frame

def Face_Video_result():   
    # 동영상 파일 경로
    video_path = ROOT_DIR + '/image/face_color_output.mp4'

    # 동영상 파일 로드
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        i = 0
        while cap.isOpened():
            i+=1
            success, frame = cap.read()
            try:
                if not success or frame.shape is not None:
        
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    HO_landmark = holistic.process(frame) 
                    FC_landmark = face_mesh.process(frame)
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    F_Shoulder(HO_landmark)
                    frame = Face_line(FC_landmark, frame)
                    Eye_lip_angle(LEYE_END.x,LEYE_END.y,REYE_END.x,REYE_END.y,LLIP.x,LLIP.y,RLIP.x,RLIP.y)
                    Face_angle(GLABELLA.x,GLABELLA.y,UPPERLIP.x,UPPERLIP.y)
                    Nose_chin_glabelly()
                    
                    '''mp_drawing.draw_landmarks(
                        frame,
                        HO_landmark.face_landmarks,
                        mp_holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    ) ''' 
                    cv2.imshow('Output', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                Sum_Face(i)
                F_CENTER_SCORE(FACE_DEG.guide)
                F_LR_SCORE()

                cv2.destroyAllWindows()
                break

        cap.release()

            
