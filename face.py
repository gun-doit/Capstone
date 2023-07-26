#깃헙에서 library 파일 업뎃필요!
from Library import *


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


def GuideText(frame):
    #텍스트 위치 계산
    text_x, text_y = int((HEIGHT - len(STR.guide)*7) / 2), 40 #가로 중앙으로 설정

    frame = Image.fromarray(frame)
    ImageDraw.Draw(frame).text((text_x,text_y), STR.guide, font=FONT, fill=GREEN)

    frame = np.array(frame)
    return frame

#얼굴 가이드라인 겹치기      
def Media_Face():
    count = 0
    saveon = False
    Start_Time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        while cv2.waitKey(1) < 0:
            frames = PIPELINE.wait_for_frames()
            depth_frames = ALIGN.process(frames)
            frame = frames.get_color_frame()
            depth = depth_frames.get_depth_frame()

            if not frame:
                continue

            frame = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_BGR2RGB)

            HO_landmark = holistic.process(frame) 
            FC_landmark = face_mesh.process(frame) 
            
            # #가이드라인 확인 5초마다
            if time.time() - Start_Time > N_SECONDS:
                count += 1
                Start_Time = time.time()

                if INFace(depth, FC_landmark):
                    saveon = face_save()

            #가이드 라인 추가 및 텍스트 설정
            frame = cv2.circle(frame,(317,146),5,(255,255,0),-1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            frame = cv2.bitwise_and(frame,FACE_GUIDELINE)
            frame = cv2.circle(frame, (int(GLABELLA.x),int(GLABELLA.y)),5,(255,0,0),-1)
            frame = GuideText(frame)
            
            #화면 표시
            cv2.imshow("", frame)

            # 종료 조건
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            # 타임 아웃
            if count > COUNTOUT:
                break
            
            if saveon:
                #저장완료
                break

#얼굴 영상 저장
def face_save():
    FACEVIDEO_COLOR_IMWRITER = ROOT_DIR + '/image/face_color_output.jpg'
    #* 촬영 시작
    STR.guide = '촬영을 시작합니다 3'
    
    stime = cv2.getTickCount()  # 시작 시간 기록
    # RGB 프레임을 받아옴
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        while cv2.waitKey(1) < 0:

            frames = PIPELINE.wait_for_frames()
            depth_frames = ALIGN.process(frames)
            color_frame = frames.get_color_frame()  
            depth_frame = depth_frames.get_depth_frame()
            
            if not color_frame: continue
        
            # RGB 프레임을 이미지로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_iamge = np.asanyarray(depth_frame.get_data())
            
            #가이드 라인 체크
            frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.bitwise_and(frame, FACE_GUIDELINE)
            
            if not INFace(depth_frame, results):
                STR.guide = "가이드라인 안에 들어와주세요"
                return False
            
            frame = cv2.flip(frame,1)
            frame = GuideText(frame)
            cv2.imshow("", frame)


            ctime = cv2.getTickCount()  # 현재 시간 기록
            etime = (ctime - stime) / cv2.getTickFrequency()  # 경과 시간 계산


            # 5초가 경과하면 녹화 종료
            if 1 < etime < 2:
                STR.guide = '촬영을 시작합니다 2'
            elif 2 < etime < 3:
                STR.guide = '촬영을 시작합니다 1'
                
            # 동영상에 프레임을 추가
            elif etime > 3:
                STR.guide = "촬영중입니다. 움직이지마세요."
                cv2.imwrite(FACEVIDEO_COLOR_IMWRITER, cv2.flip(color_image,1))
            if etime > 5:
                break

        print('이미지 저장 완료')
        
    return True

#얼굴 안내선
def Face_line(Landmarks, frame):
    face_landmarks = Landmarks.multi_face_landmarks[0] 

    ## 이마 끝 ##
    forhead_landmark = face_landmarks.landmark[10]
    FORHEAD.x = int(forhead_landmark.x * WIDTH)
    FORHEAD.y = int(forhead_landmark.y * HEIGHT)
    ## 턱 끝 중앙 ##
    chin_landmark = face_landmarks.landmark[152]
    CHIN.x = int(chin_landmark.x * WIDTH)
    CHIN.y = int(chin_landmark.y * HEIGHT)
    ## 얼굴 왼쪽 끝##
    face_left_landmark = face_landmarks.landmark[234]
    FC_LEFT_END.x = int(face_left_landmark.x * WIDTH)
    FC_LEFT_END.y = int(face_left_landmark.y * HEIGHT)
    ## 얼굴 오른쪽 끝##
    face_right_landmark = face_landmarks.landmark[454]
    FC_RIGHT_END.x = int(face_right_landmark.x * WIDTH)
    FC_RIGHT_END.y = int(face_right_landmark.y * HEIGHT)
    
    return frame

def Face_color(face_img):
    x1 = FC_LEFT_END.x
    y1 = FORHEAD.y
    x2 = FC_RIGHT_END.x
    y2 = CHIN.y
    
    #olor = (255, 0, 0)  # 사각형 색상 (파란색)
    #thickness = 1  # 사각형 두께
    #cv2.rectangle(face_img, (x1, y1), (x2, y2), color, thickness)
    face_img = face_img[y1:y2, x1:x2]

    # 사각형이 그려진 이미지를 저장합니다.
    FACEVIDEO_R_IMWRITER = ROOT_DIR + "/image/face_with_rectangle.jpg"
    cv2.imwrite(FACEVIDEO_R_IMWRITER, cv2.flip(face_img,1))
    
    face_img_ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0,133,77], dtype = np.uint8)
    upper = np.array([255,173,127], dtype = np.uint8)
    
    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)

    skin = cv2.bitwise_and(face_img_ycrcb, face_img_ycrcb, mask = skin_msk)
    skin = cv2.cvtColor(skin, cv2.COLOR_YCrCb2BGR)
    
    FACEVIDEO_R_IMWRITER = ROOT_DIR + "/image/face_rgb.jpg"
    cv2.imwrite(FACEVIDEO_R_IMWRITER, cv2.flip(skin,1))

    # 이미지를 BGR에서 HSV로 변환합니다.
    hsv_image = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)

    # 모든 픽셀의 색상 값을 평균하여 구합니다.
    average_color = np.mean(hsv_image, axis=(0, 1))
    
    # HSV 색상값을 BGR 컬러로 변환
    bgr_color = cv2.cvtColor(np.array([[average_color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]

    # 단색 이미지 생성
    color_image = np.zeros((640, 480, 3), dtype=np.uint8)
    color_image[:, :] = bgr_color

    # 이미지를 창에 표시
    cv2.imshow("Color Image", color_image)
    
    # BGR 컬러를 RGB로 변환
    rgb_color = int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])

    print(rgb_color)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#얼굴 사진 저장
def Face_Color_Image_result():   
    # 이미지 파일 경로
    image_path = 'C:/lab/Demo/image/face_color_output.jpg'

    # 이미지 파일 로드
    frame = cv2.imread(image_path)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        HO_landmark = holistic.process(frame) 
        FC_landmark = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
        frame = Face_line(FC_landmark, frame)
        
        Face_color(frame)


PIPELINE.start(CONFIG)
Media_Face()
cv2.destroyAllWindows()
PIPELINE.stop()

Face_Color_Image_result()


