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

def Eye_guideline(Depth,Landmarks):
    ## 얼굴의 좌표를 받아옴 ##
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
        # ## 왼쪽눈 눈꼬리 ##
        # Leye_end_landmark = face_landmarks.landmark[263]
        # E_LEYE.x = int(Leye_end_landmark.x * WIDTH)
        # E_LEYE.y = int(Leye_end_landmark.y * HEIGHT)
        # ## 왼쪽눈 눈앞머리 ##
        # Leye_front_landmark = face_landmarks.landmark[362]
        # F_LEYE.x = int(Leye_front_landmark.x * WIDTH)
        # F_LEYE.y = int(Leye_front_landmark.y * HEIGHT)
        # ## 오른쪽눈 눈꼬리 ##
        # Reye_end_landmark = face_landmarks.landmark[33]
        # E_REYE.x = int(Reye_end_landmark.x * WIDTH)
        # E_REYE.y = int(Reye_end_landmark.y * HEIGHT)
        # ## 오른쪽눈 눈앞머리 ##
        # Reye_front_landmark = face_landmarks.landmark[133]
        # F_REYE.x = int(Reye_front_landmark.x * WIDTH)
        # F_REYE.y = int(Reye_front_landmark.y * HEIGHT)
        
        # ## 왼쪽 입꼬리 ##
        # Llip_landmark = face_landmarks.landmark[308]
        # LLIP.x = (Llip_landmark.x * WIDTH)
        # LLIP.y = (Llip_landmark.y * HEIGHT)
        # ## 오른쪽 입꼬리 ##
        # Rlip_landmark = face_landmarks.landmark[78]
        # RLIP.x = (Rlip_landmark.x * WIDTH)
        # RLIP.y = (Rlip_landmark.y * HEIGHT)
        # ## 윗입술 중앙 ##
        # upper_lip_landmark = face_landmarks.landmark[0]
        # UPPERLIP.x = (upper_lip_landmark.x * WIDTH)
        # UPPERLIP.y = (upper_lip_landmark.y * HEIGHT)
        
        # ## 코끝 ##
        # nose_tip_landmark = face_landmarks.landmark[1]
        # NOSE_TIP.x = (nose_tip_landmark.x * WIDTH)
        # NOSE_TIP.y = (nose_tip_landmark.y * HEIGHT)
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
def Media_Eye():
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

                if Eye_guideline(depth, FC_landmark):
                    saveon = Eye_Save()

            """
            mp_drawing.draw_landmarks(
                frame,
                HO_landmark.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            """

            #가이드 라인 추가 및 텍스트 설정
            frame = cv2.circle(frame,(317,146),5,(255,255,0),-1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)
            frame = cv2.bitwise_and(frame,FACE_GUIDELINE)
            frame = cv2.circle(frame, (int(GLABELLA.x),int(GLABELLA.y)),5,(255,0,0),-1)
            frame = GuideText(frame)
            
            
            # frame = cv2.circle(frame,(int(F_REYE.x),int(F_REYE.y)),5,(255,0,0),-1)
            # frame = cv2.circle(frame,(int(E_REYE.x),int(E_REYE.y)),5,(0,255,0),-1)
            # frame = cv2.circle(frame,(int(F_LEYE.x),int(F_LEYE.y)),5,(255,0,0),-1)
            # frame = cv2.circle(frame,(int(E_LEYE.x),int(E_LEYE.y)),5,(0,255,0),-1)
            # frame = cv2.circle(frame,(int(U_REYE.x),int(U_REYE.y)),5,(0,0,255),-1)
            # frame = cv2.circle(frame,(int(U_LEYE.x),int(U_LEYE.y)),5,(0,0,255),-1)
            # frame = cv2.circle(frame,(int(D_LEYE.x),int(D_LEYE.y)),5,(255,0,255),-1)
            # frame = cv2.circle(frame,(int(D_REYE.x),int(D_REYE.y)),5,(255,0,255),-1)
    
            
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
    
# 눈 저장
def Eye_Save():
    FACEVIDEO_COLOR_WRITER = ROOT_DIR + '/image/eye_size_output.jpg'
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
            
            if not Eye_guideline(depth_frame, results):
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
                cv2.imwrite(FACEVIDEO_COLOR_WRITER, cv2.flip(color_image,1))
            if etime > 5:
                break

        print('이미지 저장 완료')
        
    return True


def Eye(Landmarks, frame):
    face_landmarks = Landmarks.multi_face_landmarks[0]

    ## 왼쪽눈 눈꼬리 ##
    Leye_end_landmark = face_landmarks.landmark[263]
    E_LEYE.x = int(Leye_end_landmark.x * WIDTH)
    E_LEYE.y = int(Leye_end_landmark.y * HEIGHT)
    ## 왼쪽눈 눈앞머리 ##
    Leye_front_landmark = face_landmarks.landmark[362]
    F_LEYE.x = int(Leye_front_landmark.x * WIDTH)
    F_LEYE.y = int(Leye_front_landmark.y * HEIGHT)
    ## 오른쪽눈 눈꼬리 ##
    Reye_end_landmark = face_landmarks.landmark[33]
    E_REYE.x = int(Reye_end_landmark.x * WIDTH)
    E_REYE.y = int(Reye_end_landmark.y * HEIGHT)
    ## 오른쪽눈 눈앞머리 ##
    Reye_front_landmark = face_landmarks.landmark[133]
    F_REYE.x = int(Reye_front_landmark.x * WIDTH)
    F_REYE.y = int(Reye_front_landmark.y * HEIGHT)
    
    ## 왼쪽눈 눈위 ##
    Leye_up_landmark = face_landmarks.landmark[386]
    U_LEYE.x = int(Leye_up_landmark.x * WIDTH)
    U_LEYE.y = int(Leye_up_landmark.y * HEIGHT)
    ## 왼쪽눈 눈아래 ##
    Leye_down_landmark = face_landmarks.landmark[374]
    D_LEYE.x = int(Leye_down_landmark.x * WIDTH)
    D_LEYE.y = int(Leye_down_landmark.y * HEIGHT)
    ## 오른쪽눈 눈위 ##
    Reye_up_landmark = face_landmarks.landmark[159]
    U_REYE.x = int(Reye_up_landmark.x * WIDTH)
    U_REYE.y = int(Reye_up_landmark.y * HEIGHT)
    ## 오른쪽눈 눈아래 ##
    Reye_down_landmark = face_landmarks.landmark[145]
    D_REYE.x = int(Reye_down_landmark.x * WIDTH)
    D_REYE.y = int(Reye_down_landmark.y * HEIGHT)
      
    return frame


## 눈 가로세로 길이 측정 함수 ##
def Eye_Length(frame):
    
    dry = D_REYE.y
    ury = U_REYE.y
    dly = D_LEYE.y
    uly = U_LEYE.y
    elx = E_LEYE.x
    flx = F_LEYE.x
    frx = F_REYE.x
    erx = E_REYE.x

    hei_REYE = (dry - ury)
    hei_LEYE = (dly - uly)
    
    wid_LEYE = (elx - flx)
    wid_REYE = (frx - erx)

    HEI_REYE.guide = hei_REYE
    HEI_LEYE.guide = hei_LEYE
    WID_REYE.guide = wid_REYE
    WID_LEYE.guide = wid_LEYE
    
    HEI_LEYE_TEXT.guide = str(HEI_REYE.guide)
    WID_LEYE_TEXT.guide = str(WID_LEYE.guide)
    HEI_REYE_TEXT.guide = str(HEI_REYE.guide)
    WID_REYE_TEXT.guide = str(WID_REYE.guide)

def Eye_Size():
    image_path = 'image/eye_size_output.jpg'
    #파일 로드
    frame = cv2.imread(image_path)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FC_landmark = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = Eye(FC_landmark,frame)
        Eye_Length(frame)

def triangle_area(loc1,loc2,loc3):
    #loc -> x, y
    area = abs((loc1.x*640*(loc2.y*480-loc3.y*480) + loc2.x*640*(loc3.y*480-loc1.y*480) + loc3.x*640*(loc1.y*480-loc2.y*480)) / 2 )
    return area

def eyes_area():
    LEFT_EYE_LOC = [(33,246,7),(246,7,163),(246,161,163),(161,160,144),(144,160,145),(160,159,145),(159,145,153),(153,159,158),(158,153,154),(158,157,154),(157,154,155),(155,157,173),(173,155,133)]
    RIGHT_EYE_LOC = [(362,398,382),(382,398,384),(384,382,381),(381,384,385),(381,385,380),(380,385,386),(380,374,386),(386,387,374),(374,373,387),(373,387,388),(373,390,388),(390,388,466),(390,466,249),(249,466,263)]

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    image_path = 'image/eye_size_output.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        ) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 왼쪽 눈 (예시로 눈의 133번, 145번, 159번 랜드마크를 사용합니다)
                left_eye_landmarks = [
                    face_landmarks.landmark[33],
                    face_landmarks.landmark[7],
                    face_landmarks.landmark[163],
                    face_landmarks.landmark[144],
                    face_landmarks.landmark[145],
                    face_landmarks.landmark[153],
                    face_landmarks.landmark[154],
                    face_landmarks.landmark[155],
                    face_landmarks.landmark[133],
                    face_landmarks.landmark[173],
                    face_landmarks.landmark[157],
                    face_landmarks.landmark[158],
                    face_landmarks.landmark[159],
                    face_landmarks.landmark[160],
                    face_landmarks.landmark[161],
                    face_landmarks.landmark[246],
                ]
                
                right_eye_landmarks = [
                    face_landmarks.landmark[362],
                    face_landmarks.landmark[382],
                    face_landmarks.landmark[398],
                    face_landmarks.landmark[384],
                    face_landmarks.landmark[385],
                    face_landmarks.landmark[386],
                    face_landmarks.landmark[387],
                    face_landmarks.landmark[388],
                    face_landmarks.landmark[466],
                    face_landmarks.landmark[263],
                    face_landmarks.landmark[249],
                    face_landmarks.landmark[390],
                    face_landmarks.landmark[373],
                    face_landmarks.landmark[374],
                    face_landmarks.landmark[380],
                    face_landmarks.landmark[381],
                ]
                 # 이미지에 랜드마크 점 그리기
                for landmark in left_eye_landmarks:
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 점의 크기와 색상 설정
                for landmark in right_eye_landmarks:
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 점의 크기와 색상 설정

                # 눈의 넓이 계산
                left_eye_area = 0.0
                for i in LEFT_EYE_LOC:
                    left_eye_area += triangle_area(face_landmarks.landmark[i[0]],face_landmarks.landmark[i[1]],face_landmarks.landmark[i[2]])
               
                right_eye_area = 0.0
                for i in RIGHT_EYE_LOC:
                    right_eye_area += triangle_area(face_landmarks.landmark[i[0]],face_landmarks.landmark[i[1]],face_landmarks.landmark[i[2]])
                    
        # 결과 이미지 저장 (옵션)
        #cv2.imwrite("D:/Capstone/code/src/output_image.png", image)

        LEFT_EYE_AREA.guide = left_eye_area
        RIGHT_EYE_AREA.guide = right_eye_area
            
