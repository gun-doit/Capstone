from Library import *
from guideline import *

###############얼굴###############
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

            if 0 <= int(CHIN.x) < WIDTH and 0 <= int(CHIN.y) < HEIGHT:
                if 0 <= int(FORHEAD.x) < WIDTH and 0 <= int(FORHEAD.y) < HEIGHT:
                    CHIN.z = depth.get_distance(int(CHIN.x), int(CHIN.y))
                    FORHEAD.z = depth.get_distance(int(FORHEAD.x), int(FORHEAD.y))
                    MIDDLE_LR_F.z = (CHIN.z + FORHEAD.z) / 2
            
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
    FACEVIDEO_COLOR_WRITER = cv2.VideoWriter(ROOT_DIR + '/image/face_color_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (WIDTH, HEIGHT), isColor = True)
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
                FACEVIDEO_COLOR_WRITER.write(cv2.flip(color_image,1))
            if etime > 8:
                break

        # 동영상 저장 종료
        FACEVIDEO_COLOR_WRITER.release()

        print('동영상 저장 완료')
        
    return True
