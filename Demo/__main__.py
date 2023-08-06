from tkinter import *
from tkinter import ttk
from Library import *
from media import *
from result import *
from walking import *
from face_color import *
from eye_size import *

from PIL import ImageTk, Image
from tkcalendar import DateEntry
from dataclasses import dataclass
import os
import math


from scipy.stats import norm 
from scipy import stats as ss
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
IMAGE_DIR = ROOT_DIR + '/assets/images/'
BASE_BG = '#E1E9F6'
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

@dataclass
class data:
    gender: str = ''
    phone: str = ''
    height: int = 0
    weight: int = 0
    birth: str = ''

################################ Custom functions ###################################

def image2photo(path,size):
    image = Image.open(path)
    image = image.resize(size,Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    return photo

def MoveScreen(self,Move):
    self.place_forget()
    self.master.SCREEN[Move].place(x=0,y=0,width=640, height=480)


class SpiderChart(Canvas):
    #a canvas that displays datapoints as a SpiderChart
    DATA = [('Face_lr', F_SCORE_LR_I.guide),('Face_center', F_SCORE_CENTER_I.guide), ('Shoulder', S_SCORE_I.guide), ('Face_color', 67), ('Walk', 22)]
    print(F_SCORE_CENTER_I, S_SCORE_I, F_SCORE_LR_I)
    width = 280
    height = 280
    
    def __init__(self, master, datapoints, concentrics=7, scale=150):
        super().__init__(master, width=self.width, height=self.height)
        self.scale = scale * (180/280)
        self.center = self.width // 2, self.height // 2
        self.labels = tuple(d[0] for d in datapoints)
        self.values = tuple(d[1] for d in datapoints)
        self.num_pts = len(self.labels)
        self.concentrics = [n/(concentrics) for n in range(1, concentrics + 1)]
        self.draw()
        
    def position(self, x, y):
        cx, cy = self.center
        return x + cx, cy - y
    
    def draw_circle_from_radius_center(self, radius):
        rad = radius * self.scale
        x0, y0 =  self.position(-rad, rad)
        x1, y1 =  self.position(rad, -rad)
        return self.create_oval(x0, y0, x1, y1)
    
    def draw_label(self, idx, label):
        angle = idx * (2 * math.pi) / self.num_pts
        d = self.concentrics[-1] * self.scale
        x, y = d * math.cos(angle), d * math.sin(angle)
        self.create_line(*self.center, *self.position(x, y), dash=(1, 3))
        d *= 1.2 
        x, y = d * math.cos(angle), d * math.sin(angle)
        self.create_text(*self.position(x, y), text=label)
        
    def draw_polygon(self):
        points = []
        for idx, val in enumerate(self.values):
            d = (val / 100) * self.scale
            angle = idx * (2 * math.pi) / self.num_pts
            x, y = d * math.cos(angle), d * math.sin(angle)
            points.append(self.position(x, y))
        self.create_polygon(points, fill='skyblue')
        
        for idx, val in enumerate(self.concentrics):
            lines = []
            for i in range(5):
                x = val * self.scale
                x_rotated = x * math.cos(math.radians(72 * i)) - 0 * math.sin(math.radians(72 * i))
                y_rotated = x * math.sin(math.radians(72 * i)) + 0 * math.cos(math.radians(72 * i))
                lines.append(self.position(x_rotated, y_rotated))
            self.create_polygon(lines, fill='', outline='black', width=1)
            
        
    def draw(self):
        self.draw_polygon()
        for idx, label in enumerate(self.labels):
            self.draw_label(idx, label)
    
################################ Language Screen ################################
def ImageButton(master, image_path, size, text=" ", loc=LEFT ,**kwargs):
    if image_path:
        # Load Image and Resize
        image = Image.open(image_path)
        image = image.resize(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Setting Button Styles
        style = ttk.Style()
        style.configure(
            "RoundedButton.TButton",
            borderwidth=0,
            relief="flat",
            background=BASE_BG,
            foreground="#000",
            #padding=(0, 5, 0, 5),
            font=("Arial", 15),
        )
        style.map(
            "RoundedButton.TButton",
            background=[("active", "#aaa")],
            foreground=[("active", "#000")],
        )
        button = ttk.Button(
            master,
            image=photo,
            text=text,
            compound=loc,
            style="RoundedButton.TButton",
            **kwargs,
        )
        button.photho = photo
    else:
        # Setting Button Styles
        style = ttk.Style()
        style.configure(
            "RoundedButton.TButton",
            borderwidth=0,
            relief="flat",
            background=BASE_BG,
            foreground="#000",
            padding=(0, 5, 0, 5),
            font=("Arial", 15),
        )
        style.map(
            "RoundedButton.TButton",
            background=[("active", "#aaa")],
            foreground=[("active", "#000")],
        )
        button = ttk.Button(
            master,
            text=text,
            style="RoundedButton.TButton",
            **kwargs,
        )
    return button

class LangSelectScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        self.lang_label = Label(self)        
        #self.lang_label.place(padx=0, pady=30)
        size = (45,30)
        
        self.ko_button = ImageButton(self,IMAGE_DIR+'KO.png',size=size,text='   한국어',loc=LEFT, command=lambda :MoveScreen(self,'LOGIN'))
        self.cn_button = ImageButton(self,IMAGE_DIR+'CN.png',size=size,text='   中國語',loc=LEFT, command=lambda :MoveScreen(self,'LOGIN'))
        self.ja_button = ImageButton(self,IMAGE_DIR+'JA.png',size=size,text='   日本語',loc=LEFT, command=lambda :MoveScreen(self,'LOGIN'))
        self.en_button = ImageButton(self,IMAGE_DIR+'EN.png',size=size,text='   English',loc=LEFT, command=lambda :MoveScreen(self,'LOGIN'))
        
        self.ko_button.place(x=220,y=75,width=200,height=50)
        self.en_button.place(x=220,y=150,width=200,height=50)
        self.ja_button.place(x=220,y=225,width=200,height=50)
        self.cn_button.place(x=220,y=300,width=200,height=50)
        
################################ Login Screen ################################
class LoginScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        #local value
        self.ID = ''
        size = (30,30)
        pos =(50,138)
        #Load Image and resize
        self.input_lbl = Label(self)
        self.input_lbl.config(background='#FFF')
        self.input_lbl.place(x=50,y=70, width=270, height=50)
        
        image = Image.open(IMAGE_DIR+'Phone.png').resize((20,20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label = ttk.Label(self, image=photo,background='#FFF')
        self.image_label.place(x=70, y= 85)
        self.image_label.photo = photo
        
        self.PhoneNumber_lbl = Label(self, text='Input Phone Number', font=('Arial 15'),background='#FFF')
        self.PhoneNumber_lbl.place(x=100,y=70,width=220, height=50)
        
        for row_y,row in enumerate(KEY_LIST):
            for col_x, key in enumerate(row):
                key_button = ImageButton(self,image_path=IMAGE_DIR+key+'.png',size=(25,30),loc=CENTER,text=None, command=lambda k=key: self.key_press(k))
                key_button.place(x=pos[0]+(95*col_x), y=pos[1]+(70.78*row_y), width=80, height=59.78)
        
        self.Logo = Frame(self,background='#2D9CDB')
        self.Logo.place(x=365, y=70, width=225, height=268)
        
        self.sign_button = ImageButton(self,image_path=None,size=(225,50),text='회원 가입',loc=LEFT, command=lambda :MoveScreen(self,'TERMS'))
        self.sign_button.place(x=365,y=350,width=225,height=50)
        
    def key_press(self, k):
        
        if k == 'Back':
            if self.ID != '':
                self.ID = self.ID[:-1]
        elif k == 'Login':
            #*to home screen
            MoveScreen(self,'HOME')
        else:
            if len(self.ID) == 13:
                return
            if len(self.ID) == 3 or len(self.ID) == 8:
                self.ID += '-'
            self.ID += k
        
        self.PhoneNumber_lbl.config(text=self.ID)
        
        
################################ Sign in Screen ################################
class TermsScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        choice = IntVar()
        self.choice = 0
        self.termcontent = Frame(self,background='#D9D9D9')
        self.termcontent.place(x=145, y=16, width=350, height=400)
        
        self.agree = Radiobutton(self.termcontent,background='#D9D9D9',variable=choice,value=1,font='Arial 10',text='동의함',command=lambda : self.Choose(1))
        self.disagree = Radiobutton(self.termcontent,background='#D9D9D9',variable=choice,value=0,font='Arial 10',text='동의안함',command=lambda : self.Choose(0))
        self.agree.place( x=60, y=360, width=100, height=25)
        self.disagree.place( x=190, y=360, width=100, height=25)
        
        self.nextBtn = ImageButton(self,None,(100,40),text='Next',loc=CENTER, command=lambda :MoveScreen(self,'SIGN'))
        self.backBtn = ImageButton(self,None,(100,40),text='Back',loc=CENTER, command=lambda :MoveScreen(self,'LOGIN'))
        self.nextBtn.configure(state='disabled')
        self.nextBtn.place(x=335,y=430, width=100, height=40)
        self.backBtn.place(x=205,y=430, width=100, height=40)
        
    def Choose(self,k):
        if k == 1:
            self.choice = 1
            self.nextBtn.configure(state='normal')
        else:
            self.choice = 0
            self.nextBtn.configure(state='disabled')

class SignScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        #local value
        gender = IntVar()
        self.person = data()
        
        ############################### Content ###################################
        #pos_content = (145,16)
        self.gender = ''
        
        self.content = Frame(self, background='#D9D9D9')
        self.content.place(x=145, y=16, width=350, height=400)
        
        self.templbl = Label(self.content, text='데이터 입력란', background='#D9D9D9')
        self.templbl.place(x=0,y=0, width=350, height=36)
        
        ############################### DateContent ###############################
        self.datacontent = Frame(self.content, background='#D9D9D9')
        self.datacontent.place(x=46,y=60, width=258, height=175)
        
        #gender
        self.gender_lbl = Label(self.datacontent, text='성별 : ', font='Arial 10', compound=LEFT, background='#D9D9D9')
        self.gender_lbl.place(x=0,y=0, width=100, height=35)
        self.male =Radiobutton(self.datacontent, text='남',background='#D9D9D9', variable=gender,value=0, font='Arial 10', command=self.Gender(0))
        self.female =Radiobutton(self.datacontent, text='여',background='#D9D9D9', variable=gender,value=1, font='Arial 10', command=self.Gender(1))
        self.male.place(x=109, y=5, width=50, height=25)
        self.female.place(x=190, y=5, width=50, height=25)
        
        #*Phone
        self.phone_lbl = Label(self.datacontent, text='전화 번호 : ',  font='Arial 10', compound=LEFT, background='#D9D9D9')
        self.phone_lbl.place(x=0,y=35,width=100,height=35)
        self.phoneFrame = Entry(self.datacontent, text='Input Phone Number', font='Arial 10')
        self.phoneFrame.bind("<Button-1>", self.entry_click)
        self.phoneFrame.place(x=109,y=40, width=140, height=25)
        self.KeyFrame = Frame(self)
        
        #*birth
        self.birth_lbl = Label(self.datacontent, text='생년 월일 : ',  font='Arial 10', compound=CENTER, background='#D9D9D9')
        self.birth_lbl.place(x=0,y=70,width=100,height=35)
        self.date_picker = DateEntry(self.datacontent, background='darkblue', foreground='white', borderwidth=2)
        self.date_picker.place(x=109,y=75, width=140, height=25)
        
        #height, weight
        self.height_lbl = Label(self.datacontent, text='키 : cm', font='Arial 10', compound=LEFT, background='#D9D9D9')
        self.weight_lbl = Label(self.datacontent, text='몸무게 : kg', font='Arial 10', compound=LEFT, background='#D9D9D9')
        self.height_lbl.place(x=0,y=105, width=100, height=35)
        self.weight_lbl.place(x=0,y=140, width=100, height=35)
        #padding 16
        
        self.slider_height = ttk.Scale(self.datacontent, from_=80, to=220, orient=HORIZONTAL, length=140, command=self.update_height)
        self.slider_weight = ttk.Scale(self.datacontent, from_=30, to=160, orient=HORIZONTAL, length=140, command=self.update_weight)
        self.slider_height.place(x=109, y=110)
        self.slider_weight.place(x=109, y=145)
        ###########################################################################
        ############################### 질병 체크 ##################################
        self.check_box = Frame(self.content, bg='#D9D9D9')
        self.check_box.place(x=70, y=241,width=204, height=131)
        #box size = (75,20) y:상하 패딩 7, x:좌우패딩 55
        style = ttk.Style()
        style.configure("Custom.TCheckbutton", font=("Arial", 12), background='#D9D9D9')
        for row_y, row in enumerate(DISEASE_LIST):
            for col_x, item in enumerate(row):
                checkbox = ttk.Checkbutton(self.check_box, text=item, style='Custom.TCheckbutton')
                checkbox.place(x=0+(col_x*130), y=0+(row_y*27), width=75, height=20)
        
        ############################### 질병 체크 ###################################
        ############################### Content ###################################
        
        self.nextBtn = ImageButton(self,None,(100,40),text='Next',loc=CENTER, command=lambda :self.get_input('CHARACTER'))
        self.backBtn = ImageButton(self,None,(100,40),text='Back',loc=CENTER, command=lambda :MoveScreen(self,'TERMS'))
        #self.nextBtn.configure(state='disabled')
        self.nextBtn.place(x=335,y=430, width=100, height=40)
        self.backBtn.place(x=205,y=430, width=100, height=40)
        
    def entry_click(self,event):
        def key_press(self, k):
            if k == 'Back':
                if self.person.phone != '':
                    self.person.phone = self.person.phone[:-1]
            elif k == 'Login':
                self.KeyFrame.place_forget()
                print('enter')
            else:
                if len(self.person.phone) == 13:
                    return
                if len(self.person.phone) == 3 or len(self.person.phone) == 8:
                    self.person.phone += '-'
                self.person.phone += k
            self.phoneFrame.delete(0,'end')
            self.phoneFrame.insert(0, self.person.phone)
        #초기화
        self.person.phone = ''
        self.phoneFrame.delete(0,'end') 
        
        self.KeyFrame.place(x=291,y=146, width=158, height=200)
        self.KeyFrame.lift()
        for row_y,row in enumerate(KEY_LIST):
            for col_x, key in enumerate(row):
                key_button = ImageButton(self.KeyFrame,image_path=IMAGE_DIR+key+'.png',size=(27,23),loc=CENTER,text=' ', command=lambda k=key: key_press(self,k))
                key_button.place(x=10+(48*col_x), y=10+(53*row_y), width=43, height=38)
            
    def update_height(self,value):
        self.height_lbl.config(text=f'키 : {int(float(value))} cm')
        
    def update_weight(self,value):
        self.weight_lbl.config(text=f'몸무게 : {int(float(value))} kg')

    def get_input(self, To):
        MoveScreen(self, To)
        
    def Gender(self,k):
        if k == 0:
            self.person.gender = 'Male'
        else:
            self.person.gender = 'Female'
        
class CharacterScreen(Frame):            
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        self.character_lbl = Label(self,background=BASE_BG)
        self.character_lbl.place(x=120,y=137,width=450,height=200)
        
        # 라디오버튼을 위한 IntVar 변수 생성
        self.radio_var = IntVar()
        self.image_list = []
        self.selected_button = None
        style = ttk.Style()
        style.configure('Custom.TRadiobutton', background=BASE_BG)
        i = 0
        for row_y,row in enumerate(CHARACTER_LIST):
            for col_x, key in enumerate(row):
                # 이미지 로드
                image = Image.open(IMAGE_DIR+key+'.png')
                image = image.resize((60, 60), Image.Resampling.LANCZOS)  # 이미지 크기 조정
                photo = ImageTk.PhotoImage(image)
                self.image_list.append(photo)
                
                key_button = ttk.Radiobutton(self.character_lbl, text=None, image=photo, compound="left", variable=self.radio_var,style="Custom.TRadiobutton", value=i, command=lambda idx=i : self.select_button(idx))
                key_button.place(x=0+(113*col_x), y=0+(113*row_y), width=80, height=60)
                i+=1
                
        self.nextBtn = ImageButton(self,None,(100,40),text='Next',loc=CENTER, command=lambda :MoveScreen(self,'LOGIN'))
        self.backBtn = ImageButton(self,None,(100,40),text='Back',loc=CENTER, command=lambda :MoveScreen(self,'SIGN'))
        #self.nextBtn.configure(state='disabled')
        self.nextBtn.place(x=335,y=430, width=100, height=40)
        self.backBtn.place(x=205,y=430, width=100, height=40)
    
    def select_button(self,idx):
            self.selected_button = idx
        
    def get_input(self):
        #self.selected_button 넘겨주기
        return
    
################################ Home Screen ################################    
class HomeScreen(Frame):            
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        #User data
        self.User_lbl = Label(self)        
        #load user data
        image = Image.open(IMAGE_DIR+'토끼.png').resize((60,60), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.photo = photo
        self.User_lbl.config(image=photo)
        self.User_lbl.place(x=20, y=20, width=60, height=60)
        
        
        self.shoulder = image2photo(IMAGE_DIR+'어깨.png', (100,100))
        self.shoulder_lbl = Label(self, image=self.shoulder, bg=BASE_BG, bd=0, relief="flat")
        self.shoulder_lbl.bind("<Button-1>", lambda event :self.onclick(event,'REPORT_SHOULDER'))
        self.shoulder_lbl.place(x=177,y=111, width=100, height=100)

        self.face = image2photo(IMAGE_DIR+'얼굴.png', (100,100))
        self.face_lbl = Label(self, image=self.face, bg=BASE_BG, bd=0, relief="flat")
        self.face_lbl.bind("<Button-1>", lambda event :self.onclick(event,'REPORT_FACE'))
        self.face_lbl.place(x=287,y=111, width=100, height=100)
        
        self.fcolor = image2photo(IMAGE_DIR+'피부색.png', (100,100))
        self.fcolor_lbl = Label(self, image=self.fcolor, bg=BASE_BG, bd=0, relief="flat")
        self.fcolor_lbl.bind("<Button-1>", lambda event :self.onclick(event,'REPORT_FACE_COLOR'))
        self.fcolor_lbl.place(x=177,y=223, width=100, height=100)
        
        self.walk = image2photo(IMAGE_DIR+'보행.png', (100,100))
        self.walk_lbl = Label(self, image=self.walk, bg=BASE_BG, bd=0, relief="flat")
        self.walk_lbl.bind("<Button-1>", lambda event :self.onclick(event,'REPORT_WALKING'))
        self.walk_lbl.place(x=287,y=223, width=100, height=100)
        
        self.eye = image2photo(IMAGE_DIR+'눈.png', (100,100))
        self.eye_lbl = Label(self, image=self.eye, bg=BASE_BG, bd=0, relief="flat")
        self.eye_lbl.bind("<Button-1>", lambda event :self.onclick(event,'REPORT_EYE'))
        self.eye_lbl.place(x=287,y=331,width=100,height=100)

        self.img2 = image2photo(IMAGE_DIR+'파란색.png', (100,100))
        self.lbl4 = Label(self, image=self.img2, bg=BASE_BG, bd=0, relief="flat")
        self.lbl4.place(x=177,y=331,width=100,height=100)
        
        '''#임시 버튼
        self.button = Button(self, text='임시 버튼',command=lambda :MoveScreen(self,'LOGIN'))
        self.button.place(x=0,y=0, width=100, height=10)'''

    def onclick(self,event,To):
        MoveScreen(self,To)


################################ Report Screen ############################

#어깨 측정 및 결과
class ReportShoulderScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, background=BASE_BG,text='{User} 님의 어깨 측정 결과'.format(User='사용자'),image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        #report
        def re_start_S(event):
            PIPELINE.start(CONFIG)
            Media_Shoulder()
            cv2.destroyAllWindows()
            PIPELINE.stop()
            
            Video_result()
            
            label_text_R_TEXT = R_TEXT.guide  # 텍스트 문자열 생성
            label_text_R_TY = R_TY_TEXT.guide
            label_text_R_SCORE = S_SCORE.guide
            label_text = f"어깨비대칭: {label_text_R_TEXT}\n어깨척도: {label_text_R_TY}\n어깨점수: {label_text_R_SCORE}"
            
            self.report = Label(self,background='#D9D9D9', text=label_text, compound=TOP, padx=10, font='Arial 12')
            self.report.place(x=136,y=103, width=368, height=291)
        
        #result button
        self.resultS = image2photo(IMAGE_DIR+'결과보기.png', (100,30))
        self.resultSBtn = Label(self, text='', image=self.resultS, bg=BASE_BG, bd=0, relief="flat")
        self.resultSBtn.bind('<Button-1>', re_start_S)
        self.resultSBtn.place(x=270, y=354, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'big_home.png', (150,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=136, y=415, width=150, height=40)
        
        #report button
        self.finalreport = image2photo(IMAGE_DIR+'종합 결과.png', (150,40))
        self.finalreportBtn = Label(self, text='', image=self.finalreport, bg=BASE_BG, bd=0, relief="flat")
        self.finalreportBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.finalreportBtn.place(x=354, y=415, width=150, height=40)
    
    def onclick(self,event,To):
        MoveScreen(self,To)

#얼굴 측정 및 결과            
class ReportFaceScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, background=BASE_BG,text='{User} 님의 얼굴 측정 결과'.format(User='사용자'),image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        #report
        def re_start_F(event):
            PIPELINE.start(CONFIG)
            Media_Face()
            cv2.destroyAllWindows()
            PIPELINE.stop()
            
            Face_Video_result()
            
            if(FC_LR_TEXT.guide != '대칭' or FC_CENTER_TEXT.guide != '대칭'):
                label_text_F_TEXT = F_TEXT.guide  # 텍스트 문자열 생성
            label_text_FA_TEXT = FA_TEXT.guide
            label_text_FC_LR_TEXT = FC_LR_TEXT.guide
            label_text_FC_CENTER_TEXT = FC_CENTER_TEXT.guide
            label_text_F_SCORE_LR = F_SCORE_LR.guide
            label_text_F_SCORE_CENTER = F_SCORE_CENTER.guide
            if(FC_LR_TEXT.guide != '대칭' or FC_CENTER_TEXT.guide != '대칭'):
                label_text = f"얼굴비대칭: {label_text_F_TEXT}\n눈과입의각도: {label_text_FA_TEXT}\n좌우 안면비대칭: {label_text_FC_LR_TEXT}\n중앙 안면비대칭: {label_text_FC_CENTER_TEXT}\n좌우안면점수: {label_text_F_SCORE_LR}\n중앙안면점수: {label_text_F_SCORE_CENTER}" # 라벨 생성 및 텍스트 설정
            else:
               label_text = f"눈과입의각도: {label_text_FA_TEXT}\n좌우 안면비대칭: {label_text_FC_LR_TEXT}\n중앙 안면비대칭: {label_text_FC_CENTER_TEXT}\n좌우안면점수: {label_text_F_SCORE_LR}\n중앙안면점수: {label_text_F_SCORE_CENTER}" # 라벨 생성 및 텍스트 설정 
                
            self.report = Label(self,background='#D9D9D9', text=label_text, compound=TOP, padx=10, font='Arial 12')
            self.report.place(x=136,y=103, width=368, height=291)
        
        #result button
        self.resultF = image2photo(IMAGE_DIR+'결과보기.png', (100,30))
        self.resultFBtn = Label(self, text='', image=self.resultF, bg=BASE_BG, bd=0, relief="flat")
        self.resultFBtn.bind('<Button-1>', re_start_F)
        self.resultFBtn.place(x=270, y=354, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'big_home.png', (150,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=136, y=415, width=150, height=40)
        
        #report button
        self.finalreport = image2photo(IMAGE_DIR+'종합 결과.png', (150,40))
        self.finalreportBtn = Label(self, text='', image=self.finalreport, bg=BASE_BG, bd=0, relief="flat")
        self.finalreportBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.finalreportBtn.place(x=354, y=415, width=150, height=40)

    def onclick(self,event,To):
        MoveScreen(self,To)

#얼굴 안색 측정 및 결과
class ReportFaceColorScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, background=BASE_BG,text='{User} 님의 피부 색 측정 결과'.format(User='사용자'),image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        #report
        def re_start_FC(event):
            PIPELINE.start(CONFIG)
            Media_FaceColor()
            cv2.destroyAllWindows()
            PIPELINE.stop()
            
            Face_Color_Image_result()
            
            label_text_Face_Color1 = RGB_COLOR1.guide
            label_text_Face_Color2 = RGB_COLOR2.guide
            
            label_text = f"얼굴 평균 RGB COLOR:\n 왼쪽얼굴: {label_text_Face_Color1}\n 오른쪽 얼굴: {label_text_Face_Color2}" # 라벨 생성 및 텍스트 설정
            
            image_path1 = 'C:/Users/User/Desktop/new/image/rgb_color_output1.jpg'
            image_path2 = 'C:/Users/User/Desktop/new/image/rgb_color_output1.jpg'

            # PIL 이미지로 변환
            pil_image1 = Image.open(image_path1)
            pil_image2 = Image.open(image_path2)
            # ImageTk.PhotoImage 객체 생성
            self.image_tk1 = ImageTk.PhotoImage(pil_image1)
            self.image_tk2 = ImageTk.PhotoImage(pil_image2)
                           
            self.report = Label(self,background='#D9D9D9', text=label_text, compound=tk.TOP, padx=10, font='Arial 12', anchor=tk.N)
            self.report.place(x=136,y=103, width=368, height=291)
            # Label 위젯에 PIL 이미지를 표시
            self.report = tk.Label(self, image=self.image_tk1, compound=tk.TOP, padx=10)
            self.report.place(x=166, y=216, width=132, height=93)
            self.report = tk.Label(self, image=self.image_tk2, compound=tk.TOP, padx=10)
            self.report.place(x=346, y=216, width=132, height=93)
        
        #result button
        self.resultF = image2photo(IMAGE_DIR+'결과보기.png', (100,30))
        self.resultFBtn = Label(self, text='', image=self.resultF, bg=BASE_BG, bd=0, relief="flat")
        self.resultFBtn.bind('<Button-1>', re_start_FC)
        self.resultFBtn.place(x=270, y=354, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'big_home.png', (150,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=136, y=415, width=150, height=40)
        
        #report button
        self.finalreport = image2photo(IMAGE_DIR+'종합 결과.png', (150,40))
        self.finalreportBtn = Label(self, text='', image=self.finalreport, bg=BASE_BG, bd=0, relief="flat")
        self.finalreportBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.finalreportBtn.place(x=354, y=415, width=150, height=40)

    def onclick(self,event,To):
        MoveScreen(self,To)
        
#보행 측정 및 결과
class ReportWalkingScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, background=BASE_BG,text='{User} 님의 보행 측정 결과'.format(User='사용자'),image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        label_text = f"주의사항: 오른쪽 발부터 내딛어주세요." # 라벨 생성 및 텍스트 설정
        self.report = Label(self,background='#D9D9D9', text=label_text, compound=tk.TOP, padx=10, font='Arial 12', anchor=tk.N)

        #report
        def re_start_Walk(event):
            #PIPELINE.start(CONFIG)
            #Media_Step()
            #cv2.destroyAllWindows()
            #PIPELINE.stop()
            
            One_Step_L()
            One_Step_R()
            Step_Video_result_L()
            Step_Video_result_R()
            Knee_Measure_L(CUT_OFF_T_L.x)
            Knee_Measure_R(CUT_OFF_T_R.x)

            label_text = f"왼쪽 발, 오른쪽 발" # 라벨 생성 및 텍스트 설정

            image_path1 = 'C:/Users/User/Desktop/new/image/graph_image_L.jpg'
            image_path2 = 'C:/Users/User/Desktop/new/image/graph_image_R.jpg'

            img1 = Image.open(image_path1)
            img2 = Image.open(image_path2)

            # 이미지 크기 조절
            desired_width = 132
            desired_height = 93
            img1 = img1.resize((desired_width, desired_height))
            img2 = img2.resize((desired_width, desired_height))

            # ImageTk.PhotoImage 객체 생성
            self.image_tk1 = ImageTk.PhotoImage(img1)
            self.image_tk2 = ImageTk.PhotoImage(img2)

            self.report = Label(self,background='#D9D9D9', text=label_text, compound=tk.TOP, padx=10, font='Arial 12', anchor=tk.N)
            self.report.place(x=136,y=103, width=368, height=291)               
            # Label 위젯에 PIL 이미지를 표시
            self.report = tk.Label(self, image=self.image_tk1, compound=tk.TOP, padx=10)
            self.report.place(x=166, y=216, width=132, height=93)
            self.report = tk.Label(self, image=self.image_tk2, compound=tk.TOP, padx=10)
            self.report.place(x=346, y=216, width=132, height=93)
            
        
        #result button
        self.resultF = image2photo(IMAGE_DIR+'결과보기.png', (100,30))
        self.resultFBtn = Label(self, text='', image=self.resultF, bg=BASE_BG, bd=0, relief="flat")
        self.resultFBtn.bind('<Button-1>', re_start_Walk)
        self.resultFBtn.place(x=270, y=354, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'big_home.png', (150,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=136, y=415, width=150, height=40)
        
        #report button
        self.finalreport = image2photo(IMAGE_DIR+'종합 결과.png', (150,40))
        self.finalreportBtn = Label(self, text='', image=self.finalreport, bg=BASE_BG, bd=0, relief="flat")
        self.finalreportBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.finalreportBtn.place(x=354, y=415, width=150, height=40)

    def onclick(self,event,To):
        MoveScreen(self,To) 

#눈 크기 측정 및 결과
class ReportEyeScreen(Frame):
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, background=BASE_BG,text='{User} 님의 눈 크기 측정 결과'.format(User='사용자'),image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        #report
        def re_start_Eye(event):
            PIPELINE.start(CONFIG)
            Media_Eye()
            cv2.destroyAllWindows()
            PIPELINE.stop()
            
            Eye_Size()
            eyes_area()
            
            label_text_R_EYE_HEI_TEXT = HEI_REYE_TEXT.guide
            label_text_R_EYE_WID_TEXT = WID_REYE_TEXT.guide
            label_text_L_EYE_HEI_TEXT = HEI_LEYE_TEXT.guide
            label_text_L_EYE_WID_TEXT = WID_LEYE_TEXT.guide
            label_text_left_eye_area_TEXT = LEFT_EYE_AREA.guide
            label_text_right_eye_area_TEXT = RIGHT_EYE_AREA.guide
            label_text = f"오른쪽 눈 높이: {label_text_R_EYE_HEI_TEXT}\n오른쪽 눈 너비: {label_text_R_EYE_WID_TEXT}\n왼쪽 눈 높이: {label_text_L_EYE_HEI_TEXT}\n왼쪽 눈 너비: {label_text_L_EYE_WID_TEXT}\n왼쪽 눈 영역: {label_text_left_eye_area_TEXT}\n오른쪽 눈 영역: {label_text_right_eye_area_TEXT}"
            
            self.report = Label(self,background='#D9D9D9', text=label_text, compound=TOP, padx=10, font='Arial 12')
            self.report.place(x=136,y=103, width=368, height=291)
            
        
        #result button
        self.resultF = image2photo(IMAGE_DIR+'결과보기.png', (100,30))
        self.resultFBtn = Label(self, text='', image=self.resultF, bg=BASE_BG, bd=0, relief="flat")
        self.resultFBtn.bind('<Button-1>', re_start_Eye)
        self.resultFBtn.place(x=270, y=354, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'big_home.png', (150,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=136, y=415, width=150, height=40)
        
        #report button
        self.finalreport = image2photo(IMAGE_DIR+'종합 결과.png', (150,40))
        self.finalreportBtn = Label(self, text='', image=self.finalreport, bg=BASE_BG, bd=0, relief="flat")
        self.finalreportBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.finalreportBtn.place(x=354, y=415, width=150, height=40)

    def onclick(self,event,To):
        MoveScreen(self,To) 
               
#전체 스파이더차트 결과            
class ReportScreen_1(Frame):            
    def __init__(self, master):
        super().__init__(master)
        #set Screen)
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, text='{User} 님의 종합 결과'.format(User='사용자'),background=BASE_BG,image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
        
        #Graph
        def Spider(event):
            self.graph = SpiderChart(self, DATA)
            self.graph.place(x=71, y=109, width=280, height=280)
        
       #report
        self.report_lbl = Label(self, text='설명', font='Arial 10', background="#D9D9D9", compound=LEFT)
        self.report_lbl.place(x=366, y=109, width=200, height=280)
        
        #result button
        self.resultP = image2photo(IMAGE_DIR+'결과보기.png', (150,40))
        self.resultPBtn = Label(self, text='', image=self.resultP, bg=BASE_BG, bd=0, relief="flat")
        self.resultPBtn.bind('<Button-1>', Spider)
        self.resultPBtn.place(x=100, y=415, width=100, height=30)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'small_home.png', (100,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=270, y=415, width=100, height=40)
        
        #next button
        self.next = image2photo(IMAGE_DIR+'big_next.png', (150,40))
        self.nextBtn = Label(self, text='', image=self.next, bg=BASE_BG, bd=0, relief="flat")
        self.nextBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_2'))
        self.nextBtn.place(x=390, y=415, width=150, height=40)
        
    def onclick(self,event,To):
        MoveScreen(self,To)

#전체 그래프 결과            
class ReportScreen_2(Frame):            
    def __init__(self, master):
        super().__init__(master)
        #set Screen
        self.config(background=BASE_BG)
        
        #User data
        photo = image2photo(IMAGE_DIR+'토끼.png', (60,60))
        self.User_lbl = Label(self, text='{User} 님의 종합 결과'.format(User='사용자'),background=BASE_BG, image=photo, compound=LEFT, padx=10, font='Arial 12')
        self.User_lbl.photo = photo
        self.User_lbl.place(x=136,y=33, width=368, height=60)
    
        #임시 코드
        # ------------ 진짜 평균,분산,표준편차 구하는 공식 -------------#
        #real_average = np.mean(value_list) # 진짜 평균
        #real_std = np.std(value_list) # 진짜 표준편차
        # --------------------------------------------------------------#

        average = 70 # 평균(임의)
        standard_deviation = 10 #표준편차(임의)

        my_score = 100 # 원점수
        my_score_st = (my_score - average)/standard_deviation #표준점수

        percent = 1 - round(ss.norm.cdf(my_score_st),3) # 퍼센트로 나타내기
        my_rank = percent * 100 # 백분율
        #print("사용자의 점수는 상위{:.1f}% 입니다".format(my_rank))    
        
        value_list = [100,60,50,60,80,84,55,75,70,60,65,90,80,85,68]

        figure = Figure(figsize=(3.85, 1.45), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self)
        plt = figure.add_subplot()
        plt.hist(value_list,bins=50,histtype='bar')
        plt.set_title("histogram")
        plt.set_xlabel('score')
        plt.set_ylabel('people')
        figure_canvas.get_tk_widget().place(x=127,y=109, width=385, height=200)
        
        #user report
        self.user_report = Label(self, text='사용자의 점수는 상위{:.1f}% 입니다'.format(my_rank), font='Arial 10',compound=TOP,background='#D9D9D9')
        self.user_report.place(x=91,y=315, width=450,height=80)
        #back button
        self.back = image2photo(IMAGE_DIR+'big_back.png', (150,40))
        self.backBtn = Label(self, text='', image=self.back, bg=BASE_BG, bd=0, relief="flat")
        self.backBtn.bind('<Button-1>', lambda event: self.onclick(event,'REPORT_1'))
        self.backBtn.place(x=100, y=415, width=150, height=40)
        
        #home button
        self.home = image2photo(IMAGE_DIR+'small_home.png', (100,40))
        self.homeBtn = Label(self, text='', image=self.home, bg=BASE_BG, bd=0, relief="flat")
        self.homeBtn.bind('<Button-1>', lambda event: self.onclick(event,'HOME'))
        self.homeBtn.place(x=270, y=415, width=100, height=40)
    
    def onclick(self,event,To):
        MoveScreen(self,To)
        
################################ New Screen Form ############################
# class CharacterScreen(Frame):            
#     def __init__(self, master):
#         super().__init__(master)
#         #set Screen
#         self.config(background=BASE_BG)

################################ Main Screen ################################

class MainWindow(Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Window")
        self.geometry("640x480")  # 창의 크기 설정
        #self.option_add('*Font', '궁서 20')
        self.configure(background=BASE_BG)
        self.resizable(width=False, height=False)
        
        self.SCREEN = {
            "CURRENT": Frame(self),
            "SELECTLANG": LangSelectScreen(self),
            "LOGIN":LoginScreen(self),
            "SIGN": SignScreen(self),
            "TERMS": TermsScreen(self),
            "CHARACTER": CharacterScreen(self),
            "HOME": HomeScreen(self),
            
            "REPORT_1" : ReportScreen_1(self),
            "REPORT_2" : ReportScreen_2(self),
            "REPORT_SHOULDER" : ReportShoulderScreen(self),
            "REPORT_FACE" : ReportFaceScreen(self),
            "REPORT_FACE_COLOR" : ReportFaceColorScreen(self),
            "REPORT_WALKING" : ReportWalkingScreen(self),
            "REPORT_EYE" : ReportEyeScreen(self)
            }
        #set first screen
        self.SCREEN['SELECTLANG'].place(x=0,y=0, width=640, height=480)
        
if __name__ == "__main__":
    window = MainWindow()
    window.mainloop()
