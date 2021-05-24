import cv2
import os
import csv
import sys
import numpy as np 
import pickle
from PIL import Image
import sqlite3


from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, NoTransition, CardTransition
from kivymd.uix.picker import MDThemePicker
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.toolbar import MDToolbar
from kivy.lang import Builder

from kivy.utils import platform
from kivy.properties import ObjectProperty

from profilephotomanager import ProfilePhotoManager
from specialbuttons import LabelButton, ImageButton


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

class HomeScreen(Screen):
    def Detect(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingData/trainner.yml")

        labels = {"person_name": 1}
        with open("TrainingData/facelabels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v:k for k,v in og_labels.items()}

        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                #print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                roi_color = frame[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                if conf>=50 and conf <= 87:
                    name = labels[id_]
                    cv2.putText(frame, name, (x+10,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #subitems = smile_cascade.detectMultiScale(roi_gray)
                #for (ex,ey,ew,eh) in subitems:
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()

class HelpScreen(Screen):
    pass    

class ProfilePhotoScreen(Screen):
    pass

class TakePhotoScreen(Screen):
    def TakePhoto(self):
        #print(name)
        name = self.ids.gettext.text
        
        self.flag = 0
        def face_extractor(img):
            gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.4,minNeighbors= 5)

            if faces is():
                return None
            
            for(x,y,w,h) in faces:
                cropped_face = img[y:y+h,x:x+w]
            
            return cropped_face

        def create_folder(name):
            os.chdir('faces')

            if os.path.exists(name):
                checkstring = "Sorry this username already exists.Try another username!!!!"     
                close_button = MDFlatButton(text="Close",on_release= self.close)
                self.dialog = MDDialog(title="Check Username",text=checkstring,
                                        size_hint = (0.5,1),
                                        buttons = [close_button]  )
                self.dialog.open()
        
        
            elif name is "":
                checkstring = "Please enter a Username"
                close_button = MDFlatButton(text="Close",on_release=self.close)
                self.dialog = MDDialog(title="Check Username",text=checkstring,
                                        size_hint = (0.5,1),
                                        buttons = [close_button]  )
                self.dialog.open()
                
            else:
                self.flag = 1
                os.mkdir(name)
                checkstring = "Now train your data."
                close_button = MDFlatButton(text="Close",on_release=self.close)
                self.dialog = MDDialog(title=f"Congrats, {name} you are registered successfuly",text=checkstring,
                                        size_hint = (0.5,1),
                                        buttons = [close_button]  )
                self.dialog.open()
            
        create_folder(name)

        if name.isalpha() and self.flag == 1:
            
            cap = cv2.VideoCapture(0)
            count = 101
        
            while True:
                ret, frame = cap.read()
                if face_extractor(frame) is not None:
                    count -= 1
                    
                    face = cv2.resize(face_extractor(frame),(200,200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    
                    file_name_path = name + '/' + name + str(count) + '.jpg'
                    
                    cv2.imwrite(file_name_path,face)

                    cv2.putText(face,f"wait for {count}sec.",(6,55),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
                    cv2.imshow('Face cropper', face)

                else:
                    #print("Face not found")
                    pass

                if cv2.waitKey(1) == 13 or count==1:
                    break

            cap.release()
            cv2.destroyAllWindows()
        
    def clear(self):
        self.ids.gettext.text = " "   
        
    def close(self,obj):
        self.dialog.dismiss()
       
class TrainPhotoScreen(Screen):
    
    def TrainData(self):
        #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        #image_dir = os.path.join(BASE_DIR, "faces")
        image_dir = "faces"
        #print(image_dir)

        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        recognizer=cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
            #print(root)
            for file in files:
                #print(file)
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", "-").capitalize()
                    #creating labels array
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    
                    id_ = label_ids[label]
                    #print(label,path)
                    pil_image = Image.open(path).convert("L")
                    #size = (550, 550)
                    #final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(pil_image, "uint8")
                    
                    faces = face_cascade.detectMultiScale(image_array,1.4,5)
                    for(x,y,w,h) in faces:
                        roi = image_array[y:y+h,x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)
        #print(x_train)
        #print(y_labels)
        with open('TrainingData/facelabels.pickle', 'wb') as f:
            pickle.dump(label_ids, f)           

        recognizer.train(x_train, np.array(y_labels))
        recognizer.write("TrainingData/trainner.yml")

        #print("model trianed")
        checkstring = "Your device is ready to detect faces!"
        close_button = MDFlatButton(text="Close",on_release=self.close)
        self.dialog = MDDialog(title="Model Trained!!",text=checkstring,
                                size_hint = (0.5,1),
                                buttons = [close_button]  )
        self.dialog.open()

    def close(self, obj):
        self.dialog.dismiss()

class MainApp(MDApp):
    
    def on_start(self):
        #self.theme_cls.primary_palette = 'BlueGray'
        #self.theme_cls.primary_hue = "500"
        #self.theme_cls.theme_style = "Light"
        self.connection = sqlite3.connect("myapp.db")
        self.cursor = self.connection.cursor()

        #https://kivymd.readthedocs.io/en/latest/themes/theming/
        self.cursor.execute("""SELECT * FROM theme ;""")
        current_theme = self.cursor.fetchall()
        self.connection.commit()
        
        # print(current_theme)
        # [('Purple', 'BlueGray', 'Light')]
        
        if len(current_theme) == 0:
            self.theme_cls.primary_palette = 'BlueGray'
            self.theme_cls.accent_palette = "BlueGray"
            self.theme_cls.primary_hue = "500"
            self.theme_cls.theme_style = "Light"
        else:
            self.theme_cls.primary_palette = current_theme[0][0]
            self.theme_cls.accent_palette = current_theme[0][1]
            self.theme_cls.primary_hue = "500"
            self.theme_cls.theme_style = current_theme[0][2]
        
        # Profile photo manager
        self.profile_photo_grid = self.root.ids.profilephoto_screen.ids.profilephotogrid
        self.profile_photo_manager = ProfilePhotoManager()
             
     

    def show_theme_picker(self):
        theme_dialog = MDThemePicker()
        theme_dialog.open()
        
    def print_theme(self):
        self.cursor.execute("""SELECT * FROM theme ;""")
        current_theme = self.cursor.fetchall()
        self.connection.commit()
    
        if len(current_theme) == 0:
            self.cursor.execute("""INSERT INTO theme (primary_palette, accent_palette, theme_style) VALUES (?, ?, ?);""", (self.theme_cls.primary_palette, self.theme_cls.accent_palette, self.theme_cls.theme_style))
            self.connection.commit()
        else:
            self.cursor.execute("""UPDATE theme SET primary_palette = ? , accent_palette = ? , theme_style = ? ;""", (self.theme_cls.primary_palette, self.theme_cls.accent_palette, self.theme_cls.theme_style))
            self.connection.commit()
    
    def change_profile_source(self, path):
        self.root.ids.profile.source = "C:"+path # For computer
        #self.root.ids.profile.source = path # For mobile phone
        self.root.ids.nav_drawer.toggle_nav_drawer()
        with open("profile_source.txt", "w") as f:
            #f.write(path) # For mobile phone
            f.write("C:"+path) # For computer
            
    if os.path.isfile("profile_source.txt"):
        with open("profile_source.txt", "r") as f:
            some_path = f.read()
            if len(some_path) > 0:
                img_source_path = some_path
            else:
                img_source_path = "profile.png"
    else:
        img_source_path = "profile.png"

    def change_screen(self, screen_name, direction='forward', mode = ""):
        # Get the screen manager from the kv file.
        screen_manager = self.root.ids.screen_manager
 
        if direction == "None":
            screen_manager.transition = NoTransition()
            screen_manager.current = screen_name
            return
 
        screen_manager.transition = CardTransition(direction=direction, mode=mode)
        screen_manager.current = screen_name
 
        if screen_name == "home_screen":
            #print(screen_name)
            self.root.ids.titlename.title = "Face Detection"
        
        if screen_name == "profilephoto_screen":
            self.root.ids.titlename.title = "Change Profile"
            #print("Screen name is ", screen_name)
            if platform == 'android':
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
        
        if screen_name == "takephoto_screen":
            self.root.ids.titlename.title = "Take Photos"
            #print("Screen name is ", screen_name)

        if screen_name == "trainphoto_screen":
            self.root.ids.titlename.title = "Train Photos"
            #print("Screen name is ", screen_name)
                    

MainApp().run()
