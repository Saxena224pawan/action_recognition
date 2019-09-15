import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import face_recognition
import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import pandas as pd
import time
import imageio
import threading
from tkinter import *
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
           raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = 640
        self.height = 480
    
        
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None) 
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
class App:
    def __init__(self,window1,kfe,kfn,vid_loc):
        self.window1=window1
        self.window = tk.Frame(window1)
        self.photo1 = ""
        self.kfe= kfe
        self.kfn= kfn
        self.data=[]
        self.video_location= vid_loc
        self.df = pd.DataFrame()
        self.b1 = tk.Button(self.window1, text="OK", command=self.callback)
        self.b1.pack(side="bottom")
        self.listbox = Listbox(self.window1)
        self.listbox.pack( expand =1 )
        self.thread = None
        self.stopEvent = None
        self.vid=""
        self.delay=10

        # open video source

        # Create a canvas that can fit the above video source size
        self.canvas1 = tk.Canvas(self.window1, width=640, height=480)
        self.canvas1.pack(padx=5, pady=10, side="left")
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.video_capture = cv2.VideoCapture(self.video_location)
        
        self.path = os.getcwd()
        self.variables = []
        length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            os.mkdir('peoples')
        except:
            pass
        for frames in range(length):
            for y in self.kfn:
                try:
                    os.mkdir("peoples/frame_%d/%s" %(frames,y))
                except:
                    pass
            try:
                os.mkdir("peoples/frame_%d/Unknown" %(frames))
            except:
                pass
            
        self.x=0
        
        known_face_encodings, known_face_names = self.kfe,self.kfn
        face_locations = []
        face_encodings = []
        frame_number=0
        
        self.out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (640,480))
        if self.video_capture.isOpened():
            for frame_number in range(length):
                ret, frame = self.video_capture.read(frame_number)
                if ret:
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_frame = frame[:, :, ::-1]
                    # Find all the faces and face enqcodings in the frame of video
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance =0.5)
                        name = "Unknown"
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] :
                            name = known_face_names[best_match_index]
                            print("%s detected in frame number %d"%(name,frame_number))
                            if best_match_index not in self.variables:
                                self.listbox.insert(END , str(name + " : Detected "))
                                self.variables.append(best_match_index)
                            self.data.append([name,np.int(frame_number),np.int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))])
                        if name =="Unknown":
                            fimage =rgb_frame[top:bottom,left:right]
                            fimage=cv2.cvtColor(fimage, cv2.COLOR_BGR2RGB)
                            #cv2.imwrite("gdrive/My Drive/video/videos/%s/%d.jpg" %(name ,x),fimage)
                            cv2.imwrite("peoples/frame_%d/%s/%d.jpg" %(frame_number,name,self.x),fimage)
                            self.x+=1
                        else:
                            fimage =rgb_frame[top:bottom,left:right]
                            fimage=cv2.cvtColor(fimage, cv2.COLOR_BGR2RGB)
                            #cv2.imwrite("gdrive/My Drive/video/videos/%s/%d.jpg" %(name ,x),fimage)
                            cv2.imwrite("peoples/frame_%d/%s/%d.jpg" %(frame_number,name,self.x),fimage)
                            self.x+=1
                            
                        
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.out.write(frame)                        
        if self.out.isOpened():
            self.out.release()
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.vid1 = MyVideoCapture('output.mp4')
        self.update()
    def update(self):
        ret, frame = self.vid1.get_frame() 
        if ret:
            self.photo1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas1.create_image(0, 0, image = self.photo1, anchor = tk.NW) 
        self.window.after(self.delay, self.update)
       
    def callback(self):
        self.df= pd.DataFrame(self.data,columns =["Name","Frame Number","Time"])
        max_1 = self.df.groupby('Name').max()
        min_1 = self.df.groupby('Name').min()
        self.attend = pd.concat([min_1,max_1])
        self.attend.to_csv("attend.csv")

        self.df.to_csv("data.csv")
        self.window1.destroy()        
        

class ImageOpener(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.known_face_enc=[]
        self.face_name=[]
        self.video=""
        self.geometry("400x600+40+40")
        self.configure(background='#33F9FF')
        self.create_Widgets()
        self.title("Action Recognition System")
    def load_images_from_folder(self,folder):
        for filenames in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filenames))
            if img is not None:
                encodings =face_recognition.face_encodings(img)[0]
                self.known_face_enc.append(np.asarray(encodings))
                self.face_name.append(filenames.split('.')[0])
    def create_Widgets(self):
        self.btn4 = Button(self, text='Proceed with Folder',command=self.opendirectory).place(relx=0.5, rely=0.75, anchor=S)
        self.btn5 = Button(self, text='Record Video',command=self.record_video).place(relx=0.5, rely=0.7, anchor=S)
        self.btn3 = Button(self, text='Proceed with video',command=self.Proceed).place(relx=0.5, rely=0.8, anchor=S)
        self.btn2 = Button(self, text='open video',command=self.open_video).place(relx=0.5, rely=0.85, anchor=S)
    def record_video(self):
        VideoApp(tk.Tk(),"Record Video")
    def opendirectory(self):

        filedirectory = filedialog.askdirectory(title=' folders')
        self.dir_label= tk.Label(self,text="Image Directory")
        self.dir_label.pack(pady = 20)
        self.E_folder = Entry(self)
        self.E_folder.insert(100,str(filedirectory))
        self.E_folder.config(width =90)
        self.E_folder.pack()
        self.load_images_from_folder(filedirectory)
    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename
    def open_img(self):
        self.image_label= tk.Label(self,text="Images")
        self.image_label.pack()
        x = self.openfn()
        img = Image.open(x)
        img3 = face_recognition.load_image_file(str(x))
        encodings =face_recognition.face_encodings(img3)[0]    
        self.known_face_enc.append(np.asarray(encodings))
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel =  Label(self, image=img)
        panel.image = img
        panel.pack()
        self.E1 = Entry(self)
        self.button = Button(self, text="Get", command = self.on_button)
        self.E1.pack()
        self.button.pack()
    def open_video(self):
        self.video_label= tk.Label(self,text="Video Label")
        self.video_label.pack()
        x2= self.openfn()
        self.video=str(x2)
        self.E2 = Entry(self)
        self.E2.insert(100,str(x2))
        self.E2.pack()
    def Proceed(self):
        root1=tk.Toplevel()
        root1.geometry("780x720")
        root1.title("Action Recognition")
        App(root1,self.known_face_enc,self.face_name,self.video)
        root1.mainloop()
    def on_button(self):
        print("Name",str(self.E1.get()))
        if str(self.E1.get()) not in self.face_name:
            self.face_name.append(str(self.E1.get()))    

stopb = None
import tkinter
import cv2
class VideoApp():
    def __init__(self, window, window_title):
        self.window = window
        self.window.title = window_title
        self.ok = False
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #create videowriter
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('webcam.avi',self.fourcc,10,(640,480))
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.opencamera = tkinter.Button(window, text="Start Recording", command=self.open_camera)
        self.opencamera.pack()
        self.closecamera = tkinter.Button(window, text="Stop Recording", command=self.close_camera)
        self.closecamera.pack()
        self.delay = 10
        self.update()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.window.mainloop()

    def update(self):
        ret, frame = self.video.read()
        if self.ok:
            self.out.write(frame)
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(master= self.canvas, image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB  )))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def open_camera(self):
        self.ok = True
        print("Recording Start")
        print(self.ok)


    def close_camera(self):
        print("camera closed")
        print("Video Saved/Overwrite as webcam.avi ")
        self.ok = False
        self.video.release()
        self.out.release()

    def __del__(self):
        if self.video.isOpened():
            self.video.release()
            self.out.release()

"""

class MyVideoCapture:
    def __init__(self, kfe,kfn,video_source1):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source1)
        self.kfe=kfe
        self.kfn= kfn
        
        if not self.vid1.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        known_face_encodings, known_face_names= self.kfe,self.kfn
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        face_locations = []
        face_encodings = []
        text=[]
        frame_number = 0
        if self.vid1.isOpened():
            ret1, frame1 = self.vid1.read()
            
            if ret1:
                # Return a boolean success flag and the current frame converted to BGR
                rgb_frame = frame1[:, :, ::-1]
            
                # Find all the faces and face enqcodings in the frame of video
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
                # Loop through each face in this frame of video
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
            
                    # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]
            
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        text.append("%s detected in frame number %d"%(name,frame_number))
                        print("%s detected in frame number %d"%(name,frame_number))
    
                    # Draw a box around the face
                    cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)
            
                    # Draw a label with a name below the face
                    cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame1, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                self.delay = 15
                self.update(ret1,frame1,text)
                self.window.mainloop()
                
                # Display the resulting image
                frame_number+=1
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB,text)
            else:
                return ret1, None
        else:
            return ret1, None

    # Release the video source when the object is destroyed
    
# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recogni
# +tion library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
"""
gui = ImageOpener()
gui.mainloop()



"""print(length)
# Load some sample pictures and learn how to recognize them.
pawan_image = face_recognition.load_image_file("pawan.jpeg")

pawan_face_encoding = face_recognition.face_encodings(pawan_image)[0]

karan_image = face_recognition.load_image_file("karan.jpeg")
karan_face_encoding = face_recognition.face_encodings(karan_image)[0]

navnidhi_image = face_recognition.load_image_file("navnidhi.jpeg")
navnidhi_face_encoding = face_recognition.face_encodings(navnidhi_image)[0]
known_face_encodings = [
    pawan_face_encoding,
    karan_face_encoding,
    navnidhi_face_encoding
]
print(known_face_encodings)
# Initialize some variables
known_face_names = [
    "Pawan",
    "karan",
    "Navnidhi"
]
"""


