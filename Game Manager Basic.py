from tkinter import *
from tkinter import messagebox
import subprocess
import os
import tkinter.font as tkFont
import re
import webbrowser
import threading
from PIL import Image, ImageTk
from tkinter import Label
from time import time
import cv2

def detect_camera_indexes():
    camera_indexes = []
    for index in range(-6,100):
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                break
            else:
                camera_indexes.append(index)
                cap.release()
            index += 1
    return camera_indexes

# Detect camera indexes
# camera_indexes = detect_camera_indexes()
# print("Detected camera indexes:", camera_indexes)
from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
import cv2
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("./model.json", "./model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        return fr

w=1280
h=720
menubg = "./Images/menubg.png"
temple = "./Images/templerun.png"              
hv=0
wv=0
av=0
it=0
ft=0
lt=0
gamemet={"Temp Run 2":4,"Hill Climb":4.5,"Angry Birds":4.5,"Subway Surfers":6,"Gesture Fit":7,"Tennis":0}
cgame=0
recommend=""
def run(w):
    
    w.loop()

def init():
    global window
    global recommend
    window=Tk()
    obj=page(window)
    typ=0
    obj.addgame("Temp Run 2 ","Louise Patra/Temple Run 2/",typ=typ)
    obj.addgame("Hill Climb ","Louise Patra/Hill Climbing/",typ=typ)
    obj.addgame("Angry Birds ","Louise Patra/Angry Birds/",typ=typ)
    # obj.addgame("Tic Tac Toe ","Louise Patra/Tic Tac Toe/",typ=typ)
    # obj.addgame("Rock Paper Scissor ","Manisha/Rock Paper Scissor/",typ=typ)
    # obj.addgame("Fruit Ninja ","Manisha/Fruit Ninja/",typ=typ)
    # obj.addgame("Tic Tac Toe","Manisha/Tic Tac Toe/",typ=typ)
    # obj.addgame("Fox Run - Hands ","Sunil/Fox Run/Hands/",typ=typ)
    # obj.addgame("Fox Run - Pose ","Sunil/Fox Run/Pose/",typ=typ)
    # obj.addgame("Ping Pong ","Sunil/Ping Pong/",typ=typ)
    # obj.addgame("Subway Surfers ","Old/Subway Surfers/",typ=typ)
    obj.addgame("Subway Surfers ","Kamaljeet/Subway Surfers/",typ=typ)
    obj.addgame("Gesture Fit","Manisha/GestureFit/",typ=2)
    obj.addgame("Tennis","",typ=typ)
    img= (Image.open(menubg))
    resized_image= img.resize((w,h))
    new_image= ImageTk.PhotoImage(resized_image)
    imgl = Label(window,image=new_image).place(x=0, y=0, relwidth=1, relheight=1)
    window.geometry(str(w)+"x"+str(h))
    window.title("Welcome")
    hu = Label(window, text="Hello User :D",font=('Helvetica bold', 24)).place(relx=0.5, rely=0.25, anchor="center")
    # age_val_var= StringVar()
    # weight_val_var = StringVar()
    # height_val_var = StringVar()
    global age_val,weight_val,height_val
    age = Label(window, text="Age :",bg="white",fg="black",font=('Helvetica bold', 12)).place(relx=.40, rely=.35, anchor="center")
    age_val = Entry(window,bg="white",fg="black")
    age_val.place(relx=.50, rely=.35, anchor="center")
    weight = Label(window, text="Weight :",bg="white",fg="black",font=('Helvetica bold', 12)).place(relx=.40, rely=.45, anchor="center")
    weight_val = Entry(window,bg="white",fg="black")
    weight_val.place(relx=.50, rely=.45, anchor="center")
    height = Label(window, text="Height :",bg="white",fg="black",font=('Helvetica bold', 12)).place(relx=.40, rely=.55, anchor="center")
    height_val = Entry(window,bg="white",fg="black")
    height_val.place(relx=.50, rely=.55, anchor="center")
    recommend ="hello"
    button = Button(window,width=20,bg="#B9CD5C",font=('Helvetica bold', 24),text="GET Started?", command=obj.run ).place(relx=0.5, rely=0.65, anchor="center")
    # print(age_val_var.get(),weight_val_var.get(),height_val_var.get())
    window.bind_all('<Escape>', lambda x:window.destroy())
    window.protocol("WM_DELETE_WINDOW",lambda :window.destroy())
    # window.resizable(False,False)
    window.mainloop()
    
class page:
    def __init__(self,prevwin):
        self.window=Toplevel(prevwin)
        self.img= (Image.open(menubg))
        resized_image= self.img.resize((w,h),Image.BILINEAR)
        new_image= ImageTk.PhotoImage(resized_image)
        imgl = Label(self.window,image=new_image).place(x=0, y=0, relwidth=1, relheight=1)
        self.window.geometry(str(w)+"x"+str(h))
        self.window.title("GAMES")
        self.i=0
        Label(self.window, text="Select one game to play :",font=('Helvetica bold', 24)).place(relx=0.5, rely=0.08, anchor="center")
        self.window.bind_all('<Escape>', lambda x:self.quit())
        self.window.protocol("WM_DELETE_WINDOW",lambda :self.quit())
        # self.window.resizable(False,False)
        self.window.withdraw()
        self.prevwin=prevwin
    def run(self):
        global age_val,height_val,weight_val
        print( age_val.get(),height_val.get(),weight_val.get())
        cap = cv2.VideoCapture(0)
        l=[]
        while True:
            # Read a frame from the camera
            ret, fr = cap.read()

            # Check if the frame was successfully captured
            if not ret:
                break
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]

                roi = cv2.resize(fc, (48, 48))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                l.append(pred)
                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            # Display the frame
            cv2.imshow('Frame', fr)
            if len(l)>100:
                break
            # Wait for the 'q' key to be pressed to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture object and close the OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        counted_elements = {}
        for element in l:
            if element in counted_elements:
                counted_elements[element] += 1
            else:
                counted_elements[element] = 1

        # Find the element with the highest occurrence
        max_occurrence_element = max(counted_elements, key=counted_elements.get)
        global recommend
        # ["Angry", "Disgust",
        #             "Fear", "Happy",
        #             "Neutral", "Sad",
        #             "Surprise"]
        if max_occurrence_element == "Angry":
            recommend = "You seem to be feeling angry! Why not channel that energy into playing Angry Birds? It's a great way to let off steam and have some fun. Enjoy!"
        elif max_occurrence_element == "Disgust":
            recommend = "I can see that you're feeling disgusted. Sorry to see that. How about playing Temple Run 2? It's an exciting game that can help you take your mind off things and refresh your mood."
        elif max_occurrence_element == "Fear":
            recommend = "It looks like you're feeling scared. Don't worry, we all feel fear sometimes. Why not distract yourself and have some fun by playing Subway Surfers? It can help you feel more adventurous and overcome your fears!"
        elif max_occurrence_element == "Happy":
            recommend = "You radiate happiness! Keep that positive energy going and play Fruit Ninja. It's a joyful game that will match your mood perfectly. Have a blast!"
        elif max_occurrence_element == "Sad":
            recommend = "I can see that you're feeling sad. Remember, it's okay to feel down sometimes. Let's do something uplifting together. How about trying FitMotionX? It combines fitness and fun to help you feel better physically and emotionally."
        elif max_occurrence_element == "Surprise" or max_occurrence_element == "Neutral":
            recommend = "You seem surprised or neutral. It's always intriguing to encounter the unexpected. How about playing any game you like? Choose one that resonates with your current mood and enjoy the adventure!"
            
        messagebox.showinfo("Recommendation",f"{recommend}")
        self.window.deiconify()
        self.prevwin.withdraw()
    # def addage(self):
    #     button = Button(window,width=20,bg="#B9CD5C",font=('Helvetica bold', 24),text="GET Started?", command=obj.run).place(relx=0.5, rely=0.55, anchor="center")
    def quit(self):
        self.prevwin.destroy()
    def addgame(self,name,folder,typ=0):
        global cgame
        cgame=name
        self.i+=1
        game(self.prevwin,self.window,name,self.i,folder,typ)
        
class game:
    def __init__(self,main,prevwin,name,n,folder,typ=0):
        self.name=name
        self.prevwin=prevwin
        self.main=main
        self.type=typ
        self.file=folder+"game.py"
        self.pid=None
        self.folder=folder
        self.about=self.readfile("about.txt")
        self.reqtext=self.readfile("req.txt")
        
        Button(self.prevwin,width=28,font=('Helvetica bold', 15),bd=5,bg="green",fg="white",activebackground="#B9CD5C", padx=5, pady=5,text=str(n)+". "+name, command=self.start).place(relx=0.5, rely=0.04+0.07*(n+1), anchor="center")
    def start(self):
        global lt
        it=time()
        

        self.window=Toplevel(self.main)
        
        
        
        self.window.geometry(str(w)+"x"+str(h))
        self.window.title(self.name)
        # Label(self.window, image=Image.open(temple)).place(x=0, y=0, relwidth=1, relheight=1)
        self.window.bind_all('<Escape>', lambda x:self.quit())
        self.window.protocol("WM_DELETE_WINDOW",lambda :self.quit())
        
        self.img= (Image.open(temple))
        resized_image = self.img.resize((w, h),Image.BICUBIC)
        new_image = ImageTk.PhotoImage(resized_image)
        imgl = Label(self.window, image=new_image).place(x=0, y=0, relwidth=1, relheight=1)
        Button(self.window,activebackground="#12218A", bg="#1D37E8",activeforeground="#fff", text="Back",command=self.back,width=5, bd=5, padx=5, pady=5).place(relx=.1,rely=.1,anchor='center')
        Label(self.window, text="Click to Launch the game",font=('Helvetica bold', 24)).place(relx=0.5, rely=0.1, anchor="center")
        Button(self.window, text="Launch",command=self.launch,width=17, bd=5, padx=5, pady=5).place(relx=.4,rely=.2,anchor='center')
        
        if self.reqtext:
            # Label(self.window, image=Image.open(temple)).place(x=0, y=0, relwidth=1, relheight=1)
            Button(self.window, text="Instal Requirements",command=self.req,width=17, bd=5, padx=5, pady=5).place(relx=.6,rely=.2,anchor='center')
        if self.about:
            # Label(self.window, image=Image.open(temple)).place(x=0, y=0, relwidth=1, relheight=1)
            Label(self.window, text="About / Instructions",font=('Helvetica bold', 24)).place(relx=0.5, rely=0.35, anchor="center")
            frame = Frame(self.window, width=250, height=250)
            frame.place(relx=0.25, rely=0.4, relwidth=0.5, relheight=1.0)
            text = Text(frame, wrap=WORD)
            scrollbar = Scrollbar(frame)
            scrollbar.place(relx=1.0, rely=0.0, relheight=0.5, anchor=NE)
            text.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=0.5)
            text.config(yscrollcommand=scrollbar.set, font=('Roboto', 12, 'bold'))
            if "*[" in self.about and "]*" in self.about:
                for i in re.split("(\*\[.*]\*)",self.about):
                    if i[0]=="*":
                        k=re.split("\((.*)\).*\((.*)\)",i)
                        # print(k[1],k[2])
                        text.insert(END, k[1],("link",k[2]))
                        text.tag_bind(k[2], "<Enter>", self.open_link)
                        text.tag_bind(k[2], "<Leave>", self.leave_link)
                        text.tag_configure(k[2], foreground="blue", underline=False)
                    else:
                        # print(i)
                        text.insert(END, i)
            else:text.insert(END, self.about)
            text.config(state=DISABLED)
        
        # self.window.resizable(False,False)img = Image.open(temple)
        

        self.window.withdraw()
        self.run()
    def open_link(self,event):
        link_url = event.widget.tag_names(CURRENT)[1]
        event.widget.tag_configure(link_url, foreground="blue", underline=True)
        event.widget.tag_bind(link_url, "<Button-1>", lambda event: webbrowser.open(link_url))

    def leave_link(self,event):
        link_url = event.widget.tag_names(CURRENT)[1]
        event.widget.tag_configure(link_url, underline=False)
        
    def readfile(self,file):
        try:
            with open(self.folder+file,"r") as file:
                return(file.read())
        except:
            return(None)
    def launch(self):
        if self.type==0:
            proc = subprocess.Popen(['python', self.file,str(gamemet[cgame]),str(weight_val)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,start_new_session=True)
            self.pid = proc.pid
            # while True:
            #     output = proc.stdout.readline()
            #     error = proc.stderr.readline()
            #     if output == '' and error == '' and proc.poll() is not None:
            #         break
            #     if output:
            #         print(output.strip())
            #     if error:
            #         print(error.strip())
            print(f"Running script in the background with PID {self.pid}")

        elif self.type==1:
            output_file = 'output.txt'
            # print(f"python3 {self.file} ")
            os.system(f"python3 './{self.file}'> {output_file} 2>&1")

            # Read the output from the file
            with open(output_file, 'r') as f:
                output = f.read()

            # Print the output
            print(output)
        elif self.type==2:
            print(self.file)
            output_file = 'output.txt'
            # print(f"python3 {self.file} ")
            os.system(f"python -u {os.path.normpath(self.file)}> {output_file} 2>&1")

            # Read the output from the file
            with open(output_file, 'r') as f:
                output = f.read()
            print(output)
    def run(self):
        self.window.deiconify()
        self.prevwin.withdraw()
    def quit(self):
        self.main.destroy()
        if self.pid!=None:
            os.kill(self.pid, 15)
            # os.kill(pid, 9)
    def back(self):
        ft=time()-it
        print(ft)
        self.window.withdraw()
        self.prevwin.deiconify()
        if self.pid!=None:
            os.kill(self.pid, 15)
            # os.kill(pid, 9)
    def req(self):
        a=Label(self.window, text="Even though its looking like its hang,\nBut its installing the requirements.",font=('Helvetica bold',10),fg="red")
        a.place(relx=0.8, rely=0.2, anchor="center")
        self.window.update()
        input_lines = self.reqtext.strip().split('\n')
        input_text = '\n'.join(input_lines[:]) 
        if self.type==0:
            proc = subprocess.Popen(
            self.reqtext.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
            
            
            proc.stdin.write(input_text)
            proc.stdin.close()

            while True:
                output = proc.stdout.readline()
                error = proc.stderr.readline()
                if output == '' and error == '' and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
                if error:
                    print(error.strip())

            # Get the PID of the process
            pid = proc.pid
            print("PID:", pid)
        elif self.type==1:
            command = input_text
            output_file = 'output.txt'
            os.system(f"{command} > {output_file} 2>&1")

            # Read the output from the file
            with open(output_file, 'r') as f:
                output = f.read()

            # Print the output
            print(output)
        a.place_forget()
        
        
init()
