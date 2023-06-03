import webbrowser
url = "https://poki.com/en/g/temple-run-2-jungle-fall"
webbrowser.open(url, new=0)
import cv2
import mediapipe as mp
import pyautogui
from time import time
import sys
import random
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
#temple run
#https://poki.com/en/g/temple-run-2-jungle-fall
l=[1,1,1]
u=[1,1,1]
p=0
lr=ud=1
def check(a):
    global p,l,u,k,lr,ud
    if p and (a[8].y+a[12].y+a[16].y+a[20].y)>(a[5].y+a[9].y+a[13].y+a[17].y):
        p=0
        pyautogui.typewrite(["space"])
    elif not p and (a[8].y+a[12].y+a[16].y+a[20].y)<(a[5].y+a[9].y+a[13].y+a[17].y):
        p=1
    if l[1] and 1/3<(a[5].x+a[17].x+a[0].x)/3<=2/3:
        if lr==2:
            pyautogui.typewrite(["left"])
        elif lr==0:
            pyautogui.typewrite(["right"])
        l=[1,0,1]
        lr=1
    elif l[2] and (a[5].x+a[17].x+a[0].x)/3<=1/3:
        pyautogui.typewrite(["right"])
        l=[1,1,0]
        lr=2
    elif l[0] and 2/3<(a[5].x+a[17].x+a[0].x)/3:
        pyautogui.typewrite(["left"])
        l=[0,1,1]
        lr=0
    if u[1] and 1/3<(a[5].y+a[17].y+a[0].y)/3<=2/3:
        u=[1,0,1]
        ud=1
    elif u[2] and 2/3<(a[5].y+a[17].y+a[0].y)/3:
        pyautogui.typewrite(["down"])
        u=[1,1,0]
        ud=2
    elif u[0] and (a[5].y+a[17].y+a[0].y)/3<=1/3:
        pyautogui.typewrite(["up"])
        u=[0,1,1]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
s, image = cap.read()
a,b,c=np.shape(image)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,max_num_hands=2
    ,static_image_mode=False
) as hands:
  kt=time()
  s=0
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        check(hand_landmarks.landmark)
        x,y=(hand_landmarks.landmark[5].x+hand_landmarks.landmark[17].x+hand_landmarks.landmark[0].x)/3,(hand_landmarks.landmark[5].y+hand_landmarks.landmark[17].y+hand_landmarks.landmark[0].y)/3
        image = cv2.circle(image, (int(x*b),int(y*a)), radius=2, color=(0, 255, 0), thickness=-1)
    for i in range(a):
        image[i][b//3]=image[i][2*b//3]=[0,255,0]
    for i in range(b):
        image[a//3][i]=image[2*a//3][i]=[0,255,0]
    s+=random.randint(500,800)/10000
    image=cv2.flip(image, 1)
    cv2.putText(image, f"Calories={s}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Temple Run 2, Play with your gestures!', image)
    kt=time()
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
