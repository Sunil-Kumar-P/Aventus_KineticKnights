import webbrowser
url = "https://www.gameflare.com/online-game/hill-climbing/"
webbrowser.open(url, new=0)
import cv2
import mediapipe as mp
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
l=1
r=1
def check(a,b):
    global l
    global r
    if len(b)>1:
        if(r and a[13].x+a[0].x<1 and b[0].classification[0].index==0 and a[9].y<=a[12].y):
            pyautogui.keyDown("right")
            # pyautogui.keyUp("left")
            # print("released right")
            r=0
        elif (not r and a[13].x+a[0].x<1 and b[0].classification[0].index==0 and a[9].y>a[12].y):
            pyautogui.keyUp("right")
            # print("released right")
            r=1
        if(l and a[13].x+a[0].x>=1 and b[1].classification[0].index==1 and a[9].y<=a[12].y):
            pyautogui.keyDown("left")
            # pyautogui.keyUp("right")
            # print("pressed left")
            l=0
            # r=1
        elif (not l and a[13].x+a[0].x>=1 and b[1].classification[0].index==1 and a[9].y>a[12].y):
            pyautogui.keyUp("left")
            # print("released left")
            l=1
        if(r and a[13].x+a[0].x<1 and b[1].classification[0].index==0 and a[9].y<=a[12].y):
            pyautogui.keyDown("right")
            # pyautogui.keyUp("left")
            # print("pressed right")
            r=0
        elif (not r and a[13].x+a[0].x<1 and b[1].classification[0].index==0 and a[9].y>a[12].y):
            pyautogui.keyUp("right")
            # print("released right")
            r=1
        if(l and a[13].x+a[0].x>=1 and b[0].classification[0].index==1 and a[9].y<=a[12].y):
            pyautogui.keyDown("left")
            # pyautogui.keyUp("right")
            # print("pressed left")
            l=0
            # r=1
        elif (not l and a[13].x+a[0].x>=1 and b[0].classification[0].index==1 and a[9].y>a[12].y):
            # print("released left")
            pyautogui.keyUp("left")
            l=1
    else:
        if(r and a[13].x+a[0].x<1 and b[0].classification[0].index==0 and a[9].y<=a[12].y):
            pyautogui.keyDown("right")
            # pyautogui.keyUp("left")
            # print("pressed right")
            r=0
        elif (not r and a[13].x+a[0].x<1 and b[0].classification[0].index==0 and a[9].y>a[12].y):
            pyautogui.keyUp("right")
            # print("released right")
            r=1
        if(l and a[13].x+a[0].x>=1 and b[0].classification[0].index==1 and a[9].y<=a[12].y):
            pyautogui.keyDown("left")
            # pyautogui.keyUp("right")
            # print("pressed left")
            l=0
            # r=1
        elif (not l and a[13].x+a[0].x>=1 and b[0].classification[0].index==1 and a[9].y>a[12].y):
            # print("released left")
            pyautogui.keyUp("left")
            l=1
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
        check(hand_landmarks.landmark,results.multi_handedness) 
        x,y=(hand_landmarks.landmark[5].x+hand_landmarks.landmark[17].x+hand_landmarks.landmark[0].x)/3,(hand_landmarks.landmark[5].y+hand_landmarks.landmark[17].y+hand_landmarks.landmark[0].y)/3
        image = cv2.circle(image, (int(x*b),int(y*a)), radius=2, color=(0, 255, 0), thickness=-1)
    for i in range(a):
        image[i][b//2]=[0,255,0]
    cv2.imshow('Hill Climbing, Play with your gestures!', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
