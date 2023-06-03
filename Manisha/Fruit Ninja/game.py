import cv2
import mediapipe as mp
import pyautogui
import pyautogui as pg
import time
import webbrowser
import os
import importlib
# import win32api
import ctypes

# Define the POINT structure
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# Set the cursor position to x=100, y=100


url = "https://www.crazygames.com/game/fruit-ninja"
webbrowser.open(url, new=0)
pg.FAILSAFE = False
w,h=pg.size()

def init():
    global flag
    if os.name == 'posix':
        print("Running on a Unix-like operating system")
        from Xlib import X, display
        flag = 0
        global dis,root
        dis = display.Display()
        root = dis.screen().root
    elif os.name == 'nt':
        print("Running on Windows")
        # global win32api
        # win32api = importlib.import_module('win32api')
        flag = 1
def chngcsr(x,y):
    if flag:
        # win32api.SetCursorPos((x,y))
        point = POINT(x, y)
        ctypes.windll.user32.SetCursorPos(point.x, point.y)
    else:
        root.warp_pointer(x, y)
        dis.sync()
def check(a):
    if a[12].y>=a[9].y:
        pg.mouseDown()
#     elif a[12].y<a[9].y:
#         pg.mouseUp()
#     else:
    elif a[12].y<a[9].y:
        u=int((1-a[12].x)*w)
        v=int((a[12].y)*h)
        chngcsr(u,v)
#         pg.mouseDown(button = 'left')
#         time.sleep(1000)

init()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
    
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # print(hand_landmarks.landmark
        check(hand_landmarks.landmark)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    for i in range(len(image)):
        image[i][int(len(image[0])*.875)]=[255,0,0]
        image[i][int(len(image[0])*.625)]=[255,0,0]
        image[i][int(len(image[0])*.375)]=[255,0,0]
        image[i][int(len(image[0])*.125)]=[255,0,0]
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()