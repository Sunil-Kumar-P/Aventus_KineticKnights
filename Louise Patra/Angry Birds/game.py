import webbrowser
url = "https://kiz10.com/angry-birds/"
webbrowser.open(url, new=0)
import cv2
import mediapipe as mp
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
w,h=pyautogui.size()
l=1
def check(a):
    global l
    if not l and a[12].y>=a[9].y:
        pyautogui.mouseDown(w*(1-(a[17].x+a[5].x+a[0].x)/3), h*(a[5].y+a[17].y+a[0].y)/3)
        l=1
    elif l and a[12].y<a[9].y:
        pyautogui.mouseUp()
        l=0
    else:
        pyautogui.moveTo(w*(1-(a[17].x+a[5].x+a[0].x)/3), h*(a[5].y+a[17].y+a[0].y)/3)
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
        check(hand_landmarks.landmark)
        x,y=(hand_landmarks.landmark[5].x+hand_landmarks.landmark[17].x+hand_landmarks.landmark[0].x)/3,(hand_landmarks.landmark[5].y+hand_landmarks.landmark[17].y+hand_landmarks.landmark[0].y)/3
        image = cv2.circle(image, (int(x*b),int(y*a)), radius=2, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Angry Birds, Play with your gestures!', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()