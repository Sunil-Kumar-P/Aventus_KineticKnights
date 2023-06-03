import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

boundaries = {"left":0.6,"right":0.4,"bottom":0.6,"top":0.4}
states = ("left","right","up","down","center","space")
x,y = 0.5,0.5
cap = cv2.VideoCapture(0)
cur_state = "idle"
new_state = "idle"
new_state1 = "idle"
tutorial = True
flag = False

def move(key):
    print(key)
    pyautogui.press(key)
def transition():
    global cur_state
    global new_state1
    if new_state!= cur_state:
        if cur_state == states[4] and (new_state == states[1] or new_state == states[0]):
            move(new_state)
        elif new_state == states[4] and (cur_state == states[1] or cur_state == states[0]):
            if cur_state == states[1]:
                move(states[0])
            else:
                move(states[1])
        elif new_state == states[3] or new_state == states[2]:
            move(new_state)
    if new_state1 == states[5]:
            move(new_state1)
    new_state1 = 'idle'
    cur_state = new_state

def check():
    global cur_state
    global new_state
    global new_state1
    if y < boundaries["top"]:
            new_state = states[2]
    elif y > boundaries["bottom"]:
            new_state = states[3]
    elif x >boundaries["left"]:
        new_state = states[0]
    elif x <boundaries["right"]:
            new_state = states[1]
    else:
        new_state = "center"
    if zy > boundaries["top"] and zy < boundaries["bottom"] and zx <boundaries["left"] and zx > boundaries["right"] and wy > boundaries["top"] and wy < boundaries["bottom"] and wx <boundaries["left"] and wx > boundaries["right"]:
            new_state1 = states[5]
    
    transition()
def start(y):
    global flag
    if(y<boundaries["top"]):
        flag = True
        pyautogui.press('s')
        print("start")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    cap.set(3,640)
    cap.set(4,480)
    if not success:
      print("Ignoring empty camera frame.")

      continue
    h, w, c = image.shape
    if (tutorial):
        cv2.rectangle(image, (0, int(boundaries["top"]*h)), (int(w*boundaries["right"]), int(boundaries["bottom"]*h)), (0, 255, 0), 1)
        cv2.rectangle(image, (w, int(boundaries["top"]*h)), (int(w*boundaries["left"]), int(boundaries["bottom"]*h)), (0, 255, 0), 1)
        cv2.rectangle(image, (0, 0), (w, int(boundaries["top"]*h)), (0, 255, 0), 1)
        cv2.rectangle(image, (0, int(boundaries["bottom"]*h)), (w, h), (0, 255, 0), 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.rectangle(image, (int(x * w) - 1, int(y * h) - 1), (int(x * w) + 1, int(y * h) + 1), (0, 255, 0), 2)
        x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x + results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y + results.pose_landmarks.landmark[
            mp_pose.PoseLandmark(24).value].y) / 2
        zy = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        zx = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
        wx = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
        wy = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        
        if flag:
            check()
        if not flag:
            start(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y)
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()



