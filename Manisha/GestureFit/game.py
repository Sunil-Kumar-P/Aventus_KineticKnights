import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import datetime
import math
import os
import random
from scipy import spatial

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculateAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians =  np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0]- b[0])
    angle = np.abs ( radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360 - angle
        
    return angle

def Average(lst):
    return sum(lst) / len(lst)

def dif_compare(x,y):
    average = []
    for i,j in zip(range(len(list(x))),range(len(list(y)))):
        result = 1 - spatial.distance.cosine(list(x[i].values()),list(y[j].values()))
        average.append(result)
    score = math.sqrt(2*(1-round(Average(average),2)))
    return score

def diff_compare_angle(x,y):
    new_x = []
    for i,j in zip(range(len(x)),range(len(y))):
        z = np.abs(x[i] - y[j])/((x[i]+ y[j])/2)
        new_x.append(z)
    return Average(new_x)

def compare_pose(image,angle_point,angle_user, angle_target):
    angle_user = np.array(angle_user)
    angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    cv2.rectangle(image,(0,0), (370,40), (255,255,255), -1)
    cv2.putText(image, str("Score:"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
    height, width, _ = image.shape   
    
    if angle_user[0] < (angle_target[0] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5) 
        
    if angle_user[0] > (angle_target[0] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[0][0]*width), int(angle_point[0][1]*height)),30,(0,0,255),5)

        
    if angle_user[1] < (angle_target[1] -15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)
        
    if angle_user[1] >(angle_target[1] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[1][0]*width), int(angle_point[1][1]*height)),30,(0,0,255),5)

        
    if angle_user[2] < (angle_target[2] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)

    if angle_user[2] > (angle_target[2] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[2][0]*width), int(angle_point[2][1]*height)),30,(0,0,255),5)

    if angle_user[3] < (angle_target[3] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)

    if angle_user[3] > (angle_target[3] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[3][0]*width), int(angle_point[3][1]*height)),30,(0,0,255),5)

    if angle_user[4] < (angle_target[4] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)

    if angle_user[4] > (angle_target[4] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[4][0]*width), int(angle_point[4][1]*height)),30,(0,0,255),5)

    if angle_user[5] < (angle_target[5] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)
        

    if angle_user[5] > (angle_target[5] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[5][0]*width), int(angle_point[5][1]*height)),30,(0,0,255),5)

    if angle_user[6] < (angle_target[6] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)
        

    if angle_user[6] > (angle_target[6] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[6][0]*width), int(angle_point[6][1]*height)),30,(0,0,255),5)


    if angle_user[7] < (angle_target[7] - 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)

    if angle_user[7] > (angle_target[7] + 15):
        stage = stage + 1
        cv2.circle(image,(int(angle_point[7][0]*width), int(angle_point[7][1]*height)),30,(0,0,255),5)
    
    if stage!=0:
        cv2.putText(image, str("FIGHTING!"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        pass
    else:
        cv2.putText(image, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        

def extractKeypoint(path):
    IMAGE_FILES = [path] 
    stage = None
    joint_list_video = pd.DataFrame([])
    count = 0
    
    with mp_pose.Pose(min_detection_confidence =0.5, min_tracking_confidence = 0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)   
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_h, image_w, _ = image.shape
            w,h=image.shape[1],image.shape[0]
 
            try:

                landmarks = results.pose_landmarks.landmark
                
   

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                joints = []
                joint_list = pd.DataFrame([])

                for i,data_point in zip(range(len(landmarks)),landmarks):
                    joints = pd.DataFrame({
                                           'frame': count,
                                           'id': i,
                                           'x': data_point.x,
                                           'y': data_point.y,
                                           'z': data_point.z,
                                           'vis': data_point.visibility
                                            },index = [0])
                    joint_list = joint_list.append(joints, ignore_index = True)
                
               
               
   
            except:
                pass
            joint_list_video = joint_list_video._append(joint_list, ignore_index = True)
            cv2.rectangle(image,(0,0), (100,255), (255,255,255), -1)
            cv2.putText(image, str(1),tuple(np.multiply(right_elbow,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(2),tuple(np.multiply(left_elbow,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(3),tuple(np.multiply(right_shoulder,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(4),tuple(np.multiply(left_shoulder,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(5),tuple(np.multiply(right_hip,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(6),tuple(np.multiply(left_hip,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(7),tuple(np.multiply(right_knee,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
            cv2.putText(image, str(8),tuple(np.multiply(left_knee,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)

            keypoints = []
            for point in landmarks:
                keypoints.append({
                    'X': point.x,
                    'Y': point.y,
                    'Z': point.z,
                    })

            angle = []
            angle_list = pd.DataFrame([])
            angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
            angle.append(int(angle1))
            angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
            angle.append(int(angle2))
            angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
            angle.append(int(angle3))
            angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
            angle.append(int(angle4))
            angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
            angle.append(int(angle5))
            angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
            angle.append(int(angle6))
            angle7 = calculateAngle(right_hip, right_knee, right_ankle)
            angle.append(int(angle7))
            angle8 = calculateAngle(left_hip, left_knee, left_ankle)
            angle.append(int(angle8))

            cv2.putText(image, 'ID', (10,14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            cv2.putText(image, str(1), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(2), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(3), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(4), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(5), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(6), (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(7), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(8), (10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)

            cv2.putText(image, 'Angle', (40,12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle1)), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle2)), (40,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle3)), (40,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle4)), (40,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle5)), (40,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle6)), (40,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle7)), (40,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle8)), (40,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)


            #Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 2),
                                     mp_drawing.DrawingSpec(color = (0,255,0),thickness = 4, circle_radius = 2)

                                     )          

            #cv2.imshow('MediaPipe Feed',image)
            print(angle)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break
            
        # cv2.destroyAllWindows()
    return landmarks, keypoints, angle, image

def imgchk():
    # folderPath = "Pose"
    # myList = os.listdir(folderPath)
    # print(myList)
    # rannum = random.randint(0,4)
    # imgpath=myList[rannum]
    # return imgpath
    img1="p3.jpg"
    img2="p4.jpg"
    img_index=0
    swap_time=0
    if time.time() - swap_time > 5:
        img_index = (img_index + 1) % 2
        swap_time = time.time()
    imgpath = img1 if img_index == 0 else img2
    # return imgpath

def mainfun(l):
    
    # folderPath = r"C:\Users\vinee\Downloads\Games Manager Basic\Manisha\GestureFit\Pose"
    # myList = os.listdir(folderPath)
    # print(myList)
    # rannum = random.randint(0,2)
    # print(rannum)
    # imgpath=myList[rannum]
    #name of image must go here
    
    # path = "C:/Users/vinee/Downloads/Games Manager Basic/Manisha/GestureFit/Pose"
    # path+="/"+imgpath
    # print(path)
    # path=f"Pose/p3.jpg" 
    cap = cv2.VideoCapture(0)
    climit=6
    pose=mp_pose.Pose(min_detection_confidence =0.5, min_tracking_confidence = 0.5)
    cap.isOpened()
    for i in l:
        print(i)
        path="C:/Users/vinee/Downloads/Games Manager Basic/Manisha/GestureFit/"+i
        x = extractKeypoint(path)
        pTime = 0
        waitTime = 10
        start_time = time.time()
        prevTime = time.time()
        newTime = time.time()
        psc=0

        scale_percent = 60
        width= int(x[3].shape[1] * scale_percent / 100)
        height = int(x[3].shape[0] * scale_percent / 100)
        dim= (width,height)
        resized = cv2.resize(x[3], dim, interpolation = cv2.INTER_AREA)

        angle_target = x[2]
        point_target = x[1]
        if 1:
            # z=
            countdow=20
            while 1:
                ret,frame= cap.read()
                frame=cv2.flip(frame,1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                w,h=1280,720
                image = cv2.resize(image, (1280, 720))

        
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility*100, 2)]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility*100, 2)]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility*100, 2)]
                    
                    angle_point = []
                    
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    angle_point.append(right_elbow)
                    
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    angle_point.append(left_elbow)
                    
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    angle_point.append(right_shoulder)
                    
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    angle_point.append(left_shoulder)
                    
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    angle_point.append(right_hip)
                    
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    angle_point.append(left_hip)
                    
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    angle_point.append(right_knee)
                    
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle_point.append(left_knee)
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    
                    keypoints = []
                    for point in landmarks:
                        keypoints.append({
                            'X': point.x,
                            'Y': point.y,
                            'Z': point.z,
                            })
                    
                    # print("person",keypoints)
                    p_score = dif_compare(keypoints, point_target)      
                    
                    angle = []
                    
                    angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                    angle.append(int(angle1))
                    angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                    angle.append(int(angle2))
                    angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                    angle.append(int(angle3))
                    angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                    angle.append(int(angle4))
                    angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                    angle.append(int(angle5))
                    angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                    angle.append(int(angle6))
                    angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                    angle.append(int(angle7))
                    angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                    angle.append(int(angle8))
                    
                    cv2.putText(image, str(1),tuple(np.multiply(right_elbow,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(2),tuple(np.multiply(left_elbow,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(3),tuple(np.multiply(right_shoulder,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(4),tuple(np.multiply(left_shoulder,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(5),tuple(np.multiply(right_hip,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(6),tuple(np.multiply(left_hip,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(7),tuple(np.multiply(right_knee,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    cv2.putText(image, str(8),tuple(np.multiply(left_knee,[w, h]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0,0,0], 2 , cv2.LINE_AA)
                    
                
                    x, y = left_ankle
                    x,y=x*1280,y*720
                    elapsed_time = int(time.time() - start_time)
                    countdown = max(0, climit - int(elapsed_time))
                    countdown_str = f"Countdown: {countdown}"
                    cv2.putText(image, countdown_str, (400, 80), cv2.FONT_HERSHEY_PLAIN, 3,(255,0, 0), 3)

                    
                    if countdown == 0:
                        print(int(max((1-p_score)*100,(1-a_score)*100))) #score value present here    
                        break
                    # if 100 < x < 1180 and 100 < y < 680:
                    # print(angle_target)  
                    compare_pose(image, angle_point,angle, angle_target)
                    a_score = diff_compare_angle(angle,angle_target)
                    
                    if (p_score >= a_score):
                        cv2.putText(image, str(int((1 - a_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

                    else:
                        cv2.putText(image, str(int((1 - p_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

                    # if((1-a_score)*100>95 or (1-p_score)*100>95):
                        # psc+=1
                    # else:
                        # psc=0

            
                    # cv2.putText(image, str(psc), (1030,70), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA) 
                except AttributeError as e:
                    pass      
        
                except Exception as e:
                    print(e)
                    break

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 4),
                                        mp_drawing.DrawingSpec(color = (0,255,0),thickness = 3, circle_radius = 3)
                                        )

                

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, f'FPS: {int(fps)}', (800, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
                # cv2.rectangle(image, (10, 10), (1260, 680), (0, 0, 255), 2)
                h, w, c = resized.shape 
                image[50:h+50, 10:w+10] = resized
                newTime = time.time()


                cv2.imshow('Gesture Fit',image)


                if cv2.waitKey(2) & 0xFF == ord('q'):
                    return
            
    cap.release()
    cv2.destroyAllWindows()

mainfun(["Pose/pose1.jpg","Pose/p3.jpg","Pose/p4.jpg","Pose/pose1.jpg","Pose/p3.jpg","Pose/p4.jpg","Pose/pose1.jpg","Pose/p3.jpg","Pose/p4.jpg"])
