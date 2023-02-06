# FACEMESH_CONTOURS , FACE_CONNECTIONS , FACE_CONTOURS , FACE_TESSELATION

import cv2
import mediapipe as mp
import numpy as np
import time

class FaceDetector():

    def __init__(self,mindectcon = 0.5):
        self.mindectcon = mindectcon

        self.npdraw = mp.solutions.drawing_utils
        self.npfacedetect = mp.solutions.face_detection
        self.face_detection = self.npfacedetect.FaceDetection(0.75)

    def findface(self,img,draw=True):

        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =self.face_detection.process(imgrgb)
        bbox = []
        if self.results.detections :
            for id2 ,detection in enumerate(self.results.detections):
                
                obj = detection.location_data.relative_bounding_box
                ih , iw , ic = img.shape
                cx = int(obj.xmin * iw) , int(obj.ymin * ih) , int(obj.width * iw) , int(obj.height * ih)              
                bbox.append([id2,cx,detection.score])
                # cv2.rectangle(img,cx,(255,0,255),2)
                img = self.drawFancy(img,bbox)
                cv2.putText(img,f' {int(detection.score[0]*100)}% ',(cx[0],cx[1]),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),6)
        return img , bbox

    def drawFancy(self,img,bboxl,l=30,t=70,rt=1):
        x , y, w, h = bboxl
        x1 , y1 = x*w , y*h
        cv2.rectangle(img,bboxl,(255,0,255),rt)
    
        cv2.line(img,(x,y),(x+l , y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)

        cv2.line(img,(x1,y),(x1-l , y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)

        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)

        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)

        return img


def main():

    cap = cv2.VideoCapture("5.mp4")
    ptime=0
    detector = FaceDetector()

    while True:

        succes , img = cap.read()
        img , lmlist = detector.findface(img)
        print(lmlist)
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img,f' FPS : {int(fps)} ',(20,78),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        resize = cv2.resize(img,(800,650))
        cv2.imshow("Image",resize)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()