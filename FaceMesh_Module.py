
import cv2
import mediapipe as mp
import numpy as np
import time
 
# FACEMESH_CONTOURS , FACE_CONNECTIONS , FACE_CONTOURS , FACE_TESSELATION

class CreateFaceMesh():
    
    def __init__(self, mode = False , maxface = 1 , minDetect =0.5 , minTrack = 0.5 ):

        self.mode = mode
        self.maxface = maxface
        self.minDetect = minDetect
        self.minTrack = minTrack


        self.npdraw = mp.solutions.drawing_utils
        self.npfacemesh=mp.solutions.face_mesh
        self.face_mesh = self.npfacemesh.FaceMesh(self.mode , self.maxface ,min_detection_confidence = 0.5 , min_tracking_confidence = 0.5)
        self.drawspc = self.npdraw.DrawingSpec(color=(0,0,255),thickness=4, circle_radius=1)

    def findFaceMesh(self,img,draw = True):
        
        self.imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgrgb)

        faces = []

        if self.results.multi_face_landmarks :
            for facelms in self.results.multi_face_landmarks :
                if draw:
                    self.npdraw.draw_landmarks(img ,facelms ,self.npfacemesh.FACEMESH_CONTOURS ,self.drawspc,self.drawspc)
                
                face = []

                for id1,lm in enumerate(facelms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x , y = int(lm.x*iw) , int(lm.y*ih)
                    # print(str(id1))
                    # cv2.putText(img,str(id1),(x,y),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)
                    # print(id1,x,y)
                    face.append([x,y])
                faces.append(face)    

        return img,faces



def main():

    cap = cv2.VideoCapture("1.mp4")
    ptime=0

    dector = CreateFaceMesh()

    while True:

        succes , img = cap.read()
        img , faces = dector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces[0])
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img,f' FPS : {int(fps)} ',(20,78),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        resize = cv2.resize(img,(800,650))
        cv2.imshow("Image",resize)
        cv2.waitKey(1)

if __name__ == "__main__":
    # print(" Hello World")
    main()