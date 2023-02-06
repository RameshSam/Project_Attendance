import time
import cv2
import mediapipe as mp

class BodyPoseDetector():

    def __init__(self,mode=False , complexity = 1, landmark = True ,dectcon = 0.5 , trackcon = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.landmark = landmark
        self.dectcon = dectcon
        self.trackcon = trackcon

        self.npPose = mp.solutions.pose
        self.Pose = self.npPose.Pose(self.mode , self.complexity, self.landmark , min_detection_confidence= 0.5 , min_tracking_confidence = 0.5 )
        self.npdraw = mp.solutions.drawing_utils
        self.color = self.npdraw.DrawingSpec(color=(0,255,0))
        self.color1 = self.npdraw.DrawingSpec(color=(0,0,0),thickness=5)
        
    def findbodypose(self,img,draw = True):

        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.Pose.process(imgrgb)
        if self.results.pose_landmarks :
            if draw :
                self.npdraw.draw_landmarks(img,self.results.pose_landmarks,self.npPose.POSE_CONNECTIONS,self.color,self.color1)
        return img

    def getposition(self,img,draw=True):
        lmlist = []
        if self.results.pose_landmarks :

            for id1, lm in enumerate(self.results.pose_landmarks.landmark):
                h , w , c =  img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                lmlist.append([id1,cx,cy])
                if draw:
                    # cv2.putText(img,str(id1),(cx,cy),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),4)
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)

        return lmlist

def main():

    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture("4.mp4")
    dector = BodyPoseDetector()

    while True :
        success , img = cap.read()
        img = dector.findbodypose(img)
        lmlist = dector.getposition(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        cv2.putText(img,f" FPS : {int(fps)}",(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        ptime =  ctime

        resize = cv2.resize(img,(1000,600))
        cv2.imshow("Image",resize)
        cv2.waitKey(1)

if __name__ == "__main__" :
    main()