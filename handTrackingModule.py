import cv2
import mediapipe as mp
import time


class handDector():

    def __init__(self,mode=False,maxhands=2,detconfidence=0.5,trackconfidence=0.5):

        self.mode = mode
        self.maxhands = maxhands
        self.detconfidence = detconfidence
        self.trackconfidence = trackconfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode , self.maxhands , min_detection_confidence = 0.5 , min_tracking_confidence = 0.5)
        self.mpdraw = mp.solutions.drawing_utils
    
    def detect_hand(self,img,draw=True):

        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)

        if self.results.multi_hand_landmarks :
            for handlms in self.results.multi_hand_landmarks :
                if draw:
                    self.mpdraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        # print(len(self.results.multi_hand_landmarks))

        return img
                
    def path(self,img,handno=0, draw = True):
        
        lmlist = []
        if self.results.multi_hand_landmarks :
            myhand = self.results.multi_hand_landmarks[handno]
            for id1 ,lm in enumerate(myhand.landmark):
                # print(id,lm)
                h , w, c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                # print(id1,cx,cy)
                cv2.putText(img,str(id1),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,0),2)
                lmlist.append([id1,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmlist

def main():

    ptime = 0
    ctime = 0

    cap = cv2.VideoCapture("2.mp4")
    # cap = cv2.VideoCapture(1)
    detector = handDector()

    while True:
        success , img = cap.read()
        img = detector.detect_hand(img)
        lm_list = detector.path(img,draw=False)
        # if len(lm_list) != 0:
        #     print(lm_list[4])

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(img,f' FPS : {int(fps)} ',(20,78),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),5)
        resize = cv2.resize(img,(850,650))
        cv2.imshow("Image",resize)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()