import cv2
import numpy as np
import time
from geometry import *
from threading import Thread, Lock

class CamInput(Thread):
    def __init__(self):
        super().__init__()
        self.rects = []
        self.colours = [] # [(lower, upper)]
        self.video = cv2.VideoCapture(0)
        self.lockRects = Lock()
        self.video.set(3,200)
        self.video.set(4,200)
        self.running = True

    def set_colours(self, colours):
        self.colours = colours

    def get_rects(self):
        self.lockRects.acquire()
        rects = [i for i in self.rects]
        self.lockRects.release()
        # mode 0 = normal, 1 = can only be shot through, 2 = no collison can not be shot thorugh  
        # debug, returns one of each Block type
        # return [(Rect(Vec2(50, 0), Vec2(50, 50)), 0), (Rect(Vec2(50, 100), Vec2(50, 50)), 1), (Rect(Vec2(50, 150), Vec2(50, 50)), 2)]
        return rects

    # ru methode for thread
    def run(self):
        # blue
        lower = np.array([87, 155, 92])
        upper = np.array([120, 255, 255])
        self.colours.append((lower, upper, 0))

        # green
        lower = np.array([42, 49, 56])
        upper = np.array([95, 223, 255])
        self.colours.append((lower, upper, 1))

         # red
        lower = np.array([0, 166, 110])
        upper = np.array([20, 255, 185])
        self.colours.append((lower, upper, 2))

        while self.running:
            success, img = self.video.read()
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            rects = []
            for l, u, m in self.colours:
                mask = cv2.inRange(image, l, u)

                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) != 0:
                    for contour in contours:
                        if cv2.contourArea(contour) > 1000:
                            x, y, w, h = cv2.boundingRect(contour)
                            # cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 225), 3)
                            rects.append((Rect(Vec2(x, y), Vec2(w, h)), m)) # (rect, type)

            self.lockRects.acquire() # lock
            self.rects = [i for i in rects]
            self.lockRects.release() # unlock

            # slow down to keep the camera from flickering
            time.sleep(0.3)

    # synchronous causes way to much lag
    def single_check(self):
        # green
        lower = np.array([40, 70, 70])
        upper = np.array([77, 170, 192])
        self.colours.append((lower, upper, 0))

        success, img = self.video.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for l, u, m in self.colours:
            mask = cv2.inRange(image, l, u)

            rects = []

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                for contour in contours:
                    if cv2.contourArea(contour) > 1500:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 225), 3)
                        rects.append((Rect(Vec2(x, y), Vec2(w, h)), m))

        return rects


if __name__ == "__main__":
    camin = CamInput()
    camin.run()
