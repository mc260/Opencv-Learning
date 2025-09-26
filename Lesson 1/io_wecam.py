import cv2
import numpy as np

wecam=cv2.VideoCapture(0)

while(True):
    ret,frame=wecam.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break


wecam.release()
cv2.destroyAllWindows()
