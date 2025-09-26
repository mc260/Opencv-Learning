import cv2
import numpy as np

wecam=cv2.VideoCapture(0)                     #捕获编号为0的网络摄像头

while(True):                                  #由于摄像头的图像实时更新 故永远循环播放
    ret,frame=wecam.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

#释放内存
wecam.release()
cv2.destroyAllWindows()
