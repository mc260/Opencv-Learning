import numpy as np
import cv2
from PIL import Image
from HSVfinding import get_limits

video=cv2.VideoCapture('任务视频.mp4')

fingcolar=[33,52,167]

'''
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerLimit,upperLimit = get_limits(color=fingcolar)
mask=cv2.inRange(hsv_img,lowerLimit,upperLimit)
mask_pil=Image.fromarray(mask)
bbox=mask_pil.getbbox()
print(bbox)
if bbox is not None:
     x1,y1,x2,y2=bbox
     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),10)
'''
ret = True
while ret:
    ret, frame = video.read()
    hsv_video=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=fingcolar)
    mask = cv2.inRange(hsv_video, lowerLimit, upperLimit)
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 10)
    cv2.imshow('frame', frame)           #播放每一帧
    if cv2.waitKey(10) & 0xFF == ord('q'):        #若触发键盘Q建 退出视频
        break

video.release()
cv2.destroyAllWindows()