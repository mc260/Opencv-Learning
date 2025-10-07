import numpy as np
import cv2
from PIL import Image
from HSVfinding import get_limits

img_1=cv2.imread('1.jpg')
img_2=cv2.imread('2.jpg')
img_3=cv2.imread('3.jpg')
img=cv2.resize(img_1,(800,600))

fingcolar=[33,52,167]
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lowerLimit,upperLimit = get_limits(color=fingcolar)
mask=cv2.inRange(hsv_img,lowerLimit,upperLimit)
mask_pil=Image.fromarray(mask)
bbox=mask_pil.getbbox()
print(bbox)
if bbox is not None:
     x1,y1,x2,y2=bbox
     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),10)

cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.waitKey(0)

