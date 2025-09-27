import numpy as np
import cv2
import os

img=cv2.imread("ICBK.png")
img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)      #BGR色彩空间转化为RBG色彩空间
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #BGR色彩空间转化为灰度图
img_HLS=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)      #BGR色彩空间转化为HLS色彩空间     #HLS常用于色彩识别

#cv2.imshow('img',img)
cv2.imshow('img_RGB',img_RGB)
cv2.imshow('img_gray',img_gray)
cv2.imshow('img_HLS',img_HLS)

cv2.waitKey(0)
cv2.destroyAllWindows()
