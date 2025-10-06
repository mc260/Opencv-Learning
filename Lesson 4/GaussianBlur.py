import cv2
import numpy as np
img = cv2.imread('brid.png')

ksize = 10                                                  #定义内核的大小   确定模糊邻域的大小
img_blur = cv2.blur(img,(ksize,ksize))                #经典模糊

cv2.imshow('image', img)
cv2.imshow('blur', img_blur)
cv2.waitKey(0)
