import cv2
import numpy as np
img = cv2.imread('brid.png')

ksize = 7                                                                   #定义卷积核内核的大小   确定模糊邻域的大小   （最好是正奇数)
img_blur = cv2.blur(img,(ksize,ksize))                                #经典模糊
img_gauss = cv2.GaussianBlur(img,(ksize,ksize),3)             #高斯模糊
img_medianblur = cv2.medianBlur(img,ksize)                                 #中值模糊

cv2.imshow('image', img)
cv2.imshow('blur', img_blur)
cv2.imshow('gauss', img_gauss)
cv2.imshow('medianblur', img_medianblur)
cv2.waitKey(0)
