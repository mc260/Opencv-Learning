#自适应阈值案例
import cv2
import numpy as np

img=cv2.imread('text.jpg')
img_copy=cv2.resize(img,(800,640))          #太大了裁一下


img_gray=cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,30)
#cv2.GaussianBlur(thresh,(11,11),3)

#cv2.imshow('original', img_copy)
cv2.imshow('gray', img_gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)