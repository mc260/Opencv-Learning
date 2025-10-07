import cv2
import numpy as np

img=cv2.imread('brid.png')
img=cv2.resize(img,(800,600))

img_edge = cv2.Canny(img,100,200)           #canny api 实现边缘绘制

cv2.imshow('image',img)
cv2.imshow('edge',img_edge)

cv2.waitKey(0)