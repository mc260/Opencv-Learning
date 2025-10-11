import cv2
import numpy as np

img=cv2.imread('brid.png')
img=cv2.resize(img,(800,600))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

img_edge = cv2.Canny(img,100,200)           #canny api 实现边缘绘制
img_dli=cv2.dilate(img_edge,kernel)
img_erode=cv2.erode(img_edge,kernel)


cv2.imshow('image',img)
cv2.imshow('edge',img_edge)
cv2.imshow('dli',img_dli)
cv2.imshow('erode',img_erode)


cv2.waitKey(0)