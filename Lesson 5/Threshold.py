import cv2
import numpy as np

img=cv2.imread('text.jpg')


img_copy=cv2.resize(img,(640,640))

cv2.imshow('original', img_copy)
cv2.waitKey(0)