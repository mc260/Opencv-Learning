import os
import numpy as np
import cv2

img=cv2.imread('team.jpg')
print(img.shape)

cropimg=img[19:515,47:516]
print(cropimg.shape)

cv2.imshow('frame2',cropimg)
cv2.imshow('frame',img)
cv2.waitKey(0)

cv2.destroyAllWindows()
