import os
import numpy as np
import cv2

img=cv2.imread('team.jpg')
img_copy=cv2.resize(img,(320,320))    #裁剪一半像素

print(img.shape)
print(img_copy.shape)

cv2.imshow('img',img)
cv2.imshow('img_copy',img_copy)
cv2.waitKey(0)

cv2.destroyAllWindows()