import cv2
import numpy as np
from PIL import Image                                           #导入Pillow库
from HSVfinding import get_limits                               #用现成的轮子    找出所检测颜色在色环中的上界与下界

fingcolar = [21,177,152]                                        #在BGR色彩空间要检测的颜色

wecan=cv2.VideoCapture(0)
while (True):
    ret,frame=wecan.read()

    hsvimg=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)                #转换为HSV色彩空间

    lowerLimit, upperLimit=get_limits(color=fingcolar)          #找出所检测颜色在色环中的上界与下界

    mask=cv2.inRange(hsvimg,lowerLimit,upperLimit)              #掩码
    """
    掩码的核心思想：通过一个二值图像来控制对原始图像的操作范围。

    主要特点：
    🎯 精确控制：只处理特定区域
    🔄 非破坏性：原始图像的其他部分保持不变
    ⚡ 高效：减少不必要的计算
    🎨 灵活：可以组合多个掩码实现复杂效果
    """

    mask_pil=Image.fromarray(mask)                              #将图像从Numpy数组转换为Pillow格式

    bbox=mask_pil.getbbox()                                     #获得边界框
    print(bbox)
    if bbox is not None:
        x1,y1,x2,y2=bbox                                        #解码bbox 得到颜色边框对角两点坐标
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),-1)        #绘制边框


    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


wecan.release()
cv2.destroyAllWindows()
