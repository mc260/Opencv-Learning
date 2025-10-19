import numpy as np
import cv2


def get_limits(color):
    c = np.uint8([[color]])                     #准备颜色数据
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)   #把BGR颜色转换为HSV颜色

    hue = hsvC[0][0][0]                         #从HSV颜色中取出色调(H)分量

    # 特殊处理：红色的问题
    # 红色在HSV色环中很特殊，它位于色环的两端
    if hue >= 165:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    #其他颜色处理
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)


    #返回上界与下界
    return lowerLimit, upperLimit