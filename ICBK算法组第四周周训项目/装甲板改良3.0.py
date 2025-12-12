import cv2
import numpy as np
from PIL import Image, ImageDraw  # 仅在需要PIL特有操作时使用，此处主要用OpenCV


def run():
    # 读取视频
    capture = cv2.VideoCapture("../ICBK算法组第三周周训项目/text.mp4")
    #capture = cv2.VideoCapture(1)

    # 获取帧率
    rate = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / rate)
    stop = False

    while not stop:
        # 读取一帧
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imshow("ori frame", frame)


        # 复制图像用于处理，避免修改原始数据
        image = frame.copy()
        if image is None or image.size == 0:  # 安全检查
            print("空帧")
            return -1

        # 转为灰度图
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binaryImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        # 形态学操作去除噪声和小物体
        kernel = np.ones((1, 1), np.uint8)
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)  # 开运算去噪声
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)  # 闭运算填充空洞

        # 查找轮廓
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result_image = frame.copy()
        lightBars = []    # 存储检测到的灯条




        # 遍历轮廓筛选灯条
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i]) # 计算当前轮廓的面积
            if area < 100 or area > 900:
                continue

            # 最小面积矩形
            rect = cv2.minAreaRect(contours[i])
            size = rect[1]
            width = min(size[0], size[1])
            height = max(size[0], size[1])
            aspectRatio = height / width

            if 3.0 < aspectRatio < 6.0 and 100 < area < 900:
                lightBars.append(rect)      # 符合条件的加入灯条列表

        # 收集所有灯条顶点    并绘制外接矩形
        all_points = []
        for lightBar in lightBars:
            # 获取旋转矩形的四个顶点坐标
            vertices = cv2.boxPoints(lightBar).astype(np.int32)
            for j in range(4):
                all_points.append(vertices[j])  # 添加到总点集


        # 如果检测到灯条，计算整体的边界框
        if all_points:

            # 计算能包围所有灯条顶点的最小直立矩形
            bounding_rect = cv2.boundingRect(np.array(all_points))

            # 在结果图像上绘制装甲板区域（黄色矩形）
            cv2.rectangle(result_image, (bounding_rect[0], bounding_rect[1]),
                          (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                          (0, 255, 255), 3)

        # 显示窗口
        cv2.imshow("result", result_image)
        cv2.imshow("mask", binaryImage)

        # 按帧率等待按键，按键则停止
        if cv2.waitKey(delay) >= 0:
            stop = True

    capture.release()
    cv2.destroyAllWindows()
    return 0

run()