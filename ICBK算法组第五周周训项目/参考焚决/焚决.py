import cv2
import numpy as np
import math


class LightDescriptor:
    def __init__(self, light_rect=None):
        if light_rect is not None:
            self.width = light_rect[1][0]
            self.length = light_rect[1][1]
            self.center = light_rect[0]
            self.angle = light_rect[2]
            self.area = light_rect[1][0] * light_rect[1][1]
        else:
            self.width = 0
            self.length = 0
            self.center = (0, 0)
            self.angle = 0
            self.area = 0


def main():
    # 读取视频
    video = cv2.VideoCapture('D:\Code\opencv learning\ICBK算法组第三周周训项目/text.mp4')

    # 变量定义
    frame = None
    channels = [None, None, None]
    binary = None
    gaussian = None
    dilatee = None

    # 创建形态学操作核
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 通道分离
        channels = cv2.split(frame)

        # 二值化 (使用蓝色通道，因为原C++代码中用的是channels[0])
        _, binary = cv2.threshold(channels[0], 220, 255, cv2.THRESH_BINARY)

        # 高斯模糊
        gaussian = cv2.GaussianBlur(binary, (5, 5), 0)

        # 膨胀
        dilatee = cv2.dilate(gaussian, element)

        # 轮廓检测
        contours, hierarchy = cv2.findContours(dilatee, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        light_infos = []  # 灯条信息列表

        # 筛选灯条
        for contour in contours:
            # 求轮廓面积
            area = cv2.contourArea(contour)

            # 去除较小轮廓 & fitEllipse的限制条件
            if area < 5 or len(contour) <= 1:
                continue

            # 用椭圆拟合区域得到外接矩形
            if len(contour) >= 5:  # fitEllipse需要至少5个点
                light_rect = cv2.fitEllipse(contour)

                # 长宽比限制
                width, height = light_rect[1]
                if width / height > 4:
                    continue

                light_infos.append(LightDescriptor(light_rect))

        # 二重循环多条件匹配灯条
        for i in range(len(light_infos)):
            for j in range(i + 1, len(light_infos)):
                left_light = light_infos[i]
                right_light = light_infos[j]

                # 计算各种参数
                angle_gap = abs(left_light.angle - right_light.angle)
                len_gap_ratio = abs(left_light.length - right_light.length) / max(left_light.length, right_light.length)

                dis = math.sqrt((left_light.center[0] - right_light.center[0]) ** 2 +
                                (left_light.center[1] - right_light.center[1]) ** 2)

                mean_len = (left_light.length + right_light.length) / 2
                lengap_ratio = abs(left_light.length - right_light.length) / mean_len

                y_gap = abs(left_light.center[1] - right_light.center[1])
                y_gap_ratio = y_gap / mean_len

                x_gap = abs(left_light.center[0] - right_light.center[0])
                x_gap_ratio = x_gap / mean_len

                ratio = dis / mean_len

                # 匹配不通过的条件
                if (angle_gap > 15 or
                        len_gap_ratio > 1.0 or
                        lengap_ratio > 0.8 or
                        y_gap_ratio > 1.5 or
                        x_gap_ratio > 2.2 or
                        x_gap_ratio < 0.8 or
                        ratio > 3 or
                        ratio < 0.8):
                    continue

                # 绘制装甲板矩形
                center = ((left_light.center[0] + right_light.center[0]) / 2,
                          (left_light.center[1] + right_light.center[1]) / 2)

                # 创建旋转矩形
                rect = ((center[0], center[1]), (dis, mean_len), (left_light.angle + right_light.angle) / 2)

                # 获取旋转矩形的四个顶点
                vertices = cv2.boxPoints(rect).astype(np.int32)

                # 绘制装甲板边框
                for k in range(4):
                    cv2.line(frame,
                             tuple(vertices[k]),
                             tuple(vertices[(k + 1) % 4]),
                             (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow("video", cv2.WINDOW_FREERATIO)
        cv2.imshow("video", frame)
        cv2.imshow("mask", dilatee)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()