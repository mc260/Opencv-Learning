import cv2
import numpy as np
from PIL import Image
from HSVfinding import get_limits
import math


def detect_armor_combined():
    # 读取视频
    video = cv2.VideoCapture('text.mp4')

    # 定义颜色阈值（根据实际情况调整）
    red_color = [33, 52, 167]
    blue_color = [247, 245, 66]

    # 选择要检测的颜色
    target_color = blue_color  # 可根据敌方颜色切换

    ret = True
    while ret:
        ret, frame = video.read()
        if not ret:
            break

        # 颜色阈值的检测
        color_detection_frame = frame.copy()
        hsv_frame = cv2.cvtColor(color_detection_frame, cv2.COLOR_BGR2HSV)
        lowerLimit, upperLimit = get_limits(color=target_color)
        color_mask = cv2.inRange(hsv_frame, lowerLimit, upperLimit)

        # 对颜色掩膜进行形态学操作
        kernel = np.ones((3,3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # 基于形状的灯条检测
        shape_detection_frame = frame.copy()
        gray = cv2.cvtColor(shape_detection_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 形态学操作
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        light_bars = []
        all_points = []

        # 筛选灯条
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150 or area > 900:
                continue

            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            min_side = min(width, height)
            max_side = max(width, height)

            aspect_ratio = max_side / min_side if min_side > 0 else 0

            # 灯条特征：长宽比在1-15之间 面积350-4000
            if 1.0 < aspect_ratio < 15.0 and 5 < area < 4000:
                light_bars.append(rect)
                vertices = cv2.boxPoints(rect).astype(np.int32)
                all_points.extend(vertices)

        # 结合两种方法的结果
        result_frame = frame.copy()

        # 显示颜色检测结果
        color_bbox = Image.fromarray(color_mask).getbbox()
        if color_bbox is not None:
            x1, y1, x2, y2 = color_bbox
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框表示颜色检测
            cv2.putText(result_frame, "color", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 改进：使用灯条顶点构建装甲板
        if len(light_bars) >= 2:
            # 尝试配对灯条
            light_bars_sorted = sorted(light_bars, key=lambda x: x[0][0])

            for i in range(len(light_bars_sorted) - 1):
                left_light = light_bars_sorted[i]
                right_light = light_bars_sorted[i + 1]

                # 计算两个灯条之间的距离
                left_center = np.array(left_light[0])
                right_center = np.array(right_light[0])
                distance = np.linalg.norm(left_center - right_center)

                # 计算灯条长度平均值
                left_length = max(left_light[1])
                right_length = max(right_light[1])
                mean_length = (left_length + right_length) / 2

                # 配对条件：距离适中，长度相近
                if (distance < mean_length * 4 and
                        distance > mean_length * 1.5 and
                        abs(left_length - right_length) < mean_length * 0.5):
                    # 获取两个灯条的顶点
                    left_vertices = cv2.boxPoints(left_light).astype(np.int32)
                    right_vertices = cv2.boxPoints(right_light).astype(np.int32)

                    # 对灯条顶点进行排序：找到最靠近对方的两个顶点
                    # 左灯条：找到x坐标最大的两个点（右侧点）
                    left_vertices_sorted = sorted(left_vertices, key=lambda x: x[0], reverse=True)
                    left_right_points = left_vertices_sorted[:2]

                    # 右灯条：找到x坐标最小的两个点（左侧点）
                    right_vertices_sorted = sorted(right_vertices, key=lambda x: x[0])
                    right_left_points = right_vertices_sorted[:2]

                    # 根据y坐标对点进行配对
                    left_right_points_sorted = sorted(left_right_points, key=lambda x: x[1])
                    right_left_points_sorted = sorted(right_left_points, key=lambda x: x[1])

                    # 构建装甲板的四个顶点
                    # 顺序：左上，右上，右下，左下
                    armor_points = np.array([
                        left_right_points_sorted[0],  # 左上（左灯条右上点）
                        right_left_points_sorted[0],  # 右上（右灯条左上点）
                        right_left_points_sorted[1],  # 右下（右灯条左下点）
                        left_right_points_sorted[1]  # 左下（左灯条右下点）
                    ], dtype=np.int32)

                    # 绘制装甲板边框
                    cv2.polylines(result_frame, [armor_points], True, (0, 255, 255), 3)

                    # 标记装甲板
                    center_x = (left_center[0] + right_center[0]) / 2
                    center_y = (left_center[1] + right_center[1]) / 2
                    cv2.putText(result_frame, "Armor",
                                (int(center_x) - 20, int(center_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 备选方案：如果灯条配对失败，使用原有的矩形检测
        elif all_points:
            bounding_rect = cv2.boundingRect(np.array(all_points))
            cv2.rectangle(result_frame,
                          (bounding_rect[0], bounding_rect[1]),
                          (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]),
                          (0, 255, 255), 3)  # 黄色框表示形状检测
            cv2.putText(result_frame, "shape",
                        (bounding_rect[0], bounding_rect[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 在灯条区域内部绘制装甲板识别框
            armor_width = bounding_rect[2] * 0.6
            armor_height = bounding_rect[3] * 0.8
            armor_x = bounding_rect[0] + (bounding_rect[2] - armor_width) / 2
            armor_y = bounding_rect[1] + (bounding_rect[3] - armor_height) / 2

            cv2.rectangle(result_frame,
                          (int(armor_x), int(armor_y)),
                          (int(armor_x + armor_width), int(armor_y + armor_height)),
                          (255, 0, 0), 3)  # 蓝色框表示最终的装甲板区域
            cv2.putText(result_frame, "Armor",
                        (int(armor_x), int(armor_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)




        # 显示各阶段结果
        cv2.imshow('color mask', color_mask)
        cv2.imshow('binary mask', binary)
        cv2.imshow('result', result_frame)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# 运行检测
detect_armor_combined()