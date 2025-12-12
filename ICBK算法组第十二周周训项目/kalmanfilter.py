import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        # 升级为6维状态：[x, y, dx, dy, ax, ay]（位置、速度、加速度）
        self.kf = cv2.KalmanFilter(6, 2)  # 6状态，2测量（仅观测x、y）

        # 1. 测量矩阵：只观测位置x、y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)

        # 2. 转移矩阵：匀加速运动模型（dt=1），修复平方符号错误
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],  # x = x + dx*dt + 0.5*ax*dt²
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],  # y = y + dy*dt + 0.5*ay*dt²
            [0, 0, 1, 0, dt, 0],  # dx = dx + ax*dt
            [0, 0, 0, 1, 0, dt],  # dy = dy + ay*dt
            [0, 0, 0, 0, 1, 0],  # ax = ax（加速度恒定）
            [0, 0, 0, 0, 0, 1]  # ay = ay
        ], np.float32)

        # 3. 过程噪声协方差：控制模型信任度（运动越规律，数值越小）
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01  # 小球运动较规律，设小一点

        # 4. 测量噪声协方差：你的测量值是精确的，数值很小
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # 5. 初始状态协方差：初始状态的不确定性（若已知初始位置，设小一点）
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1

    def predict(self, coordX, coordY):
        # 构造测量值（float32类型）
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        # 融合测量值（校正）
        self.kf.correct(measured)
        # 预测下一帧状态
        predicted = self.kf.predict()
        # 提取位置并转整数（满足cv2.circle要求）
        x = int(predicted[0][0])
        y = int(predicted[1][0])
        return (x, y)