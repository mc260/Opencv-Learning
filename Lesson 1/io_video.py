import cv2
import numpy as np

video= cv2.VideoCapture("text vedio.mp4")          #捕获视频函数 cv2.VideoCapture
ret = True
while ret:
    ret,frame = video.read()                      #frame读取三维numpy 数组    ret若读取到帧则为Ture保持循环

#video: 是一个 cv2.VideoCapture 对象，代表视频源（摄像头或视频文件）
#read(): 方法调用，执行"读取"操作
#video.read() 返回一个包含两个元素的元组：(ret, frame)

    cv2.imshow('frame', frame)           #播放每一帧
    if cv2.waitKey(40) & 0xFF == ord('q'):        #若触发键盘Q建 退出视频

        break

#释放视频内存 结束窗口进程
video.release()
cv2.destroyAllWindows()
