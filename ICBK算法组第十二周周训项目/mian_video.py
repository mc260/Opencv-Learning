import cv2
from kalmanfilter import KalmanFilter
from PIL import Image                                           #导入Pillow库
from HSVfinding import get_limits

fingcolar = [21,177,152]

kf = KalmanFilter()
cap = cv2.VideoCapture(0)

predicted = (0, 0)

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=fingcolar)
    mask = cv2.inRange(hsvimg, lowerLimit, upperLimit)
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()
    print(bbox)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cx=int((x1+x2)/2)
        cy=int((y1+y2)/2)
        predicted = kf.predict(cx, cy)

        cv2.circle(frame,(cx,cy),15,(0,0,255),-1)
        cv2.circle(frame,(predicted[0],predicted[1]),15,(0,255,0),5)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()