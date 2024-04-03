import mediapipe as mp
import cv2

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
styleDot = draw.DrawingSpec(color=(252, 191, 0), thickness=3)
styleLine = draw.DrawingSpec(color=(255, 255, 255), thickness=2)
# 设置hand参数
mp_hand = hands.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.7, max_num_hands=2)
width = 1280
height = 720
cap = cv2.VideoCapture(4)#我的摄像头是第5个
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
is_status = 0
thread_num = 0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    img.flags.writeable = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hand.process(imgRGB)
    if result.multi_hand_landmarks:
        is_status = 1
        for hs in result.multi_hand_landmarks:
            draw.draw_landmarks(img, hs, hands.HAND_CONNECTIONS, styleDot, styleLine)
            y7 = int(hs.landmark[7].y * height)
            y12 = int(hs.landmark[12].y * height)
            y16 = int(hs.landmark[16].y * height)
            if int(hs.landmark[17].x * width) > int(hs.landmark[5].x * width):
                x4 = int(hs.landmark[4].x * width)
                x2 = int(hs.landmark[2].x * width)
                if x2 - x4 > 50:
                    is_status = 0
                    thread_num = 0
                elif y7 < y12 and y7 < y16 and is_status == 1:
                    y8 = int(hs.landmark[8].y * height)
                    x8 = int(hs.landmark[8].x * width)
    else:
        is_status = 0
        thread_num = 0
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
mp_hand.close()
cap.release()
cv2.destroyAllWindows()
