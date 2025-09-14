import cv2
import numpy as np
import math

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))

            if b * c == 0:
                continue

            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))

            if angle <= math.pi / 2 and d > 10000:
                finger_count += 1
                cv2.circle(drawing, far, 8, [0, 255, 0], -1)

        return finger_count + 1
    return 0

cap = cv2.VideoCapture("gesture.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((300, 300, 3), np.uint8)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 5000:
            cv2.drawContours(drawing, [cv2.convexHull(max_contour)], -1, (255, 0, 0), 2)
            fingers = count_fingers(max_contour, drawing)

            if fingers == 0:
                gesture = "Fist"
            elif fingers == 1:
                gesture = "One"
            elif fingers == 2:
                gesture = "Two"
            elif fingers == 3:
                gesture = "Three"
            elif fingers == 4:
                gesture = "Four"
            elif fingers == 5:
                gesture = "Open Palm"
            else:
                gesture = "Unknown"
            cv2.putText(frame, f"Gesture: {gesture}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame[100:400, 100:400] = drawing

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()