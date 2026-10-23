import cv2
import numpy as np
from pynput.keyboard import Controller
import time

keyboard = Controller()

# Define Key Layout [Label, Top-Left (x,y), Bottom-Right (x,y)]
keys = [
    ["Q", (50, 50), (110, 110)],
    ["W", (120, 50), (180, 110)],
    ["E", (190, 50), (250, 110)],
    ["SPACE", (50, 130), (250, 190)]
]

# HSV color range for skin tone detection (Adjust for your lighting)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

cap = cv2.VideoCapture(0)
last_pressed = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask skin region and blur to reduce noise
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Draw Keyboard Layout
    for key, (x1, y1), (x2, y2) in keys:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, key, (x1 + 15, y1 + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assumed to be hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 2000:
            # Highest Y-point in contour represents the highest fingertip
            topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
            fx, fy = topmost
            
            # Draw pointer on fingertip
            cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

            # Check key collisions
            for key, (x1, y1), (x2, y2) in keys:
                if x1 < fx < x2 and y1 < fy < y2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
                    
                    # Dwell-time trigger: hover over key for > 0.5s to type
                    if time.time() - last_pressed > 0.5:
                        if key == "SPACE":
                            keyboard.press(' ')
                            keyboard.release(' ')
                        else:
                            keyboard.press(key.lower())
                            keyboard.release(key.lower())
                        last_pressed = time.time()

    cv2.imshow("OpenCV Air Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()