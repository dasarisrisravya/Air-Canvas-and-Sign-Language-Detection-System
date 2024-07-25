import cv2
import numpy as np
import mediapipe as mp
from collections import deque


# Initialize deques to store points for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes to keep track of points in different color arrays
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for image dilation
kernel = np.ones((5, 5), np.uint8)

# List of colors in BGR format
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default color index

# Set up the canvas window with color buttons
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize Mediapipe for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Main loop
mode = "menu"  # Default mode
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    x, y, c = frame.shape

    if mode == "menu":
        cv2.putText(frame, "Select Mode", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "1. Air Canvas", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "2. Sign Language", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    elif mode == "air_canvas":
        # Convert the frame from BGR to RGB
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)
            
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0
                elif 275 <= center[0] <= 370:
                    colorIndex = 1
                elif 390 <= center[0] <= 485:
                    colorIndex = 2
                elif 505 <= center[0] <= 600:
                    colorIndex = 3
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Paint", paintWindow)

    elif mode == "sign_language":
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]

                # Example gestures for demonstration
                if thumb_tip[1] < index_tip[1] < middle_tip[1] < ring_tip[1] < pinky_tip[1]:
                    sign_text = 'Yes'
                elif thumb_tip[1] < pinky_tip[1] and index_tip[1] < pinky_tip[1]:
                    sign_text = 'I Love You'
                elif thumb_tip[0] < index_tip[0] and index_tip[0] < middle_tip[0] and middle_tip[0] < ring_tip[0] and ring_tip[0] < pinky_tip[0]:
                    sign_text = 'Hello'
                elif thumb_tip[0] < index_tip[0] and thumb_tip[1] < index_tip[1]:
                    sign_text = 'Iconic Sign'
                elif thumb_tip[1] < index_tip[1] and thumb_tip[1] < pinky_tip[1] and index_tip[0] < pinky_tip[0]:
                    sign_text = 'Emotional Sign'
                elif index_tip[1] < thumb_tip[1] < middle_tip[1] and ring_tip[1] < pinky_tip[1]:
                    sign_text = 'Yes'
                elif index_tip[0] < thumb_tip[0] and middle_tip[0] < thumb_tip[0] and ring_tip[0] < thumb_tip[0] and pinky_tip[0] < thumb_tip[0]:
                    sign_text = 'No'
                elif index_tip[1] < thumb_tip[1] and middle_tip[1] < thumb_tip[1] and ring_tip[1] < thumb_tip[1] and pinky_tip[1] < thumb_tip[1]:
                    sign_text = 'Thank You'
                else:
                    sign_text = 'Unknown'

                cv2.putText(frame, sign_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = "air_canvas"
    elif key == ord('2'):
        mode = "sign_language"
    elif key == ord('m'):
        mode = "menu"

cap.release()
cv2.destroyAllWindows()