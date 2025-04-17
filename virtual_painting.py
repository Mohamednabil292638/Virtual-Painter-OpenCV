import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Create a canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
draw_color = (255, 0, 255)  # Default: Purple
brush_thickness = 10
xp, yp = 0, 0  # Previous point for drawing

# Get landmark positions
def get_position(handLms):
    lm_list = []
    for lm in handLms.landmark:
        h, w = 480, 640
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append((cx, cy))
    return lm_list

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = get_position(handLms)
            if lm_list:
                x1, y1 = lm_list[8]   # Index fingertip
                x2, y2 = lm_list[12]  # Middle fingertip

                # Selection Mode – 2 fingers up
                if abs(y2 - y1) < 40:
                    cv2.rectangle(img, (x1 - 50, y1 - 50), (x1 + 50, y1 + 50), draw_color, cv2.FILLED)
                    if x1 < 160:
                        draw_color = (255, 0, 255)  # Purple
                    elif x1 < 320:
                        draw_color = (0, 255, 0)    # Green
                    elif x1 < 480:
                        draw_color = (0, 0, 255)    # Red
                    else:
                        draw_color = (0, 255, 255)  # Yellow

                # Drawing Mode – only index finger
                else:
                    cv2.circle(img, (x1, y1), brush_thickness, draw_color, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    xp, yp = x1, y1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Merge canvas with webcam
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv_canvas)
    img = cv2.bitwise_or(img, canvas)

    # Color boxes
    cv2.rectangle(img, (0, 0), (160, 50), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, 'Purple', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.rectangle(img, (160, 0), (320, 50), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, 'Green', (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.rectangle(img, (320, 0), (480, 50), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, 'Red', (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.rectangle(img, (480, 0), (640, 50), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, 'Yellow', (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Display the frame
    cv2.imshow("Virtual Painter", img)

    # Key handling
    key = cv2.waitKey(1)
    if key == ord('c'):  # Clear canvas
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        xp, yp = 0, 0
    elif key == ord('s'):  # Save image
        cv2.imwrite("my_painting.png", canvas)
        print("🎉 Drawing saved as my_painting.png")
    elif key == 27:  # ESC to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()