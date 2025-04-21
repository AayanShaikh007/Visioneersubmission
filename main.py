import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
# USE ESC KEY TO EXIT PROGRAM
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

# Sensitivity control - Use this if the cursor isnt moving fast enough 
SENSITIVITY = 8000
    
prev_nose_x, prev_nose_y = None, None

SAFE_MARGIN = 20
screen_w, screen_h = pyautogui.size()

while True:
    if keyboard.is_pressed("."):
        screen_width, screen_height = pyautogui.size()
        center_x = screen_width // 2
        center_y = screen_height // 2
        pyautogui.moveTo(center_x, center_y)
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    overlay = frame.copy()
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        nose = landmarks[1]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        cv2.circle(overlay, (nose_x, nose_y), 5, (0, 255, 0), -1)

        if prev_nose_x is not None and prev_nose_y is not None:
            dx = (nose.x - prev_nose_x) * SENSITIVITY
            dy = (nose.y - prev_nose_y) * SENSITIVITY
            text = f"dx: {dx:.2f}, dy: {dy:.2f}"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.arrowedLine(overlay, (nose_x, nose_y),
                            (int(nose_x + dx), int(nose_y + dy)),
                            (255, 0, 0), 2, tipLength=0.4)
            cur_x, cur_y = pyautogui.position()
            new_x = max(SAFE_MARGIN, min(screen_w - SAFE_MARGIN, cur_x + dx))
            new_y = max(SAFE_MARGIN, min(screen_h - SAFE_MARGIN, cur_y + dy))
            
            threshold = 1
            if abs(dx) >= threshold and abs(dy)>= threshold:
                pyautogui.moveTo(new_x, new_y)

        prev_nose_x, prev_nose_y = nose.x, nose.y

    alpha = 0.9
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow("Head Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
