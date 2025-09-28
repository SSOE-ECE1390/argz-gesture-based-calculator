#GitHub repos used for code:
# - https://github.com/Mordekai66/finger-counter-mediapipe/blob/main/main.py
# - https://github.com/HarshitDolu/Finger-Counter-using-mediapipe/blob/main/Finger_counter.py
# - https://github.com/Sousannah/hand-tracking-using-mediapipe/blob/main/hand_tracking.py
# - https://github.com/Real-J/Finger-Counting-with-OpenCV-and-MediaPipe/blob/main/finger_counting.py

#Mediapipe docs
# - https://mediapipe-studio.webapps.google.com/studio/demo/gesture_recognizer


import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks, handedness_label, image_width, image_height):
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    fingers = 0
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip_idx].y < lm[pip_idx].y:
            fingers += 1
    if handedness_label == "Right":
        if lm[THUMB_TIP].x < lm[THUMB_IP].x:
            fingers += 1
    else:
        if lm[THUMB_TIP].x > lm[THUMB_IP].x:
            fingers += 1

    return fingers

def main(): 
    #FOR NON-MAC CHANGE to cv2.VideoCapture(0)
    # WINDOWS SPECIFIC is cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("webcam err")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            total_fingers = 0
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  
                    fingers_up = count_fingers(hand_lms, label, frame.shape[1], frame.shape[0])
                    total_fingers += fingers_up
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2)
                    )
                    coords = hand_lms.landmark[0]  
                    x, y = int(coords.x * frame.shape[1]), int(coords.y * frame.shape[0])
                    cv2.putText(frame, f"{label} hand: {fingers_up}",
                                (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (10, 10), (260, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Count: {total_fingers}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("idk", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

main()