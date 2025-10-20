import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import numpy as np

# Download default gesture model if it doesn't exist
if not os.path.exists("gesture_recognizer.task"):
    print("Downloading default MediaPipe gesture_recognizer.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    urllib.request.urlretrieve(url, "gesture_recognizer.task")
    print("Model downloaded successfully.")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def count_fingers(hand_landmarks, handedness_label, image_width, image_height):
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP = 4, 3
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
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("webcam err")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    main.previous_gesture = None

    smoothed_roi = None
    smoothing_factor = 0.15  


    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
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
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

         
            pose_results = pose.process(rgb)
            roi_box = None

            if pose_results.pose_landmarks:
                l_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                l_shoulder_xy = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                r_shoulder_xy = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                l_hip_xy = (int(l_hip.x * w), int(l_hip.y * h))
                r_hip_xy = (int(r_hip.x * w), int(r_hip.y * h))

                x_min = min(l_shoulder_xy[0], r_shoulder_xy[0]) - 60
                x_max = max(l_shoulder_xy[0], r_shoulder_xy[0]) + 60
                y_min = min(l_shoulder_xy[1], r_shoulder_xy[1]) - 40
                y_max = int((l_hip_xy[1] + r_hip_xy[1]) / 2)

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                roi_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

          
                if smoothed_roi is None:
                    smoothed_roi = roi_box
                else:
                    smoothed_roi = smoothed_roi * (1 - smoothing_factor) + roi_box * smoothing_factor

            # --- Draw ROI
            if smoothed_roi is not None:
                x_min, y_min, x_max, y_max = smoothed_roi.astype(int)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            else:
                x_min, y_min, x_max, y_max = 0, 0, w, h  # full frame if no person yet

            # --- Detect Hands and Filter by ROI
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            left_fingers, right_fingers = 0, 0

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    wrist = hand_lms.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                    # Skip hands outside smoothed ROI
                    if not (x_min <= wrist_x <= x_max and y_min <= wrist_y <= y_max):
                        continue

                    fingers_up = count_fingers(hand_lms, label, w, h)
                    if label == "Left":
                        left_fingers = fingers_up
                    else:
                        right_fingers = fingers_up

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2),
                    )

                    coords = hand_lms.landmark[0]
                    x, y = int(coords.x * w), int(coords.y * h)
                    cv2.putText(frame, f"{label}: {fingers_up}",
                                (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Display finger counts
            cv2.rectangle(frame, (10, 10), (380, 160), (0, 0, 0), -1)
            cv2.putText(frame, f"Left Hand: {left_fingers}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Right Hand: {right_fingers}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # --- Gesture recognition
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            gesture_result = recognizer.recognize(mp_image)
            invalid_gestures_list = {"None", "Victory", "Pointing_Up", "Open_Palm", ""}
            confidence_threshold = 0.6
            previous_gesture = getattr(main, "previous_gesture", None)
            most_recent_gesture = previous_gesture

            candidate = None
            if gesture_result and getattr(gesture_result, "gestures", None):
                try:
                    candidate = gesture_result.gestures[0][0]
                except Exception:
                    pass

            if candidate is not None:
                category_name = getattr(candidate, "category_name", None)
                score = getattr(candidate, "score", 0.0)
                if ((category_name is not None) and
                    (str(category_name) not in invalid_gestures_list) and
                        (score >= confidence_threshold)):
                    main.previous_gesture = candidate
                    most_recent_gesture = candidate

            if most_recent_gesture is not None and \
               getattr(most_recent_gesture, "category_name", None) not in invalid_gestures_list:
                score = getattr(most_recent_gesture, "score", 0.0)
                gesture_text = f"{most_recent_gesture.category_name} ({score:.2f})"
            else:
                gesture_text = "Waiting for operation"

            cv2.putText(frame, gesture_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --- Calculator
            sum_fingers = left_fingers + right_fingers
            product_fingers = left_fingers * right_fingers
            difference_fingers = left_fingers - right_fingers
            quotient_fingers = round(left_fingers / right_fingers, 2) if right_fingers != 0 else 0.0

            remembered_name = getattr(getattr(main, "previous_gesture", None), "category_name", None)

            operation_text, operation_result_text = None, None
            if remembered_name:
                if "Thumb_Up" in remembered_name:
                    operation_text = "Addition (L + R)"
                    operation_result_text = f"{sum_fingers}"
                elif "Thumb_Down" in remembered_name:
                    operation_text = "Subtraction (L - R)"
                    operation_result_text = f"{difference_fingers}"
                elif "Closed_Fist" in remembered_name:
                    operation_text = "Multiplication (L * R)"
                    operation_result_text = f"{product_fingers}"
                elif "ILoveYou" in remembered_name:
                    operation_text = "Division (L / R)"
                    operation_result_text = f"{quotient_fingers}"

            if operation_text and operation_result_text:
                cv2.putText(frame, operation_text, (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Result: {operation_result_text}",
                            (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)


            cv2.imshow("Finger Counter with Smooth ROI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
