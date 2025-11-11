import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import time
import numpy as np
import math

if not os.path.exists("gesture_recognizer.task"):
    print("Downloading default MediaPipe gesture_recognizer.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    urllib.request.urlretrieve(url, "gesture_recognizer.task")
    print("Model downloaded successfully.")

# Initialize MediaPipe models
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# --------------------------------------------------------------------------------
# Function for counting fingers
def count_fingers(hand_landmarks, handedness_label, image_width, image_height, roi=None):
    """
    Count how many fingers are raised on a given hand using landmark positions.
    """

    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky fingertips
    FINGER_PIPS = [6, 10, 14, 18]  # Corresponding knuckles (PIP joints)
    FINGER_MCPS = [5, 9, 13, 17]   # MCP joints

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    hand_w = (max(xs) - min(xs)) if xs else 0.0
    hand_h = (max(ys) - min(ys)) if ys else 0.0
    scale = max(hand_w, hand_h) or 1e-6
    margin_y = 0.04 * scale
    margin_d = 0.06 * scale

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def angle(a, b, c):
        ux, uy = a.x - b.x, a.y - b.y
        vx, vy = c.x - b.x, c.y - b.y
        du = math.hypot(ux, uy) or 1e-6
        dv = math.hypot(vx, vy) or 1e-6
        cosang = (ux * vx + uy * vy) / (du * dv)
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    def finger_inside_roi(idxs):
        if roi is None:
            return True
        x0, y0, x1, y1 = roi
        for idx in idxs:
            px = int(lm[idx].x * image_width)
            py = int(lm[idx].y * image_height)
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                return False
        return True

    fingers = 0
    wrist = lm[0]

    for tip_idx, pip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS):
        if not finger_inside_roi([mcp_idx, pip_idx, tip_idx]):
            continue

        tip_up = lm[tip_idx].y < (lm[pip_idx].y - margin_y)
        straight = angle(lm[mcp_idx], lm[pip_idx], lm[tip_idx]) > 160.0
        radial = dist(wrist, lm[tip_idx]) > dist(wrist, lm[pip_idx]) + margin_d

        if (tip_up and radial) or (straight and radial):
            fingers += 1

    if finger_inside_roi([THUMB_MCP, THUMB_IP, THUMB_TIP]):
        thumb_straight = angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP]) > 160.0
        thumb_radial = dist(wrist, lm[THUMB_TIP]) > dist(wrist, lm[THUMB_IP]) + (margin_d * 0.5)
        if thumb_straight and thumb_radial:
            fingers += 1

    return fingers


# --------------------------------------------------------------------------------
# 3D solution -between hands
def draw_3d_result(frame, results, result_value):


    if not (results and results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2):
        return


    left_wrist = results.multi_hand_landmarks[0].landmark[0]
    right_wrist = results.multi_hand_landmarks[1].landmark[0]

    mid_x = (left_wrist.x + right_wrist.x) / 2
    mid_y = (left_wrist.y + right_wrist.y) / 2
    mid_z = (left_wrist.z + right_wrist.z) / 2

    h, w, _ = frame.shape
    px, py = int(mid_x * w), int(mid_y * h)


    scale = max(0.6, 2.0 - abs(mid_z) * 4.0)

   
    for offset in range(3):
        cv2.putText(frame, str(result_value), (px - 60 + offset, py + offset),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)

 
    cv2.putText(frame, str(result_value), (px - 60, py),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 4, cv2.LINE_AA)


# --------------------------------------------------------------------------------
# Main loop for live hand tracking and finger counting
def main():

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Webcam error")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Gesture recognizer setup
    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Initialize variables
    gesture_result = ""
    most_recent_gesture = ""
    main.previous_gesture = None
    frame_count = 0
    prev_time = time.time()

    smoothed_roi = None
    alpha_w = 0.15
    alpha_h = 0.35

    # State machine variables
    state = 1
    first_number = None
    operation = None
    second_number = None
    stable_value = None
    stable_start = None
    stable_required_seconds = 5.0

    def reset_stable(new_candidate, now):
        return new_candidate, now

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=6,
        model_complexity=0,
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

            # Pose for ROI
            pose_results = pose.process(rgb)
            x_min, y_min, x_max, y_max = 0, 0, w, h

            if pose_results.pose_landmarks:
                l_sh = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                l_sh_xy = (int(l_sh.x * w), int(l_sh.y * h))
                r_sh_xy = (int(r_sh.x * w), int(r_sh.y * h))
                l_hip_xy = (int(l_hip.x * w), int(l_hip.y * h))
                r_hip_xy = (int(r_hip.x * w), int(r_hip.y * h))

                pad_w = 60
                x_min = max(0, min(l_sh_xy[0], r_sh_xy[0]) - pad_w)
                x_max = min(w, max(l_sh_xy[0], r_sh_xy[0]) + pad_w)
                base_y_min = max(0, min(l_sh_xy[1], r_sh_xy[1]) - 40)
                base_y_max = min(h, int((l_hip_xy[1] + r_hip_xy[1]) / 2))
                y_min, y_max = base_y_min, base_y_max

            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results and results.multi_hand_landmarks:
                hand_min_y = h
                hand_max_y = 0
                PAD_Y = 30

                for hand_lms in results.multi_hand_landmarks:
                    ys = [int(pt.y * h) for pt in hand_lms.landmark]
                    if not ys:
                        continue
                    hand_min_y = min(hand_min_y, min(ys))
                    hand_max_y = max(hand_max_y, max(ys))

                if hand_max_y > 0:
                    y_min = min(y_min, hand_min_y - PAD_Y)
                    y_max = max(y_max, hand_max_y + PAD_Y)
                y_min = max(0, y_min)
                y_max = min(h, y_max)

            new_roi = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            if smoothed_roi is None:
                smoothed_roi = new_roi
            else:
                smoothed_roi = np.array([
                    smoothed_roi[0] * (1 - alpha_w) + new_roi[0] * alpha_w,
                    smoothed_roi[1] * (1 - alpha_h) + new_roi[1] * alpha_h,
                    smoothed_roi[2] * (1 - alpha_w) + new_roi[2] * alpha_w,
                    smoothed_roi[3] * (1 - alpha_h) + new_roi[3] * alpha_h
                ], dtype=np.float32)

            xi, yi, xa, ya = smoothed_roi.astype(int)
            cv2.rectangle(frame, (xi, yi), (xa, ya), (0, 255, 0), 2)

            # Finger counting
            left_fingers = 0
            right_fingers = 0
            if results and results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    fingers_up = count_fingers(hand_lms, label, w, h, roi=(xi, yi, xa, ya))

                    if label == "Left":
                        left_fingers = fingers_up
                    else:
                        right_fingers = fingers_up

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2)
                    )

            # Gesture recognition
            frame_count += 1
            if frame_count % 3 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                gesture_result = recognizer.recognize(mp_image)
                frame_count = 0

            invalid_gestures_list = {"None", "Victory", "Pointing_Up", "Open_Palm", ""}
            confidence_threshold = 0.60
            previous_gesture = getattr(main, "previous_gesture", None)
            most_recent_gesture = previous_gesture

            candidate = None
            if gesture_result and getattr(gesture_result, "gestures", None):
                try:
                    candidate = gesture_result.gestures[0][0]
                except Exception:
                    candidate = None

            if candidate is not None:
                category_name = getattr(candidate, "category_name", None)
                score = getattr(candidate, "score", 0.0)

                if ((category_name is not None) and (str(category_name) not in invalid_gestures_list) and
                        (score >= confidence_threshold)):
                    main.previous_gesture = candidate
                    most_recent_gesture = candidate
                else:
                    most_recent_gesture = previous_gesture

            if ((most_recent_gesture is not None) and
                    (getattr(most_recent_gesture, "category_name", None) not in invalid_gestures_list)):
                score = getattr(most_recent_gesture, "score", 0.0)
                gesture_text = f"{most_recent_gesture.category_name} ({score:.2f})"
            else:
                gesture_text = "Waiting for operation"

            remembered_name = None
            if getattr(main, "previous_gesture", None) is not None:
                remembered_name = getattr(getattr(main, "previous_gesture", None),
                                          "category_name", None)

            if remembered_name:
                if "Thumb_Up" in remembered_name:
                    operation_text = "Addition"
                elif "Thumb_Down" in remembered_name:
                    operation_text = "Subtraction"
                elif "Closed_Fist" in remembered_name:
                    operation_text = "Multiplication"
                elif "ILoveYou" in remembered_name:
                    operation_text = "Division"
                else:
                    operation_text = None
            else:
                operation_text = None

            # Top-left info box
            cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
            cv2.putText(frame, gesture_text,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            if operation_text is not None:
                cv2.putText(frame, f"{operation_text}",
                            (350, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

            # --------------------------------------------------------------------------------
            # Calculator State Machine

            candidate_number = (left_fingers + right_fingers)
            candidate_gesture_name = remembered_name
            now = time.time()

            # State 1: first operand
            if state == 1:
                cv2.putText(frame, "Enter the first operand",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                if stable_value is None:
                    stable_value, stable_start = reset_stable(candidate_number, now)

                if candidate_number != stable_value:
                    stable_value, stable_start = reset_stable(candidate_number, now)

                if (stable_start is not None) and ((now - stable_start) >= stable_required_seconds):
                    first_number = stable_value
                    state = 2
                    stable_value = None
                    stable_start = None

            # State 2: operation
            elif state == 2:
                cv2.putText(frame, "Enter the desired operation",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                if (stable_value is None) and (candidate_gesture_name is not None):
                    stable_value, stable_start = reset_stable(candidate_gesture_name, now)

                if candidate_gesture_name != stable_value:
                    stable_value, stable_start = reset_stable(candidate_gesture_name, now)

                if (stable_start is not None) and (stable_value not in invalid_gestures_list) and \
                        ((now - stable_start) >= stable_required_seconds):
                    operation = stable_value
                    state = 3
                    stable_value = None
                    stable_start = None

            # State 3: second operand
            elif state == 3:
                cv2.putText(frame, "Enter the second operand",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                if stable_value is None:
                    stable_value, stable_start = reset_stable(candidate_number, now)

                if candidate_number != stable_value:
                    stable_value, stable_start = reset_stable(candidate_number, now)

                if (stable_start is not None) and ((now - stable_start) >= stable_required_seconds):
                    second_number = stable_value
                    state = 4

            # State 4: show result
            elif state == 4:
                cv2.putText(frame, "Calculation complete.",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'r' to restart the calculator.",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                if "Thumb_Up" in operation:
                    result_value = first_number + second_number
                elif "Thumb_Down" in operation:
                    result_value = first_number - second_number
                elif "Closed_Fist" in operation:
                    result_value = first_number * second_number
                elif "ILoveYou" in operation:
                    if second_number == 0:
                        result_value = "Divide by zero"
                    else:
                        result_value = round(first_number / second_number, 4)
                else:
                    result_value = "ERROR"

                gesture_symbol_map = {
                    "Thumb_Up": "+",
                    "Thumb_Down": "-",
                    "Closed_Fist": "*",
                    "ILoveYou": "/"
                }

                cv2.putText(
                    frame,
                    f"{first_number} {gesture_symbol_map.get(operation, operation)} {second_number} = {result_value}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Draw the floating result between hands
                draw_3d_result(frame, results, result_value)

            # Timer / input display
            if state in [1, 3]:
                if stable_start is not None:
                    remaining = (stable_required_seconds - (now - stable_start))
                else:
                    remaining = stable_required_seconds
                cv2.putText(frame, f"Input: {candidate_number}  Timer: {remaining:.1f} sec",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            elif state == 2:
                if stable_start is not None:
                    remaining = max(0, stable_required_seconds - (now - stable_start))
                else:
                    remaining = stable_required_seconds
                cv2.putText(frame, f"Input: {candidate_gesture_name}  Timer: {remaining:.1f} sec",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            current_time = time.time()
            fps = round(1 / (current_time - prev_time))
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps}",
                        (1100, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show window
            cv2.imshow("ARGZ Gesture-Based Calculator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif key in [ord('r'), ord('R')]:
                first_number = None
                operation = None
                second_number = None
                stable_value = None
                stable_start = None
                state = 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
