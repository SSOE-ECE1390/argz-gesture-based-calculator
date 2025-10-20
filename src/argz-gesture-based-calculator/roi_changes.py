import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import numpy as np
import math

if not os.path.exists("gesture_recognizer.task"):
    print("Downloading default MediaPipe gesture_recognizer.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    urllib.request.urlretrieve(url, "gesture_recognizer.task")
    print("Model downloaded successfully.")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def count_fingers(hand_landmarks, handedness_label, image_width, image_height, roi=None):
    lm = hand_landmarks.landmark

    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20]     
    FINGER_PIPS = [6, 10, 14, 18]   
    FINGER_MCPS = [5, 9, 13, 17]     
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    hand_w = (max(xs) - min(xs)) if xs else 0.0
    hand_h = (max(ys) - min(ys)) if ys else 0.0
    scale = max(hand_w, hand_h) or 1e-6
    margin_y = 0.04 * scale
    margin_d = 0.06 * scale

    def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)

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
    alpha_w = 0.15
    alpha_h = 0.35

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
            if results.multi_hand_landmarks:
                hand_min_y = h
                hand_max_y = 0
                PAD_Y = 30

                for hand_lms in results.multi_hand_landmarks:
                    ys = [int(pt.y * h) for pt in hand_lms.landmark]
                    if not ys: continue
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

            left_fingers, right_fingers = 0, 0
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label

                    fingers_up = count_fingers(
                        hand_lms, label, w, h,
                        roi=(xi, yi, xa, ya)
                    )

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
                    lx, ly = int(coords.x * w), int(coords.y * h)
                    cv2.putText(frame, f"{label}: {fingers_up}",
                                (lx - 40, ly - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (10, 10), (420, 160), (0, 0, 0), -1)
            cv2.putText(frame, f"Left Hand: {left_fingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Right Hand: {right_fingers}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
                    candidate = None
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
            cv2.putText(frame, gesture_text, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
                cv2.putText(frame, operation_text, (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Result: {operation_result_text}",
                            (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

            cv2.imshow("Finger Counter (Dynamic ROI + Per-Finger Gating)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

