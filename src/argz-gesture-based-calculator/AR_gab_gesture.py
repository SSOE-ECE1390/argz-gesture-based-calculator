import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import time
import numpy as np
import math
import open3d as o3d
from pathlib import Path

# --------------------------------------------------------------------------------
# Download default gesture model if it doesn't exist
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
# Find 7-DOF Helmert Transform (from lecture notebook)
def find_helmert_transform(src_points, dst_points):
    """
    Finds the 7-parameter Helmert (similarity) transformation between
    two sets of corresponding 3D points using least squares.
    """
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    src_norm_sq = np.sum(src_centered ** 2)
    dst_norm_sq = np.sum(dst_centered ** 2)
    scale = np.sqrt(dst_norm_sq / src_norm_sq)
    R *= scale
    t = dst_centroid - (R @ src_centroid)
    t = np.expand_dims(t, axis=1)
    return R, t


# --------------------------------------------------------------------------------
# Function for counting fingers (same as original)
def count_fingers(hand_landmarks, handedness_label, image_width, image_height, roi=None):
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    FINGER_MCPS = [5, 9, 13, 17]

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
# Draw 3D result mesh between both hands
def draw_3d_result(frame, results, value):
    """
    Displays the calculation result as a 3D STL mesh between both hands.
    """
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        return

    # Select 7 reference points from both hands
    hand_points = []
    for hand in results.multi_hand_landmarks[:2]:
        wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
        thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]
        palm = hand.landmark[mp_hands.HandLandmark.PINKY_MCP]
        hand_points.extend([
            [wrist.x, wrist.y, wrist.z],
            [thumb_tip.x, thumb_tip.y, thumb_tip.z],
            [index_tip.x, index_tip.y, index_tip.z],
            [middle_tip.x, middle_tip.y, middle_tip.z],
            [ring_tip.x, ring_tip.y, ring_tip.z],
            [pinky_tip.x, pinky_tip.y, pinky_tip.z],
            [palm.x, palm.y, palm.z],
        ])
    hand_points = np.array(hand_points[:7])

    # Canonical cube for reference alignment
    cube_ref = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0],
        [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]
    ])

    # Compute Helmert transform
    R, t = find_helmert_transform(cube_ref, hand_points)

    # Load STL (or fallback cube)
    script_path = Path(__file__).absolute().parent
    stlfile = script_path / "number.stl"  # put your STL file here
    if stlfile.exists():
        mesh = o3d.io.read_triangle_mesh(str(stlfile))
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        verts -= np.mean(verts, axis=0)
        verts /= np.max(np.abs(verts))
        verts += 0.5
    else:
        verts = np.array([
            [0,0,1], [1,0,1], [1,1,1], [0,1,1],
            [0,0,0], [1,0,0], [1,1,0], [0,1,0]
        ])
        tris = np.array([
            [0,1,2],[0,2,3],[4,5,6],[4,6,7],
            [0,1,5],[0,5,4],[2,3,7],[2,7,6],
            [1,2,6],[1,6,5],[0,3,7],[0,7,4]
        ])

    # Transform vertices
    verts_t = (R @ verts.T + t @ np.ones((1, verts.shape[0]))).T

    # Draw mesh into frame
    h, w, _ = frame.shape
    zmean = np.mean(verts_t[tris, 2], axis=1)
    for tri in tris[np.argsort(zmean)]:
        pts = np.zeros((3, 2), dtype=np.int32)
        pts[:, 0] = (verts_t[tri, 0] * w).astype(int)
        pts[:, 1] = (verts_t[tri, 1] * h).astype(int)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1)

    # Label the 3D object
    cx = int(np.mean(verts_t[:, 0]) * w)
    cy = int(np.mean(verts_t[:, 1]) * h)
    cv2.putText(frame, str(value), (cx - 25, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3, cv2.LINE_AA)


# --------------------------------------------------------------------------------
# Main loop (same logic as your working version)
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Webcam error")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    gesture_result = ""
    most_recent_gesture = ""
    main.previous_gesture = None
    frame_count = 0
    prev_time = time.time()
    smoothed_roi = None
    alpha_w = 0.15
    alpha_h = 0.35

    # Calculator state variables
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

            # ROI smoothing
            new_roi = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            if smoothed_roi is None:
                smoothed_roi = new_roi
            else:
                smoothed_roi = np.array([
                    smoothed_roi[0]*(1-alpha_w)+new_roi[0]*alpha_w,
                    smoothed_roi[1]*(1-alpha_h)+new_roi[1]*alpha_h,
                    smoothed_roi[2]*(1-alpha_w)+new_roi[2]*alpha_w,
                    smoothed_roi[3]*(1-alpha_h)+new_roi[3]*alpha_h
                ], dtype=np.float32)
            xi, yi, xa, ya = smoothed_roi.astype(int)
            cv2.rectangle(frame, (xi, yi), (xa, ya), (0, 255, 0), 2)

            # Finger counting
            left_fingers, right_fingers = 0, 0
            if results and results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    fingers_up = count_fingers(hand_lms, label, w, h, roi=(xi, yi, xa, ya))
                    if label == "Left": left_fingers = fingers_up
                    else: right_fingers = fingers_up
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Gesture recognition
            frame_count += 1
            if frame_count % 3 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                gesture_result = recognizer.recognize(mp_image)
                frame_count = 0

            invalid_gestures = {"None", "Victory", "Pointing_Up", "Open_Palm", ""}
            candidate = None
            if gesture_result and getattr(gesture_result, "gestures", None):
                try: candidate = gesture_result.gestures[0][0]
                except Exception: pass
            if candidate and getattr(candidate, "score", 0.0) >= 0.6:
                if getattr(candidate, "category_name", None) not in invalid_gestures:
                    main.previous_gesture = candidate
                    most_recent_gesture = candidate
            previous_gesture = getattr(main, "previous_gesture", None)
            gesture_name = getattr(previous_gesture, "category_name", None) if previous_gesture else None

            # Operation text
            op_map = {"Thumb_Up": "+", "Thumb_Down": "-", "Closed_Fist": "*", "ILoveYou": "/"}
            op_text = {"Thumb_Up": "Addition", "Thumb_Down": "Subtraction",
                       "Closed_Fist": "Multiplication", "ILoveYou": "Division"}.get(gesture_name, None)

            # Draw gesture info box
            cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
            cv2.putText(frame, gesture_name or "Waiting", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if op_text:
                cv2.putText(frame, op_text, (350, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Calculator logic
            candidate_number = left_fingers + right_fingers
            now = time.time()

            if state == 1:
                cv2.putText(frame, "Enter the first operand", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                if stable_value is None: stable_value, stable_start = candidate_number, now
                if candidate_number != stable_value: stable_value, stable_start = candidate_number, now
                if (now - stable_start) >= stable_required_seconds:
                    first_number = stable_value; state = 2; stable_value = None

            elif state == 2:
                cv2.putText(frame, "Enter the operation", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                if stable_value is None: stable_value, stable_start = gesture_name, now
                if gesture_name != stable_value: stable_value, stable_start = gesture_name, now
                if (now - stable_start) >= stable_required_seconds:
                    operation = stable_value; state = 3; stable_value = None

            elif state == 3:
                cv2.putText(frame, "Enter the second operand", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                if stable_value is None: stable_value, stable_start = candidate_number, now
                if candidate_number != stable_value: stable_value, stable_start = candidate_number, now
                if (now - stable_start) >= stable_required_seconds:
                    second_number = stable_value; state = 4; stable_value = None

            elif state == 4:
                cv2.putText(frame, "Calculation complete!", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                op_char = op_map.get(operation, "+")
                if op_char == "+": result_value = first_number + second_number
                elif op_char == "-": result_value = first_number - second_number
                elif op_char == "*": result_value = first_number * second_number
                elif op_char == "/":
                    result_value = "Divide by zero" if second_number == 0 else round(first_number / second_number, 2)
                else: result_value = "?"

                # Draw 3D result
                draw_3d_result(frame, results, result_value)

            # Timer display
            if stable_start:
                remaining = max(0.0, stable_required_seconds - (now - stable_start))
                cv2.putText(frame, f"Input: {candidate_number}  Timer: {remaining:.1f}s",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Gesture Calculator", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
