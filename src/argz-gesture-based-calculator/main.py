import cv2
import mediapipe as mp
import os, urllib.request
import time
import numpy as np
import math
import pickle
from pathlib import Path

# The path to the model file must be in the same directory as this script.
# Must also pip install scikit-learn
MODEL_PATH = 'gesture_classifier_model.pkl' 

# Index Finger Landmark Indices (5, 6, 7, 8)
# 5: Index finger base (MCP), 6: PIP, 7: DIP, 8: Tip
INDEX_FINGER_LANDMARKS = [5, 6, 7, 8] 
# ---------------------------------------------------

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# --- Custom Gesture Classifier Functions (Copied from detect_math_gesture.py) ---
def load_classifier(model_path):
    """Loads the trained model and label map from the pickle file."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            label_map = data['label_map']
            
            # Reverse the map for display purposes
            label_reverse_map = {v: k for k, v in label_map.items()}
            
            # We also return the label_map for mapping names to operations later
            return model, label_reverse_map, label_map 
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run training first.")
        # Exit the application cleanly if the model isn't found
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def extract_index_finger_features_single(hand_landmarks):
    """
    Extracts and normalizes the 4 index finger landmarks relative to the wrist (landmark 0) 
    for a single hand, matching the training script logic.
    
    Returns: A 12-element numpy array.
    """
    # Extract wrist landmark (index 0) for normalization
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    normalized_features = []
    
    # Iterate ONLY through the index finger landmarks and normalize
    for i in INDEX_FINGER_LANDMARKS:
        landmark = hand_landmarks.landmark[i]
        
        # Calculate coordinates relative to the wrist
        rel_x = landmark.x - wrist_x
        rel_y = landmark.y - wrist_y
        rel_z = landmark.z - wrist_z
        
        normalized_features.extend([rel_x, rel_y, rel_z])

    return np.array(normalized_features)


def extract_dual_index_finger_features(results):
    """
    Extracts features for up to two hands (index fingers only).
    Pads with zeros if only one hand is found, resulting in a 24-element vector.

    Args:
        results: The MediaPipe hands processing results object.

    Returns:
        np.array or None: A flattened 24-element feature vector, or None if no hands detected.
    """
    all_features = []
    num_hands_detected = 0

    if results.multi_hand_landmarks:
        num_hands_detected = len(results.multi_hand_landmarks)
        
        # Ensure that if we have more than 2 hands, we only process the first two detected by MediaPipe
        for i in range(min(2, num_hands_detected)):
            hand_landmarks = results.multi_hand_landmarks[i]
            features = extract_index_finger_features_single(hand_landmarks)
            all_features.append(features)
    
    if num_hands_detected == 0:
        return None
    
    # Padding: If only one hand is detected, append 12 zeros for the missing hand
    if num_hands_detected == 1:
        # 4 index finger landmarks * 3 coordinates = 12 zero padding elements
        padding = np.zeros(12, dtype=np.float32) 
        all_features.append(padding)

    # Concatenate features into a single 24-element vector and reshape for the classifier
    return np.concatenate(all_features).reshape(1, -1)


# --- Original count_fingers function (kept for number input) ---
def count_fingers(hand_landmarks, handedness_label, image_width, image_height, roi=None):
    """
    Count how many fingers are raised on a given hand using landmark positions.

    Args:
        hand_landmarks: The 21 hand landmarks detected by MediaPipe.
        handedness_label: 'Left' or 'Right' (determined by MediaPipe).
        image_width, image_height: Dimensions of the current video frame.
        roi: Region Of Interest wherein fingers will be counted

    Returns:
        The number of fingers currently detected as being raised (0-5).
    """

    # Indices of relevant landmarks in MediaPipe's hand model
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20] # Index, Middle, Ring, Pinky fingertips
    FINGER_PIPS = [6, 10, 14, 18] # Corresponding knuckles (PIP joints)
    FINGER_MCPS = [5, 9, 13, 17]  # 


    # Calculations used for finger counting
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    hand_w = (max(xs) - min(xs)) if xs else 0.0
    hand_h = (max(ys) - min(ys)) if ys else 0.0
    scale = max(hand_w, hand_h) or 1e-6
    margin_y = 0.04 * scale
    margin_d = 0.06 * scale

    # Helper function for calculating finger distances
    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    # Helper function for calculating finger angles
    def angle(a, b, c):
        ux, uy = a.x - b.x, a.y - b.y
        vx, vy = c.x - b.x, c.y - b.y
        du = math.hypot(ux, uy) or 1e-6
        dv = math.hypot(vx, vy) or 1e-6
        cosang = (ux * vx + uy * vy) / (du * dv)
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    # Helper function for determining whether a finger is inside of the cropped ROI
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

    # Initialize finger count as 0 and the wrist location
    fingers = 0
    wrist = lm[0]


    # Count the number of valid fingers inside of the ROI
    for tip_idx, pip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS):
        if not finger_inside_roi([mcp_idx, pip_idx, tip_idx]):
            continue

        tip_up = lm[tip_idx].y < (lm[pip_idx].y - margin_y)
        straight = angle(lm[mcp_idx], lm[pip_idx], lm[tip_idx]) > 160.0
        radial = dist(wrist, lm[tip_idx]) > dist(wrist, lm[pip_idx]) + margin_d

        if (tip_up and radial) or (straight and radial):
            fingers += 1

    # Count the number of valid thumbs inside of the ROI
    if finger_inside_roi([THUMB_MCP, THUMB_IP, THUMB_TIP]):
        thumb_straight = angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP]) > 160.0
        thumb_radial = dist(wrist, lm[THUMB_TIP]) > dist(wrist, lm[THUMB_IP]) + (margin_d * 0.5)
        if thumb_straight and thumb_radial:
            fingers += 1

    return fingers

# Main loop for live hand tracking and finger counting
def main():
    
    # ** FOR NON-MAC CHANGE to cv2.VideoCapture(0)
    # ** WINDOWS SPECIFIC is cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # ** MAC uses cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Webcam error")

    # Set video resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load the trained model and label maps
    global custom_model, label_reverse_map, label_map
    custom_model, label_reverse_map, label_map = load_classifier(MODEL_PATH)
    # -------------------------------

    # Initialize variables before use
    # Removed mediapipe gesture result and frame_count variables
    most_recent_gesture_id = None
    main.previous_gesture_id = None 

    # Timer used for FPS calculation
    prev_time = time.time()

    # ROI setup
    smoothed_roi = None
    alpha_w = 0.15
    alpha_h = 0.35

    # State machine variables
    state = 1                      # Current state
    first_number = None            # User's first number input
    operation = None               # User's operation input (will be the gesture name string)
    second_number = None           # User's second number input
    stable_value = None            # Current candidate being held (either number or gesture name)
    stable_start = None            # Time when candidate first observed
    stable_required_seconds = 5.0  # Time required to "accept" user input

    # Define helper function to reset stable tracking when entering a new state or when
    # candidate changes in the calculator state machine.
    def reset_stable(new_candidate, now):
        return new_candidate, now

    # Configure models for Pose (used to track body for ROI) and Hands (used to count fingers 
    # and extract features for custom gesture)
    with mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 0,
        enable_segmentation = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,              # Define number of hands used for gesture classification
        model_complexity = 1,           # 0 improves frame rate, 1 improves model accuracy
        min_detection_confidence = 0.6, 
        min_tracking_confidence = 0.6,
    ) as hands:

        #-----------------------------------------------------------------------------------------------------
        # Main Loop

        while True:
            # Read a frame from the webcam
            ok, frame = cap.read()
            if not ok:
                break

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB (MediaPipe expects RGB)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process Pose for ROI
            pose_results = pose.process(rgb)
            x_min, y_min, x_max, y_max = 0, 0, w, h


            #--------------------------------------------------------------------------------------------
            # Region Of Interest (ROI)

            # Configure the ROI based on the user's body position
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
            
            # Process Hands for Landmarks
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Adjust ROI based on hand position
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

            # Display the ROI on the frame
            xi, yi, xa, ya = smoothed_roi.astype(int)
            cv2.rectangle(frame, (xi, yi), (xa, ya), (0, 255, 0), 2)


            #--------------------------------------------------------------------------------------------
            # Finger Counting / Drawing Hands

            left_fingers = 0
            right_fingers = 0

            # Draw only the Index Finger Landmarks (matching detect_math_gesture.py)
            INDEX_FINGER_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
            INDEX_FINGER_POINTS = INDEX_FINGER_LANDMARKS

            # If hands are detected, process each one by classifying as left/right and counting fingers.
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Only count fingers inside of the ROI
                    fingers_up = count_fingers(hand_lms, label, w, h, roi=(xi, yi, xa, ya))
                                               
                    # Store count for the appropriate hand
                    if label == "Left":
                        left_fingers = fingers_up
                    else:
                        right_fingers = fingers_up

                    # Do not display finger outline in order to improve performance
                    """
                    for connection in INDEX_FINGER_CONNECTIONS:
                        start_point = hand_lms.landmark[connection[0]]
                        end_point = hand_lms.landmark[connection[1]]
                        
                        start_px = mp_drawing._normalized_to_pixel_coordinates(start_point.x, start_point.y, w, h)
                        end_px = mp_drawing._normalized_to_pixel_coordinates(end_point.x, end_point.y, w, h)

                        if start_px and end_px:
                            cv2.line(frame, start_px, end_px, (0, 255, 0), 3) # Green line
                    
                    # Draw the index finger points themselves
                    for i in INDEX_FINGER_POINTS:
                        landmark = hand_lms.landmark[i]
                        point_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, w, h)
                        if point_px:
                            cv2.circle(frame, point_px, 5, (0, 0, 255), -1) # Red dot
                    """
                    # Get approximate coordinates for the label (near wrist)
                    coords = hand_lms.landmark[0]
                    x_wrist, y_wrist = int(coords.x * frame.shape[1]), int(coords.y * frame.shape[0])
                    cv2.putText(frame, f"{label}: {fingers_up}",
                                (x_wrist - 40, y_wrist - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            #-----------------------------------------------------------------------------------------------------
            # Gesture Classification

            # Extract features using the dual-hand logic
            feature_vector = extract_dual_index_finger_features(results)
            
            # Predict the gesture
            most_recent_gesture_name = None
            if feature_vector is not None and feature_vector.size == 24:
                # Predict the gesture ID (0, 1, 2, 3...)
                prediction_id = custom_model.predict(feature_vector)[0]
                
                # Get the gesture name string (e.g., 'Add', 'Subtract')
                candidate_gesture_name = label_reverse_map.get(prediction_id, 'UNKNOWN')
                
                # We need to know the confidence, so we calculate the probability
                probabilities = custom_model.predict_proba(feature_vector)[0]
                score = probabilities[prediction_id]
            else:
                candidate_gesture_name = None
                score = 0.0

            # Apply stability logic (modified to use gesture name string)
            # Define the gestures that we should ignore. These will NOT cause the previous gesture to be updated.
            # Using only 'UNKNOWN' and None from the detection side for invalid candidates
            invalid_gestures_list = {"UNKNOWN", None} 

            # Confidence threshold to detect a new gesture
            confidence_threshold = 0.60 # Kept original threshold

            # Retrieve persistent previous gesture name
            previous_gesture_name = getattr(main, "previous_gesture_name", None)

            # Start by assuming we will keep whatever previous valid gesture we already have
            most_recent_gesture_name = previous_gesture_name

            # If we got a valid candidate gesture
            if candidate_gesture_name not in invalid_gestures_list and score >= confidence_threshold:
                main.previous_gesture_name = candidate_gesture_name
                most_recent_gesture_name = candidate_gesture_name
            else:
                # If the candidate is invalid, keep the previous valid gesture
                most_recent_gesture_name = previous_gesture_name


            # Format gesture output text
            if most_recent_gesture_name is not None and most_recent_gesture_name != "UNKNOWN":
                gesture_text = f"Gesture: {most_recent_gesture_name} (Score: {score:.2f})"
            else:
                gesture_text = "Waiting for operation gesture"

            # Map the custom gesture names to operation symbols and longer text
            operation_text = None
            gesture_symbol_map = {
                "Add": "+", 
                "Subtract": "-", 
                "Multiply": "*", 
                "Divide": "/"
            }
            
            # Get the operation text/name for display/state machine
            if most_recent_gesture_name in gesture_symbol_map:
                operation_text = most_recent_gesture_name
            else:
                 operation_text = "None" # For clearer display if we have a hand but no classified gesture


            # Display gesture result and the corresponding operation, if there is one
            cv2.rectangle(frame, (10, 10), (600, 170), (0, 0, 0), -1)
            cv2.putText(frame, gesture_text,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            if (operation_text is not None) and (operation_text != "None"):
                cv2.putText(frame, f"Operation: {operation_text}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Operation: None",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)


            #-----------------------------------------------------------------------------------------------------
            # Calculator State Machine
            
            # Current candidate values
            candidate_number = (left_fingers + right_fingers)  # fingers held up between left and right hands, 0-10
            candidate_gesture_name = most_recent_gesture_name

            # Save the time before running the state machine for this frame
            now = time.time()

            # State 1: Get first number input
            if (state == 1):
                # Display prompt
                cv2.putText(frame, f"Enter the first number",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Initialize stable value if it's currently None
                if (stable_value is None):
                    stable_value, stable_start = reset_stable(candidate_number, now)

                # If the candidate number changed, restart the timer
                if (candidate_number != stable_value):
                    stable_value, stable_start = reset_stable(candidate_number, now)

                # If the timer has completed successfully with a valid number, save the first number
                if ((stable_start is not None) and ((now - stable_start) >= stable_required_seconds)):
                    first_number = stable_value
                    state = 2

                    # Clear stable values before going to new state
                    stable_value = None
                    stable_start = None

            # State 2: Get operation input
            elif (state == 2):
                # Display prompt
                cv2.putText(frame, f"Enter the desired operation",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Initialize stable value if it's currently None
                if ((stable_value is None) and (candidate_gesture_name not in invalid_gestures_list)):
                    stable_value, stable_start = reset_stable(candidate_gesture_name, now)

                # If the candidate gesture changed, restart the timer
                if (candidate_gesture_name != stable_value):
                    stable_value, stable_start = reset_stable(candidate_gesture_name, now)

                # If the timer has completed successfully with a valid operation, save the operation
                if ((stable_start is not None) and (stable_value not in invalid_gestures_list) and ((now - stable_start) >= stable_required_seconds)):
                    operation = stable_value
                    state = 3

                    # Clear stable values before going to new state
                    stable_value = None
                    stable_start = None

            # State 3: Get second number input
            elif state == 3:
                # Waiting for second number
                cv2.putText(frame, f"Enter the second number",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Initialize stable value if it's currently None
                if (stable_value is None):
                    stable_value, stable_start = reset_stable(candidate_number, now)

                # If the candidate number changed, restart the timer
                if (candidate_number != stable_value):
                    stable_value, stable_start = reset_stable(candidate_number, now)

                # If the timer has completed successfully with a valid number, save the first number
                if ((stable_start is not None) and ((now - stable_start) >= stable_required_seconds)):
                    second_number = stable_value
                    state = 4

            # State 4: Display the calculation result
            elif state == 4:
                # Display program completion text
                cv2.putText(frame, f"Calculation complete.",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Press 'r' to restart the calculator.",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Perform the desired calculation
                if operation == "Add":
                    result_value = (first_number + second_number)
                elif operation == "Subtract":
                    result_value = (first_number - second_number)
                elif operation == "Multiply":
                    result_value = (first_number * second_number)
                elif operation == "Divide":
                    if second_number == 0:
                        result_value = "Divide by zero"
                    else:
                        result_value = round(first_number / second_number, 4)
                else:
                    result_value = "ERROR"

                # Map the gesture names to operation symbols for displaying them
                # Reuse the symbol map defined earlier
                
                # Display calculation with result on frame
                cv2.putText(frame, f"{first_number} {gesture_symbol_map.get(operation, operation)} {second_number} = {result_value}",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the current number and a countdown timer if in state 1 or 3
            if ((state == 1) or (state == 3)):
                # Show candidate number and countdown
                if (stable_start is not None):
                    remaining = max(0, stable_required_seconds - (now - stable_start))
                else:
                    remaining = stable_required_seconds
                
                # Display info on frame
                cv2.putText(frame, f"Input: {candidate_number}  Timer: {remaining:.1f} sec",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display the current operation and a countdown timer if in state 2
            elif (state == 2):
                # Show candidate operations and countdown
                if (stable_start is not None):
                    remaining = max(0, stable_required_seconds - (now - stable_start))
                else:
                    remaining = stable_required_seconds

                # Display info on frame
                cv2.putText(frame, f"Input: {candidate_gesture_name}  Timer: {remaining:.1f} sec",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #-----------------------------------------------------------------------------------------------------

            # Calculate FPS and display it on the frame
            current_time = time.time()
            fps = round(1 / (current_time - prev_time))
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps}",
                        (1100, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the processed frame in window
            cv2.imshow("ARGZ Gesture-Based Calculator", frame)

            # Exit loop when 'q' is pressed or reset calculator when 'r' is pressed
            key = cv2.waitKey(1) & 0xFF
            if ((key == ord('q')) or (key == ord('Q'))):
                break
            elif ((key == ord('r')) or (key == ord('R'))):
                # Reset state machine to initial state and restart calculator at state 1
                first_number = None
                operation = None
                second_number = None
                stable_value = None
                stable_start = None
                state = 1
                # Reset gesture tracking
                main.previous_gesture_name = None

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()

# Run the main loop
main()