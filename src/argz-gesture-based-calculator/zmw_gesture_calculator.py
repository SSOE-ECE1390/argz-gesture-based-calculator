import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request
import time
import numpy as np
import math

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

# Function for counting fingers
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


    """
    # Count other fingers (index, middle, ring, pinky)
    # A finger is considered "raised" if its tip landmark is above its PIP joint
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip_idx].y < lm[pip_idx].y: # y decreases as we go up
            fingers += 1

    # Thumb logic
    # The thumb moves sideways, so we compare x-coordinates instead of y.
    if handedness_label == "Right":
        if lm[THUMB_TIP].x < lm[THUMB_IP].x:
            fingers += 1
    else:
        if lm[THUMB_TIP].x > lm[THUMB_IP].x:
            fingers += 1
    """

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
    
    # Initialize webcam
    # ** FOR NON-MAC CHANGE to cv2.VideoCapture(0)
    # ** WINDOWS SPECIFIC is cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # ** MAC uses cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam error")

    # Set video resolution to 1280x720. Note that lowering the resolution slightly improves performance.
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Gesture recognizer setup    
    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Initialize variables before use
    gesture_result = ""
    most_recent_gesture = ""
    main.previous_gesture = None
    frame_count = 0

    # Timer used for FPS calculation
    prev_time = time.time()

    # ROI setup
    smoothed_roi = None
    alpha_w = 0.15
    alpha_h = 0.35

    # State machine variables
    state = 1                      # Current state
    first_number = None            # User's first number input
    operation = None               # User's operation input
    second_number = None           # User's second number input
    stable_value = None            # Current candidate being held
    stable_start = None            # Time when candidate first observed
    stable_required_seconds = 5.0  # Time required to "accept" user input

    # Define helper function to reset stable tracking when entering a new state or when
    # candidate changes in the calculator state machine.
    def reset_stable(new_candidate, now):
        return new_candidate, now


    """
    # Initialize MediaPipe Hands model
    with mp_hands.Hands(
        static_image_mode = False,      # Use video stream (not static images)
        max_num_hands = 2,              # Detect up to 2 hands
        model_complexity = 0,           # Using 0 improves frame rate, while 1 improves model accuracy
        min_detection_confidence = 0.6, # Minimum confidence for detection
        min_tracking_confidence = 0.6,  # Minimum confidence for tracking
    ) as hands:
    """

    # Configure models for Pose (used to track body for ROI) and Hands (used to count fingers)
    with mp_pose.Pose(
        static_image_mode = False,      # Use video stream (not static images)
        model_complexity = 0,           # 0 improves frame rate, 1 improves model accuracy
        enable_segmentation = False,    # 
        min_detection_confidence = 0.5, # Minimum confidence for detection
        min_tracking_confidence = 0.5,  # Minimum confidence for tracking
    ) as pose, mp_hands.Hands(
        static_image_mode = False,      # Use video stream (not static images)
        max_num_hands = 2,              # Detect up to 2 hands (TODO: set this higher and test with multiple sets of hands)
        model_complexity = 0,           # 0 improves frame rate, 1 improves model accuracy
        min_detection_confidence = 0.6, # Minimum confidence for detection
        min_tracking_confidence = 0.6,  # Minimum confidence for tracking
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

            # Display the ROI on the frame
            xi, yi, xa, ya = smoothed_roi.astype(int)
            cv2.rectangle(frame, (xi, yi), (xa, ya), (0, 255, 0), 2)


            #--------------------------------------------------------------------------------------------
            # Finger Counting

            # Initialize finger counts for each hand
            left_fingers = 0
            right_fingers = 0

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

                    # Draw hand landmarks and connections on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(thickness=2)
                    )

                    # Get approximate coordinates for the label (near wrist)
                    coords = hand_lms.landmark[0]
                    x, y = int(coords.x * frame.shape[1]), int(coords.y * frame.shape[0])
                    cv2.putText(frame, f"{label}: {fingers_up}",
                                (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            #-----------------------------------------------------------------------------------------------------
            # Gesture Recognition

            # Only run gesture recognition every 3 frames to improve frame rate without sacrificing much performance.
            # TODO: change this to only count operation gestures that are inside of the ROI. Right now, the operation can be anywhere in the frame.
            frame_count += 1
            if frame_count % 3 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                gesture_result = recognizer.recognize(mp_image)
                frame_count = 0

            # Define the gestures that we should ignore. These will NOT cause the previous gesture to be updated.
            invalid_gestures_list = {"None", "Victory", "Pointing_Up", "Open_Palm", ""}

            # Confidence threshold to detect a new gesture
            confidence_threshold = 0.60

            # Retrieve persistent previous gesture (may be None initially)
            previous_gesture = getattr(main, "previous_gesture", None)

            # Start by assuming we will keep whatever previous valid gesture we already have
            most_recent_gesture = previous_gesture

            # Extract the top gesture and see if it is valid. I think this try/exception is necessary to prevent
            # crashing if there isn't a valid gesture.
            candidate = None
            if gesture_result and getattr(gesture_result, "gestures", None):
                try:
                    candidate = gesture_result.gestures[0][0]
                except Exception:
                    candidate = None

            # If we were able to get a valid candidate gesture, then read and validate it.
            if candidate is not None:
                category_name = getattr(candidate, "category_name", None)
                score = getattr(candidate, "score", 0.0)

                # Check what type of gesture was detected and the confidence to determine whether we should use it.
                # At this point, we also need to ignore certain gestures based on our list.
                if ((category_name is not None) and (str(category_name) not in invalid_gestures_list) and 
                    (score >= confidence_threshold)):
                    main.previous_gesture = candidate
                    most_recent_gesture = candidate
                else:
                    # If the candidate is invalid, do NOT overwrite the previous valid gesture
                    most_recent_gesture = previous_gesture

            # Format gesture output text
            if ((most_recent_gesture is not None) and
                (getattr(most_recent_gesture, "category_name", None) not in invalid_gestures_list)):

                # Format score and add to output text with gesture
                score = getattr(most_recent_gesture, "score", 0.0)
                gesture_text = f"{most_recent_gesture.category_name} ({score:.2f})"
            else:
                # At startup, we will be waiting for the first operation gesture from the user.
                gesture_text = "Waiting for operation"

            # Get the previous gesture name, if there is one
            remembered_name = None
            if (getattr(main, "previous_gesture", None) is not None):
                remembered_name = getattr(getattr(main, "previous_gesture", None), "category_name", None)

            # Addition - "Thumb_Up"
            # Subtraction - "Thumb_Down"
            # Multiplication - "Closed_Fist"
            # Division - "ILoveYou", the Spiderman gesture
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

            # Display gesture result and the corresponding operation, if there is one
            cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
            cv2.putText(frame, gesture_text,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            if (operation_text is not None):
                cv2.putText(frame, f"{operation_text}",
                            (350, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

            #-----------------------------------------------------------------------------------------------------
            # Calculator State Machine
            #
            # State Definitions:
            # 1 - Wait for first number. If stable for stable_required_seconds, accept first_number input.
            # 2 - Wait for operation gesture. If stable for stable_required_seconds, accept operation input.
            # 3 - Wait for second number. If stable for stable_required_seconds, accept second_number input.
            # 4 - Show calculation result until 'r' is pressed, then reset to state 1

            # Current candidate values
            candidate_number = (left_fingers + right_fingers)  # fingers held up between left and right hands, 0-10
            candidate_gesture_name = None
            if (most_recent_gesture is not None) and (getattr(most_recent_gesture, "category_name", None) not in invalid_gestures_list):
                candidate_gesture_name = getattr(most_recent_gesture, "category_name", None)

            # Save the time before running the state machine for this frame
            now = time.time()

            # State 1: Get first number input
            if (state == 1):
                # Display prompt
                cv2.putText(frame, f"Enter the first operand",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
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
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Initialize stable value if it's currently None
                if ((stable_value is None) and (candidate_gesture_name is not None)):
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
                cv2.putText(frame, f"Enter the second operand",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
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
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Press 'r' to restart the calculator.",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Perform the desired calculation
                if "Thumb_Up" in operation:
                    result_value = (first_number + second_number)
                elif "Thumb_Down" in operation:
                    result_value = (first_number - second_number)
                elif "Closed_Fist" in operation:
                    result_value = (first_number * second_number)
                elif "ILoveYou" in operation:
                    if second_number == 0:
                        result_value = "Divide by zero"
                    else:
                        result_value = round(first_number / second_number, 4)
                else:
                    result_value = "ERROR"

                # Map the gesture names to operation symbols for displaying them
                gesture_symbol_map = {"Thumb_Up": "+", "Thumb_Down": "-", "Closed_Fist": "*", "ILoveYou": "/"}
                
                # Display calculation with result on frame
                cv2.putText(frame, f"{first_number} {gesture_symbol_map.get(operation, operation)} {second_number} = {result_value}",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the current number and a countdown timer if in state 1 or 3
            if ((state == 1) or (state ==3)):
                # Show candidate number and countdown
                if (stable_start is not None):
                    remaining = (stable_required_seconds - (now - stable_start))
                else:
                    remaining = stable_required_seconds
                
                # Display info on frame
                cv2.putText(frame, f"Input: {candidate_number}  Timer: {remaining:.1f} sec",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
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
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
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

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()

# Run the main loop
main()
