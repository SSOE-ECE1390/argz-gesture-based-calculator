#GitHub repos used for code:
# - https://github.com/Mordekai66/finger-counter-mediapipe/blob/main/main.py
# - https://github.com/HarshitDolu/Finger-Counter-using-mediapipe/blob/main/Finger_counter.py
# - https://github.com/Sousannah/hand-tracking-using-mediapipe/blob/main/hand_tracking.py
# - https://github.com/Real-J/Finger-Counting-with-OpenCV-and-MediaPipe/blob/main/finger_counting.py

#Mediapipe docs
# - https://mediapipe-studio.webapps.google.com/studio/demo/gesture_recognizer

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, urllib.request

# Download default gesture model if it doesn't exist
if not os.path.exists("gesture_recognizer.task"):
    print("Downloading default MediaPipe gesture_recognizer.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    urllib.request.urlretrieve(url, "gesture_recognizer.task")
    print("Model downloaded successfully.")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks, handedness_label, image_width, image_height):
    """
    Count how many fingers are raised on a given hand using landmark positions.

    Args:
        hand_landmarks: The 21 hand landmarks detected by MediaPipe.
        handedness_label: 'Left' or 'Right' (determined by MediaPipe).
        image_width, image_height: Dimensions of the current video frame.

    Returns:
        The number of fingers currently detected as being raised (0-5).
    """

    # Indices of relevant landmarks in MediaPipe's hand model
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20] # Index, Middle, Ring, Pinky fingertips
    FINGER_PIPS = [6, 10, 14, 18] # Corresponding knuckles (PIP joints)

    # Initialize finger count as 0
    fingers = 0

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

    return fingers

# Main loop for live hand tracking and finger counting
def main():
    
    # Initialize webcam
    # ** FOR NON-MAC CHANGE to cv2.VideoCapture(0)
    # ** WINDOWS SPECIFIC is cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # ** MAC uses cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("webcam err")

    # Set video resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Gesture recognizer setup    
    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    gesture_result = ""
    most_recent_gesture = ""

    # Initialize previous gesture to None before loop
    main.previous_gesture = None

    # Initialize MediaPipe Hands model
    with mp_hands.Hands(
        static_image_mode=False,      # Use video stream (not static images)
        max_num_hands=2,              # Detect up to 2 hands
        model_complexity=1,           # Default complexity
        min_detection_confidence=0.6, # Minimum confidence for detection
        min_tracking_confidence=0.6,  # Minimum confidence for tracking
    ) as hands:
        
        #-----------------------------------------------------------------------------------------------------
        # Main loop

        while True:
            # Read a frame from the webcam
            ok, frame = cap.read()
            if not ok:
                break

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB (MediaPipe expects RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            #--------------------------------------------------------------------------------------------
            # Finger counting

            # Initialize finger counts for each hand
            left_fingers = 0
            right_fingers = 0

            # If hands are detected, process each one by classifying as left/right and counting fingers.
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    fingers_up = count_fingers(hand_lms, label, frame.shape[1], frame.shape[0])
                    
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

            # Display status information box in top-left corner with lines 1 and 2
            # Note that the text colors are specified in BGR format
            cv2.rectangle(frame, (10, 10), (380, 160), (0, 0, 0), -1)
            cv2.putText(frame, f"Left Hand: {left_fingers}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Right Hand: {right_fingers}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #-----------------------------------------------------------------------------------------------------
            # Gesture recognition

            # Run gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            gesture_result = recognizer.recognize(mp_image)

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

            # Display line 3
            cv2.putText(frame, gesture_text,
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            #-----------------------------------------------------------------------------------------------------
            # Calculator
            
            # Perform all calculations using the current finger counts
            sum_fingers = left_fingers + right_fingers
            product_fingers = left_fingers * right_fingers
            difference_fingers = left_fingers - right_fingers
            quotient_fingers = round(left_fingers / right_fingers, 2) if right_fingers != 0 else 0.0

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
            else:
                operation_text = None
                operation_result_text = None

            # If there is an updated gesture, display it (lines 4 and 5)
            if operation_text is not None and operation_result_text is not None:
                cv2.putText(frame, f"{operation_text}",
                            (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Result: {operation_result_text}",
                            (20, 210), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 255), 2, cv2.LINE_AA)
            
            #-----------------------------------------------------------------------------------------------------

            # Show the processed frame in window
            cv2.imshow("Finger Counter", frame)

            # Exit loop when 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()

# Run the main loop
main()







