import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def count_fingers(hand_landmarks, handedness_label, image_width, image_height):

    # Indices of relevant landmarks in MediaPipe's hand model
    lm = hand_landmarks.landmark
    THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
    FINGER_TIPS = [8, 12, 16, 20] # Index, Middle, Ring, Pinky fingertips
    FINGER_PIPS = [6, 10, 14, 18] # Corresponding knuckles (PIP joints)

    # Initialize finger count as 0
    fingers = 0
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    hand_w = (max(xs) - min(xs)) if xs else 0.0
    hand_h = (max(ys) - min(ys)) if ys else 0.0
    scale = max(hand_w, hand_h) or 1e-6
    margin_y = 0.04 * scale         # vertical separation for original rule
    margin_d = 0.06 * scale         # radial separation (helps closed fist)

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def angle(a, b, c):
        # returns angle ABC in degrees
        ux, uy = a.x - b.x, a.y - b.y
        vx, vy = c.x - b.x, c.y - b.y
        du = math.hypot(ux, uy) or 1e-6
        dv = math.hypot(vx, vy) or 1e-6
        cosang = (ux * vx + uy * vy) / (du * dv)
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    wrist = lm[0]

    # Count other fingers (index, middle, ring, pinky)
    # A finger is considered "raised" if its tip landmark is above its PIP joint
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        # Original palm-friendly criterion
        tip_up = lm[tip_idx].y < (lm[pip_idx].y - margin_y) # y decreases as we go up
        # View-invariant backup: finger is straight and radially extended
        mcp_idx = pip_idx - 1  # MCP is one index before the PIP for these fingers
        straight = angle(lm[mcp_idx], lm[pip_idx], lm[tip_idx]) > 160.0
        radial = dist(wrist, lm[tip_idx]) > dist(wrist, lm[pip_idx]) + margin_d

        if (tip_up and radial) or (straight and radial):
            fingers += 1

    # Thumb logic
    # The thumb moves sideways, so we compare x-coordinates instead of y.
    # View-invariant: straight + radially farther from wrist than its IP (prevents fist=1)
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
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("webcam err")

    # Set video resolution to 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MediaPipe Hands model
    with mp_hands.Hands(
        static_image_mode=False,      # Use video stream (not static images)
        max_num_hands=2,              # Detect up to 2 hands
        model_complexity=1,           # Default complexity
        min_detection_confidence=0.6, # Minimum confidence for detection
        min_tracking_confidence=0.6,  # Minimum confidence for tracking
    ) as hands:
        
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

            # Display information box in top-left corner
            # Note that the text colors are specified in BGR format
            cv2.rectangle(frame, (10, 10), (380, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Left Hand: {left_fingers}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Right Hand: {right_fingers}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)

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
