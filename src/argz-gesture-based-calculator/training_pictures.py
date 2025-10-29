import cv2
import mediapipe as mp
import time
import os

# Initialize MediaPipe Hands model
# max_num_hands=2 ensures we can track two hands (two index fingers)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    # Adjusted tracking confidence to improve persistence during occlusion.
    # Lowering this helps maintain the track when one hand partially obscures the other.
    min_tracking_confidence=0.3)

# Initialize MediaPipe Drawing utilities (not strictly used for drawing, but good practice to keep)
mp_drawing = mp.solutions.drawing_utils

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

# Define the landmark indices for the entire Index Finger:
# 5: MCP (base of the finger)
# 6: PIP (first joint)
# 7: DIP (second joint)
# 8: TIP (end of the finger)
INDEX_FINGER_IDS = [5, 6, 7, 8]

# --- Image Saving Configuration ---
SAVE_INTERVAL_SECONDS = 5.0
# Define the folder where pictures will be saved.
# You can change 'index_finger_captures' to your desired path.
OUTPUT_FOLDER = "index_finger_captures" 
last_save_time = time.time()

# Create the output folder if it doesn't exist
try:
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
except Exception as e:
    print(f"Error creating output directory: {e}")

print("Index Finger Detector is running. Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB before processing
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image)

    # Convert the image back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions for coordinate conversion
    image_height, image_width, _ = image.shape

    # Check if any hands were detected
    if results.multi_hand_landmarks:
        # Iterate through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Dictionary to store the pixel coordinates of the 4 index finger points
            index_points = {}
            
            # 1. Get Coordinates and Draw Joints (Circles)
            for i in INDEX_FINGER_IDS:
                # Retrieve the normalized point data
                point = hand_landmarks.landmark[i]
                
                # Convert normalized coordinates (0.0 to 1.0) to pixel values
                x = int(point.x * image_width)
                y = int(point.y * image_height)
                
                # Store the pixel coordinates for line drawing later
                index_points[i] = (x, y)
                
                # Draw a prominent green circle at each joint
                cv2.circle(image, (x, y), 10, (0, 255, 0), cv2.FILLED) # Green joints
            
            # 2. Draw the Connecting Segments (Lines)
            if len(index_points) == 4:
                # Segment 1: MCP (5) to PIP (6)
                cv2.line(image, index_points[5], index_points[6], (255, 0, 0), 5) # Blue line
                # Segment 2: PIP (6) to DIP (7)
                cv2.line(image, index_points[6], index_points[7], (255, 0, 0), 5) # Blue line
                # Segment 3: DIP (7) to TIP (8)
                cv2.line(image, index_points[7], index_points[8], (255, 0, 0), 5) # Blue line

    # --- Image Saving Logic ---
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
        try:
            # Create a unique filename using the current timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_FOLDER, f"capture_{timestamp}.jpg")
            
            # Save the current frame
            cv2.imwrite(filename, image)
            print(f"Saved image: {filename}")
            
            # Reset the timer
            last_save_time = current_time
        except Exception as e:
            print(f"Could not save image: {e}")


    # Display the resulting image
    cv2.imshow('MediaPipe Index Finger Detection (Full Finger)', image)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
