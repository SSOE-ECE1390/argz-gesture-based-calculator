import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# --- Configuration ---
# You must still maintain the data/Add/ and data/Negative/ structure.
# Change path to data directory on your computer
DATA_DIR = Path('/Users/allen/Desktop/argz-gesture-based-calculator/src/argz-gesture-based-calculator/data')
MODEL_PATH = 'gesture_classifier_model.pkl'

# --- Index Finger Landmark Indices (5, 6, 7, 8) ---
# 5: Index finger base (MCP), 6: PIP, 7: DIP, 8: Tip
INDEX_FINGER_LANDMARKS = [5, 6, 7, 8] 

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # <--- CHANGED: Now detects up to two hands
    min_detection_confidence=0.5
)

def extract_index_finger_features_single(hand_landmarks):
    """
    Extracts and normalizes the 4 index finger landmarks relative to the wrist (landmark 0) 
    for a single hand.
    
    Returns: A 12-element numpy array.
    """
    # 1. Extract wrist landmark (index 0) for normalization
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    normalized_features = []
    
    # 2. Iterate ONLY through the index finger landmarks and normalize
    for i in INDEX_FINGER_LANDMARKS:
        landmark = hand_landmarks.landmark[i]
        
        # Calculate coordinates relative to the wrist
        rel_x = landmark.x - wrist_x
        rel_y = landmark.y - wrist_y
        rel_z = landmark.z - wrist_z
        
        normalized_features.extend([rel_x, rel_y, rel_z])

    return np.array(normalized_features)

def extract_and_normalize_landmarks(image_path):
    """
    Extracts features for up to two hands.
    If only one hand is found, the features for the second hand are padded with zeros.

    Returns:
        np.array or None: A flattened 24-element feature vector (2 hands * 4 landmarks * 3 coords)
                          if at least one hand is detected, otherwise None.
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image at {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        all_features = []
        num_hands_detected = 0

        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)
            
            # Sort hands to maintain consistency (e.g., left hand features first, then right hand)
            # MediaPipe results already include handedness, but sorting by hand type (left/right) 
            # requires slightly more complex logic. For simplicity, we process them in the 
            # order MediaPipe returns them, which is usually consistent.
            
            for i in range(min(2, num_hands_detected)):
                hand_landmarks = results.multi_hand_landmarks[i]
                features = extract_index_finger_features_single(hand_landmarks)
                all_features.append(features)
        
        if num_hands_detected == 0:
            return None
        
        # Padding: If only one hand is detected, append 12 zeros for the missing hand's index finger
        if num_hands_detected == 1:
            # 4 landmarks * 3 coordinates = 12 zero padding elements
            padding = np.zeros(12, dtype=np.float32) 
            all_features.append(padding)

        # Concatenate features into a single 24-element vector (12 features per hand)
        return np.concatenate(all_features)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_data(data_dir):
    """
    Loads features and labels from all subdirectories within the data_dir.
    (This function remains the same as it correctly iterates over directories)
    """
    features = []
    labels = []
    
    # Get all gesture folders (subdirectories)
    gesture_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not gesture_dirs:
        print(f"Error: No gesture directories found in {data_dir}. Please create folders like 'Add', 'Negative', etc.")
        return features, labels, []

    # Map folder names (labels) to integers
    label_map = {name.name: i for i, name in enumerate(gesture_dirs)}
    print(f"Loading data with Label Map: {label_map}")

    for gesture_dir in gesture_dirs:
        gesture_name = gesture_dir.name
        label_id = label_map[gesture_name]
        
        print(f"--- Processing {gesture_name} ---")
        
        # Iterate over all images (using a common set of extensions)
        for ext in ['*.jp*g', '*.png']:
            for image_path in gesture_dir.glob(ext): 
                feature_vector = extract_and_normalize_landmarks(image_path)
                
                if feature_vector is not None and feature_vector.size == 24: # Check size for safety
                    features.append(feature_vector)
                    labels.append(label_id)
                else:
                    # Note: This will print if no hand is detected, or if the feature size is unexpected
                    print(f"Skipping {image_path}: No hand detected or feature size incorrect.")
    
    return np.array(features), np.array(labels), label_map

def main():
    """
    Main function to load data, train the classifier, and save the model.
    """
    print("Starting dual index finger gesture classifier training...")

    # 1. Load Data
    X, y, label_map = load_data(DATA_DIR)

    if len(X) == 0:
        print("Training aborted: No feature data collected. Check data folder structure and image quality.")
        return

    if len(np.unique(y)) < 2:
        print("Training aborted: A classification model requires at least two distinct classes/gestures.")
        print(f"Found only {len(np.unique(y))} unique label(s).")
        return

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTotal samples loaded: {len(X)} (Features are 24 elements long, representing both index fingers.)")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


    # 3. Train Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 5. Save Model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'label_map': label_map}, f)
    
    print(f"\nSuccessfully trained and saved model to {MODEL_PATH}")

if __name__ == '__main__':
    main()

hands.close()
