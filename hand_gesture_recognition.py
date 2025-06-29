import cv2
import mediapipe as mp
import numpy as np
import math
import time

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition variables
        self.gesture_buffer = []
        self.buffer_size = 10
        self.current_gesture = "None"
        self.gesture_confidence = 0.0
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        # Vector from point2 to point1
        v1 = np.array([point1.x - point2.x, point1.y - point2.y])
        # Vector from point2 to point3
        v2 = np.array([point3.x - point2.x, point3.y - point2.y])
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip, finger_mcp):
        """Check if a finger is extended based on landmarks"""
        tip = landmarks[finger_tip]
        pip = landmarks[finger_pip]
        mcp = landmarks[finger_mcp]
        
        # For thumb, use different logic
        if finger_tip == 4:  # Thumb
            return tip.x > pip.x if landmarks[0].x < landmarks[17].x else tip.x < pip.x
        else:
            return tip.y < pip.y  # Other fingers point upward when extended
    
    def recognize_gesture(self, landmarks):
        """Recognize hand gesture based on landmarks"""
        if not landmarks:
            return "None", 0.0
        
        # Finger tip and joint indices
        fingers = {
            'thumb': (4, 3, 2),
            'index': (8, 6, 5),
            'middle': (12, 10, 9),
            'ring': (16, 14, 13),
            'pinky': (20, 18, 17)
        }
        
        # Check which fingers are extended
        extended_fingers = []
        for finger_name, (tip, pip, mcp) in fingers.items():
            if self.is_finger_extended(landmarks, tip, pip, mcp):
                extended_fingers.append(finger_name)
        
        # Gesture recognition logic
        num_extended = len(extended_fingers)
        
        if num_extended == 0:
            return "Fist", 0.9
        elif num_extended == 1:
            if 'index' in extended_fingers:
                return "Point", 0.9
            elif 'thumb' in extended_fingers:
                return "Thumbs Up", 0.9
        elif num_extended == 2:
            if 'index' in extended_fingers and 'middle' in extended_fingers:
                return "Peace", 0.9
            elif 'thumb' in extended_fingers and 'pinky' in extended_fingers:
                return "Rock On", 0.9
        elif num_extended == 5:
            return "Open Palm", 0.9
        elif num_extended == 3:
            if all(f in extended_fingers for f in ['index', 'middle', 'ring']):
                return "Three", 0.8
        elif num_extended == 4:
            if 'thumb' not in extended_fingers:
                return "Four", 0.8
        
        return "Unknown", 0.3
    
    def smooth_gesture(self, gesture, confidence):
        """Smooth gesture recognition using a buffer"""
        self.gesture_buffer.append((gesture, confidence))
        
        # Keep buffer size manageable
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        
        # Find most common gesture in buffer
        gesture_counts = {}
        total_confidence = 0
        
        for g, c in self.gesture_buffer:
            if g not in gesture_counts:
                gesture_counts[g] = []
            gesture_counts[g].append(c)
            total_confidence += c
        
        # Get most frequent gesture with highest average confidence
        best_gesture = "None"
        best_score = 0
        
        for g, confidences in gesture_counts.items():
            avg_confidence = sum(confidences) / len(confidences)
            frequency_score = len(confidences) / len(self.gesture_buffer)
            combined_score = avg_confidence * frequency_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_gesture = g
        
        return best_gesture, best_score
    
    def draw_landmarks_and_info(self, image, results):
        """Draw hand landmarks and gesture information"""
        height, width, _ = image.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Recognize gesture
                gesture, confidence = self.recognize_gesture(hand_landmarks.landmark)
                smoothed_gesture, smoothed_confidence = self.smooth_gesture(gesture, confidence)
                
                self.current_gesture = smoothed_gesture
                self.gesture_confidence = smoothed_confidence
        
        # Draw gesture information
        self.draw_ui(image)
    
    def draw_ui(self, image):
        """Draw user interface elements"""
        height, width, _ = image.shape
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Gesture text
        gesture_text = f"Gesture: {self.current_gesture}"
        confidence_text = f"Confidence: {self.gesture_confidence:.2f}"
        
        cv2.putText(image, gesture_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, confidence_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(image, "Press 'q' to quit, 'r' to reset", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Confidence bar
        bar_width = int(300 * self.gesture_confidence)
        cv2.rectangle(image, (width - 320, 20), (width - 20, 40), (50, 50, 50), -1)
        cv2.rectangle(image, (width - 320, 20), (width - 320 + bar_width, 40), 
                     (0, 255, 0) if self.gesture_confidence > 0.7 else (0, 255, 255), -1)
        cv2.putText(image, "Confidence", (width - 320, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Gesture Recognition Started!")
        print("Supported gestures: Fist, Point, Thumbs Up, Peace, Rock On, Open Palm, Three, Four")
        print("Press 'q' to quit, 'r' to reset gesture buffer")
        
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Draw landmarks and recognize gestures
            self.draw_landmarks_and_info(frame, results)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {current_fps}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.gesture_buffer.clear()
                print("Gesture buffer reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the hand gesture recognizer"""
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required libraries:")
        print("pip install opencv-python mediapipe numpy")

if __name__ == "__main__":
    main()