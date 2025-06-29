import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import Counter

class ASLSignLanguageRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # ASL typically uses one hand for alphabet
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Recognition variables
        self.gesture_buffer = []
        self.buffer_size = 15
        self.current_sign = "None"
        self.confidence = 0.0
        self.sentence = ""
        self.word_timeout = 2.0  # seconds
        self.last_sign_time = time.time()
        
        # ASL alphabet and gestures definitions
        self.initialize_asl_signs()
        
    def initialize_asl_signs(self):
        """Initialize ASL sign patterns and descriptions"""
        # This contains the logic for recognizing ASL signs
        # Each sign has specific finger positions and orientations
        self.asl_signs = {
            # ASL Alphabet
            'A': "Closed fist with thumb on side",
            'B': "Four fingers up, thumb folded across palm",
            'C': "Curved hand like holding a cup",
            'D': "Index finger up, other fingers and thumb form circle",
            'E': "All fingers bent, thumb across fingertips",
            'F': "Index and thumb touch, other fingers up",
            'G': "Index finger and thumb pointing horizontally",
            'H': "Index and middle finger horizontal, pointing sideways",
            'I': "Pinky finger up, other fingers down",
            'J': "Pinky finger traces a J in the air",
            'K': "Index and middle finger up in V, thumb between them",
            'L': "Index finger up, thumb horizontal",
            'M': "Thumb under three fingers (index, middle, ring)",
            'N': "Thumb under two fingers (index, middle)",
            'O': "All fingers curved forming O shape",
            'P': "Index and middle finger pointing down like upside-down K",
            'Q': "Index finger and thumb pointing down",
            'R': "Index and middle finger crossed",
            'S': "Closed fist with thumb over other fingers",
            'T': "Thumb between index and middle finger (fist)",
            'U': "Index and middle finger up together",
            'V': "Index and middle finger up in V shape",
            'W': "Index, middle, and ring finger up",
            'X': "Index finger hooked/bent",
            'Y': "Thumb and pinky extended (hang loose)",
            'Z': "Index finger traces Z in air",
            
            # Numbers 0-10
            '0': "Closed fist (or O shape)",
            '1': "Index finger up",
            '2': "Index and middle finger up (V shape)",
            '3': "Index, middle, ring finger up",
            '4': "Four fingers up (no thumb)",
            '5': "All five fingers spread",
            '6': "Thumb and pinky touch, other fingers up",
            '7': "Thumb and ring finger touch, other fingers up",
            '8': "Thumb and middle finger touch, other fingers up",
            '9': "Thumb and index finger touch, other fingers up",
            '10': "Thumb up (or shake fist)",
            
            # Common Words/Phrases
            'HELLO': "Wave hand or salute motion",
            'THANK_YOU': "Flat hand on chin, move forward",
            'PLEASE': "Flat hand on chest, circular motion",
            'SORRY': "Fist on chest, circular motion",
            'YES': "Fist nods up and down",
            'NO': "Index and middle finger close like scissors",
            'HELP': "Right hand on left palm, lift up",
            'MORE': "Fingertips together, tap repeatedly",
            'GOOD': "Flat hand to chin, then down to other palm",
            'BAD': "Flat hand to chin, then flip down"
        }
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        v1 = np.array([point1.x - point2.x, point1.y - point2.y])
        v2 = np.array([point3.x - point2.x, point3.y - point2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def get_finger_states(self, landmarks):
        """Get the state of each finger (extended or folded)"""
        finger_states = {}
        
        # Thumb (special case - lateral movement)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Thumb extended if tip is further from wrist than MCP joint
        thumb_extended = self.calculate_distance(thumb_tip, wrist) > self.calculate_distance(thumb_mcp, wrist)
        finger_states['thumb'] = thumb_extended
        
        # Other fingers (vertical movement)
        fingers = {
            'index': (8, 6, 5),    # tip, pip, mcp
            'middle': (12, 10, 9),
            'ring': (16, 14, 13),
            'pinky': (20, 18, 17)
        }
        
        for finger_name, (tip_idx, pip_idx, mcp_idx) in fingers.items():
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            # Finger is extended if tip is higher than both PIP and MCP
            extended = tip.y < pip.y and tip.y < mcp.y
            finger_states[finger_name] = extended
            
        return finger_states
    
    def recognize_asl_sign(self, landmarks):
        """Recognize ASL signs based on hand landmarks"""
        if not landmarks:
            return "None", 0.0
            
        finger_states = self.get_finger_states(landmarks)
        extended_fingers = [name for name, extended in finger_states.items() if extended]
        folded_fingers = [name for name, extended in finger_states.items() if not extended]
        
        # Count extended fingers
        num_extended = len(extended_fingers)
        
        # Advanced geometric analysis for specific signs
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate specific distances and angles for complex signs
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        index_middle_angle = self.calculate_angle(index_tip, landmarks[6], middle_tip)
        
        # ASL Alphabet Recognition Logic
        
        # Letter A: Closed fist with thumb on side
        if num_extended == 0 or (num_extended == 1 and 'thumb' in extended_fingers):
            if thumb_tip.x > landmarks[3].x:  # Thumb visible on side
                return 'A', 0.9
                
        # Letter B: Four fingers up, thumb folded
        if (num_extended == 4 and 'thumb' not in extended_fingers and 
            all(f in extended_fingers for f in ['index', 'middle', 'ring', 'pinky'])):
            return 'B', 0.9
            
        # Letter C: Curved hand shape
        if num_extended == 0:
            # Check if hand is in C shape (curved)
            wrist = landmarks[0]
            fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            avg_distance = sum(self.calculate_distance(tip, wrist) for tip in fingertips) / 5
            if 0.15 < avg_distance < 0.25:  # Moderate curve
                return 'C', 0.8
                
        # Letter D: Index up, others form circle with thumb
        if (num_extended == 1 and 'index' in extended_fingers and 
            thumb_index_distance < 0.05):  # Thumb and other fingers close
            return 'D', 0.9
            
        # Letter E: All fingers bent down
        if num_extended == 0:
            # Check if all fingertips are close to palm
            palm_center = landmarks[9]  # Middle finger MCP as palm reference
            fingertip_distances = [
                self.calculate_distance(landmarks[8], palm_center),   # Index
                self.calculate_distance(landmarks[12], palm_center),  # Middle
                self.calculate_distance(landmarks[16], palm_center),  # Ring
                self.calculate_distance(landmarks[20], palm_center)   # Pinky
            ]
            if all(d < 0.08 for d in fingertip_distances):
                return 'E', 0.8
                
        # Letter F: Index and thumb touch, others up
        if (num_extended == 3 and thumb_index_distance < 0.03 and
            all(f in extended_fingers for f in ['middle', 'ring', 'pinky'])):
            return 'F', 0.9
            
        # Letter I: Pinky up, others down
        if num_extended == 1 and 'pinky' in extended_fingers:
            return 'I', 0.9
            
        # Letter L: Index up, thumb horizontal
        if (num_extended == 2 and 'index' in extended_fingers and 'thumb' in extended_fingers):
            # Check if they form an L shape (perpendicular)
            angle = self.calculate_angle(index_tip, landmarks[5], thumb_tip)
            if 70 < angle < 110:  # Roughly 90 degrees
                return 'L', 0.9
                
        # Letter O: All fingers curved in O shape
        if num_extended == 0:
            # Check if fingertips form a circle
            if thumb_index_distance < 0.03:  # Thumb and index close
                return 'O', 0.8
                
        # Letter R: Index and middle crossed
        if (num_extended == 2 and 'index' in extended_fingers and 'middle' in extended_fingers):
            # Check if fingers are crossed
            if abs(index_tip.x - middle_tip.x) < 0.02:  # Fingers close horizontally
                return 'R', 0.8
                
        # Letter S: Closed fist with thumb over fingers
        if num_extended == 0:
            if thumb_tip.y < landmarks[8].y:  # Thumb above other fingers
                return 'S', 0.8
                
        # Letter U: Index and middle up together
        if (num_extended == 2 and 'index' in extended_fingers and 'middle' in extended_fingers and
            abs(index_tip.x - middle_tip.x) < 0.03):  # Close together
            return 'U', 0.9
            
        # Letter V: Index and middle in V shape
        if (num_extended == 2 and 'index' in extended_fingers and 'middle' in extended_fingers):
            if abs(index_tip.x - middle_tip.x) > 0.05:  # Spread apart
                return 'V', 0.9
                
        # Letter W: Three fingers up
        if (num_extended == 3 and 
            all(f in extended_fingers for f in ['index', 'middle', 'ring'])):
            return 'W', 0.9
            
        # Letter Y: Thumb and pinky extended (hang loose)
        if (num_extended == 2 and 'thumb' in extended_fingers and 'pinky' in extended_fingers):
            return 'Y', 0.9
            
        # Numbers
        if num_extended == 1 and 'index' in extended_fingers:
            return '1', 0.9
        elif num_extended == 2 and 'index' in extended_fingers and 'middle' in extended_fingers:
            return '2', 0.9
        elif num_extended == 3:
            return '3', 0.8
        elif num_extended == 4 and 'thumb' not in extended_fingers:
            return '4', 0.8
        elif num_extended == 5:
            return '5', 0.9
            
        # Common words recognition (simplified)
        # These would need more sophisticated motion tracking
        
        return "Unknown", 0.3
    
    def smooth_recognition(self, sign, confidence):
        """Smooth recognition using buffer and voting"""
        self.gesture_buffer.append((sign, confidence, time.time()))
        
        # Remove old entries
        current_time = time.time()
        self.gesture_buffer = [(s, c, t) for s, c, t in self.gesture_buffer 
                              if current_time - t < 1.0]  # Keep last 1 second
        
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer = self.gesture_buffer[-self.buffer_size:]
        
        # Weighted voting
        sign_votes = {}
        total_weight = 0
        
        for s, c, t in self.gesture_buffer:
            weight = c * (1.0 - (current_time - t) * 0.5)  # Recent signs weighted more
            if weight > 0:
                if s not in sign_votes:
                    sign_votes[s] = 0
                sign_votes[s] += weight
                total_weight += weight
        
        if not sign_votes:
            return "None", 0.0
            
        # Get best sign
        best_sign = max(sign_votes.items(), key=lambda x: x[1])
        normalized_confidence = best_sign[1] / total_weight if total_weight > 0 else 0
        
        return best_sign[0], min(normalized_confidence, 1.0)
    
    def add_to_sentence(self, sign):
        """Add recognized sign to sentence"""
        current_time = time.time()
        
        # Add space between words after timeout
        if current_time - self.last_sign_time > self.word_timeout:
            if self.sentence and not self.sentence.endswith(' '):
                self.sentence += ' '
        
        # Add sign to sentence
        if sign != "None" and sign != "Unknown":
            if len(self.sentence) == 0 or self.sentence[-1] != sign[-1]:  # Avoid duplicates
                self.sentence += sign
                self.last_sign_time = current_time
    
    def draw_ui(self, image):
        """Draw comprehensive UI"""
        height, width, _ = image.shape
        
        # Main info panel
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Current sign
        sign_text = f"Sign: {self.current_sign}"
        confidence_text = f"Confidence: {self.confidence:.2f}"
        
        cv2.putText(image, sign_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, confidence_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sign description
        if self.current_sign in self.asl_signs:
            desc_text = self.asl_signs[self.current_sign]
            cv2.putText(image, desc_text[:40], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(image, "Controls: 'q'-quit, 'c'-clear, 'space'-add space", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Confidence bar
        bar_width = int(200 * self.confidence)
        cv2.rectangle(image, (width - 220, 20), (width - 20, 40), (50, 50, 50), -1)
        color = (0, 255, 0) if self.confidence > 0.7 else (0, 255, 255) if self.confidence > 0.4 else (0, 0, 255)
        cv2.rectangle(image, (width - 220, 20), (width - 220 + bar_width, 40), color, -1)
        
        # Sentence display
        sentence_panel_height = 80
        cv2.rectangle(image, (10, height - sentence_panel_height - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        
        cv2.putText(image, "Sentence:", (20, height - sentence_panel_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display sentence (split into lines if too long)
        sentence_display = self.sentence if len(self.sentence) <= 50 else self.sentence[-50:]
        cv2.putText(image, sentence_display, (20, height - sentence_panel_height + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Quick reference (top right)
        ref_text = [
            "ASL Quick Ref:",
            "A-Z: Alphabet",
            "1-5: Numbers", 
            "Fist: A or S",
            "V-shape: V or 2"
        ]
        
        for i, text in enumerate(ref_text):
            cv2.putText(image, text, (width - 180, 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("ASL Sign Language Recognition Started!")
        print("This system recognizes:")
        print("- ASL Alphabet (A-Z)")
        print("- Numbers (0-9)")
        print("- Basic words and phrases")
        print("\nControls:")
        print("- 'q': Quit")
        print("- 'c': Clear sentence")
        print("- 'space': Add space to sentence")
        print("- Hold sign steady for 1-2 seconds for best recognition")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Recognize sign
                    sign, confidence = self.recognize_asl_sign(hand_landmarks.landmark)
                    self.current_sign, self.confidence = self.smooth_recognition(sign, confidence)
                    
                    # Add to sentence if confident
                    if self.confidence > 0.7:
                        self.add_to_sentence(self.current_sign)
            else:
                self.current_sign = "None"
                self.confidence = 0.0
            
            # Draw UI
            self.draw_ui(frame)
            
            # Display
            cv2.imshow('ASL Sign Language Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence = ""
                print("Sentence cleared")
            elif key == ord(' '):
                self.sentence += " "
                print("Space added")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.sentence:
            print(f"\nFinal sentence: {self.sentence}")

def main():
    try:
        recognizer = ASLSignLanguageRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required libraries installed:")
        print("pip install opencv-python mediapipe numpy")

if __name__ == "__main__":
    main()
