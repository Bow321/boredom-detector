import cv2
import mediapipe as mp

# Eye landmark indices
LEFT_EYE = [33, 133, 157, 158, 159, 160, 161, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]

def calculate_ear(landmarks, eye_indices, h, w):
    """Calculate Eye Aspect Ratio (EAR) for one eye."""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    
    vertical1 = abs(points[1][1] - points[5][1])
    vertical2 = abs(points[2][1] - points[4][1])
    horizontal = abs(points[0][0] - points[3][0])
    
    if horizontal == 0:
        return 0.5
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_head_pose(landmarks):
    """Estimate head pose based on nose and eye positions."""
    try:
        nose_x = landmarks[1].x
        left_eye_x = landmarks[33].x
        right_eye_x = landmarks[263].x
        
        face_center_x = (left_eye_x + right_eye_x) / 2
        nose_offset = (nose_x - face_center_x)
        
        if nose_offset < -0.03:
            return 'right'
        elif nose_offset > 0.03:
            return 'left'
        else:
            return 'center'
    except:
        return 'center'

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# Blink detection variables
blink_counter = 0
BLINK_THRESHOLD = 0.13
CONSECUTIVE_FRAMES = 2
last_blink_time = 0
blink_cooldown_frames = 15
current_frame = 0
blink_min_ear = 1.0  # Track lowest EAR during current blink

print("=" * 50)
print("BOREDOM DETECTOR - Blink Detection")
print(f"Threshold: {BLINK_THRESHOLD}")
print("Press 'q' to quit")
print("=" * 50)

with mp_face_mesh.FaceMesh() as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                
                # Calculate EAR
                left_ear = calculate_ear(landmarks.landmark, LEFT_EYE, h, w)
                right_ear = calculate_ear(landmarks.landmark, RIGHT_EYE, h, w)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Get head pose
                head_direction = get_head_pose(landmarks.landmark)
                
                # Draw all landmarks (small green dots)
                for lm in landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Draw eye landmarks (bigger red dots)
                for idx in LEFT_EYE + RIGHT_EYE:
                    lm = landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # BLINK DETECTION with lowest EAR tracking
                blink_detected = False
                
                if avg_ear < BLINK_THRESHOLD:
                    blink_counter += 1
                    # Track the lowest EAR during this blink
                    if avg_ear < blink_min_ear:
                        blink_min_ear = avg_ear
                else:
                    if blink_counter >= CONSECUTIVE_FRAMES and head_direction == 'center':
                        if current_frame - last_blink_time > blink_cooldown_frames:
                            blink_detected = True
                            last_blink_time = current_frame
                            # Print the TRUE lowest EAR during the blink
                            print(f"[BLINK] Lowest EAR: {blink_min_ear:.3f}")
                            blink_min_ear = 1.0  # Reset for next blink
                    blink_counter = 0
                
                # Visual feedback
                if blink_detected:
                    cv2.putText(frame, "BLINK!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Display info on screen
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Head: {head_direction}", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Threshold: {BLINK_THRESHOLD}", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
                
                if head_direction != 'center':
                    cv2.putText(frame, "LOOKING AWAY!", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        frame = cv2.flip(frame, 1)
        cv2.imshow('Boredom Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()