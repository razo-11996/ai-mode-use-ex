import cv2
import sys
import os
import urllib.request
import mediapipe as mp

# =========================
# MediaPipe Initialization
# =========================

USE_NEW_API = False
hands = None
hand_landmarker = None
mp_hands = None
mp_drawing = None

def setup_mediapipe():
    global USE_NEW_API, hands, hand_landmarker, mp_hands, mp_drawing

    if hasattr(mp, "solutions"):
        # Old API
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        USE_NEW_API = False
        print("✅ Using MediaPipe Solutions API")

    elif hasattr(mp, "tasks"):
        # New API
        USE_NEW_API = True
        print("📥 Using MediaPipe Tasks API")

        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print("📥 Downloading model...")
            urllib.request.urlretrieve(url, model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1
        )
        hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    else:
        print("❌ Unsupported MediaPipe version")
        sys.exit(1)

setup_mediapipe()

# =========================
# Helper Functions
# =========================

def count_fingers(landmarks):
    """
    Simple heuristic:
    - Thumb: compare X axis
    - Other fingers: compare Y axis
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    count = 0

    # Thumb
    if landmarks[tips[0]].x > landmarks[pips[0]].x:
        count += 1

    # Other fingers
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[pips[i]].y:
            count += 1

    return count


def draw_landmarks_old(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )


def draw_landmarks_new(frame, landmarks):
    h, w = frame.shape[:2]

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    for a, b in connections:
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x,y), 4, (255,0,0), -1)

# =========================
# Camera Loop
# =========================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fingers = 0

    if USE_NEW_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks_new(frame, landmarks)
            fingers = count_fingers(landmarks)

    else:
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw_landmarks_old(frame, hand)
            fingers = count_fingers(hand.landmark)

    cv2.putText(frame, f"Fingers: {fingers}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if hands:
    hands.close()
cv2.destroyAllWindows()
