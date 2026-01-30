import cv2
import mediapipe as mp
import time
from collections import deque

# ===================== ASSETS ===================== #

ASSETS = {
    "shock": "assets/shock.jpeg",
    "tongue": "assets/cat-tongue.jpeg",
    "squint": "assets/squint.jpeg",

    "thumbsup": "assets/thumbsup.jpeg",
    "punch": "assets/punch.jpeg",
    "gimme": "assets/gimme.jpeg",
    "cute_hand": "assets/cute-hand.jpeg",
}

# ===================== CONFIG ===================== #

CONFIRM_TIME = 1.2
FILTER_ALPHA = 0.95
SMOOTHING = 7

# Thresholds
EYE_OPEN_SHOCK = 0.035
EYE_SQUINT = 0.017
MOUTH_OPEN = 0.032

# PERFORMANCE TUNING
PROCESS_EVERY_N_FRAMES = 3     # FaceMesh every 3 frames
HAND_EVERY_N_FRAMES = 6        # Hands every 6 frames

# ===================== MEDIAPIPE ===================== #

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,   # IMPORTANT: saves time
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===================== CAMERA ===================== #

cam = cv2.VideoCapture(0)

# ↓↓↓ BIG PERFORMANCE WIN ↓↓↓
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW = "Cat Filter"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ===================== CACHE IMAGES ===================== #

CACHED_IMAGES = {
    k: cv2.imread(v) for k, v in ASSETS.items()
}

# ===================== STATE ===================== #

history = deque(maxlen=SMOOTHING)
candidate = None
candidate_start = None
active_filter = None

frame_count = 0
last_detected = "neutral"

# ===================== UTILS ===================== #

def dy(lm, a, b):
    return abs(lm[a].y - lm[b].y)

def eye_open(lm):
    return (dy(lm, 159, 145) + dy(lm, 386, 374)) / 2

def mouth_open(lm):
    return dy(lm, 13, 14)

def full_screen(frame, img):
    img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(img, FILTER_ALPHA, frame, 1 - FILTER_ALPHA, 0)

# ===================== FACE ===================== #

def face_expression(lm):
    eye = eye_open(lm)
    mouth = mouth_open(lm)

    if eye > EYE_OPEN_SHOCK:
        return "shock"
    if mouth > MOUTH_OPEN:
        return "tongue"
    if eye < EYE_SQUINT:
        return "squint"

    return "neutral"

# ===================== HAND ===================== #

def hand_expression(hand_landmarks):
    if not hand_landmarks:
        return None

    hand = hand_landmarks[0]
    thumb = hand.landmark[4]
    index = hand.landmark[8]
    wrist = hand.landmark[0]

    if thumb.y < wrist.y:
        return "thumbsup"
    if index.y > wrist.y:
        return "punch"
    if abs(index.x - wrist.x) < 0.05:
        return "gimme"

    return "cute_hand"

# ===================== MAIN LOOP ===================== #

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # -------- DETECTION (SKIPPED FRAMES) -------- #

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res = face_mesh.process(rgb)
            detected = "neutral"

            if face_res.multi_face_landmarks:
                lm = face_res.multi_face_landmarks[0].landmark
                detected = face_expression(lm)

            # Hand detection (even less often)
            if frame_count % HAND_EVERY_N_FRAMES == 0:
                hand_res = hands.process(rgb)
                hand_detected = hand_expression(hand_res.multi_hand_landmarks)
                if hand_detected:
                    detected = hand_detected

            last_detected = detected

        else:
            detected = last_detected

        # -------- STATE LOGIC -------- #

        history.append(detected)
        stable = max(set(history), key=history.count)

        now = time.time()

        if stable != "neutral":
            if candidate != stable:
                candidate = stable
                candidate_start = now
            elif now - candidate_start >= CONFIRM_TIME:
                active_filter = stable
        else:
            candidate = None
            active_filter = None

        # -------- RENDER -------- #

        if active_filter:
            img = CACHED_IMAGES.get(active_filter)
            if img is not None:
                frame = full_screen(frame, img)

        cv2.imshow(WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
