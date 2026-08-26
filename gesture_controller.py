import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time
import math
import ctypes
import threading
from sklearn.neighbors import KNeighborsClassifier
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

# --- Win32 API для управления курсором (без overhead) ---
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

def win32_move_mouse(x, y):
    user32.SetCursorPos(int(x), int(y))

def win32_lmb_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def win32_lmb_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def win32_rmb_click():
    user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

# --- Стабильный буфер сглаживания курсора (SMA) ---
class SmoothBuffer:
    def __init__(self, size=7):
        self.size = size
        self.history_x = []
        self.history_y = []

    def add(self, x, y):
        self.history_x.append(x)
        self.history_y.append(y)
        if len(self.history_x) > self.size:
            self.history_x.pop(0)
            self.history_y.pop(0)
        return np.mean(self.history_x), np.mean(self.history_y)

mouse_filter = SmoothBuffer(size=7)

# --- Асинхронный захват кадров камеры ---
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

# --- Системное аудио Windows ---
device = AudioUtilities.GetSpeakers()
volume_control = device.EndpointVolume

def toggle_system_mute():
    current_mute = volume_control.GetMute()
    volume_control.SetMute(not current_mute, None)
    print(f"[ACTION] Общий звук: {'MUTED' if not current_mute else 'ACTIVE'}")

cached_spotify_vol = None
last_spotify_check = 0

def get_spotify_volume_control():
    global cached_spotify_vol, last_spotify_check
    now = time.time()
    if cached_spotify_vol is not None and (now - last_spotify_check < 3.0):
        return cached_spotify_vol
    
    last_spotify_check = now
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        if session.Process and session.Process.name().lower() == "spotify.exe":
            cached_spotify_vol = session._ctl.QueryInterface(ISimpleAudioVolume)
            return cached_spotify_vol
    cached_spotify_vol = None
    return None

def set_spotify_volume(volume_scalar):
    vol = get_spotify_volume_control()
    if vol:
        try:
            vol.SetMasterVolume(volume_scalar, None)
            return True
        except Exception:
            return False
    return False

# --- MediaPipe Tasks API ---
MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

DATASET_FILE = "gestures_data.pkl"

if os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = {"data": [], "labels": []}

def train_classifier():
    labels_set = set(dataset["labels"])
    if len(labels_set) >= 2 and len(dataset["data"]) >= 20:
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
        knn.fit(dataset["data"], dataset["labels"])
        return knn
    return None

def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    features = []
    for lm in landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    features = np.array(features, dtype=np.float32)
    max_val = np.max(np.abs(features))
    return features / max_val if max_val > 0 else features

classifier = train_classifier()

threaded_cam = ThreadedCamera(src=0, width=640, height=480).start()
time.sleep(0.5)

MODES = ["AIR_MOUSE", "GESTURE_MUTE", "FADER_SYSTEM", "FADER_SPOTIFY"]
current_mode_idx = 0

current_stable_gesture = None
gesture_start_time = 0
HOLD_DURATION_TRIGGER = 0.25
action_executed = False

# Координаты мыши
prev_mouse_x, prev_mouse_y = SCREEN_W // 2, SCREEN_H // 2
MARGIN_X = 120
MARGIN_Y = 100
is_lmb_down = False
last_rmb_time = 0

# Фейдер
is_pinched = False
SMOOTHING = 0.35
current_vol = volume_control.GetMasterVolumeLevelScalar() * 100
FADER_TOP = 80
FADER_BOTTOM = 400

prev_time = time.time()
recording_class = None
last_sample_time = 0
SAMPLE_INTERVAL = 0.05

COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_SPOTIFY = (30, 215, 96)
COLOR_YELLOW = (0, 255, 255)
COLOR_BG_BOX = (25, 25, 25)
COLOR_BUTTON = (50, 50, 50)
COLOR_BUTTON_ACTIVE = (0, 180, 255)

BTN_RECT = [390, 12, 625, 48]

def on_mouse_click(event, x, y, flags, param):
    global current_mode_idx, is_pinched, is_lmb_down
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_RECT[0] <= x <= BTN_RECT[2] and BTN_RECT[1] <= y <= BTN_RECT[3]:
            current_mode_idx = (current_mode_idx + 1) % len(MODES)
            is_pinched = False
            if is_lmb_down:
                win32_lmb_up()
                is_lmb_down = False

cv2.namedWindow("Hand Vision - Gesture HUD")
cv2.setMouseCallback("Hand Vision - Gesture HUD", on_mouse_click)

AI_INPUT_SIZE = (256, 256)

while True:
    ret, frame = threaded_cam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1 / dt if dt > 0 else 60
    prev_time = curr_time

    frame_timestamp_ms = int(curr_time * 1000)

    # Ресайз для инференса MediaPipe
    small_frame = cv2.resize(frame, AI_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    detected_gesture = "NO HAND"
    confidence = 0.0
    raw_features = None

    active_mode = MODES[current_mode_idx]

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, landmark_points[p1], landmark_points[p2], COLOR_CYAN, 1)

        for idx, pt in enumerate(landmark_points):
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pt, 5, COLOR_RED, -1)
            else:
                cv2.circle(frame, pt, 3, COLOR_GREEN, -1)

        raw_features = normalize_landmarks(hand_landmarks)

        # 1. Мышь по стабильному суставу ладони (точка 5)
        if active_mode == "AIR_MOUSE":
            cv2.rectangle(frame, (MARGIN_X, MARGIN_Y), (w - MARGIN_X, h - MARGIN_Y), COLOR_YELLOW, 1)

            x_thumb, y_thumb = landmark_points[4]
            x_index_tip, y_index_tip = landmark_points[8]
            x_mid_tip, y_mid_tip = landmark_points[12]
            
            # Базовая стабильная точка — основание указательного пальца (MCP, 5)
            x_base, y_base = landmark_points[5]
            cv2.circle(frame, (x_base, y_base), 6, (0, 255, 255), -1)

            raw_screen_x = np.interp(x_base, [MARGIN_X, w - MARGIN_X], [0, SCREEN_W])
            raw_screen_y = np.interp(y_base, [MARGIN_Y, h - MARGIN_Y], [0, SCREEN_H])

            smooth_x, smooth_y = mouse_filter.add(raw_screen_x, raw_screen_y)

            # Deadzone порог 5px
            if math.hypot(smooth_x - prev_mouse_x, smooth_y - prev_mouse_y) > 5.0:
                win32_move_mouse(smooth_x, smooth_y)
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

            lmb_dist = math.hypot(x_index_tip - x_thumb, y_index_tip - y_thumb)
            rmb_dist = math.hypot(x_mid_tip - x_thumb, y_mid_tip - y_thumb)

            if lmb_dist < 30:
                cv2.circle(frame, (x_index_tip, y_index_tip), 9, COLOR_GREEN, -1)
                if not is_lmb_down:
                    win32_lmb_down()
                    is_lmb_down = True
                detected_gesture = "LMB / DRAG"
            else:
                if is_lmb_down:
                    win32_lmb_up()
                    is_lmb_down = False
                detected_gesture = "MOVING"

            if rmb_dist < 30:
                cv2.circle(frame, (x_mid_tip, y_mid_tip), 9, COLOR_MAGENTA, -1)
                if curr_time - last_rmb_time > 0.4:
                    win32_rmb_click()
                    last_rmb_time = curr_time
                    detected_gesture = "RIGHT CLICK"

        # 2. Mute
        elif active_mode == "GESTURE_MUTE":
            if is_lmb_down:
                win32_lmb_up()
                is_lmb_down = False

            if classifier is not None:
                distances, _ = classifier.kneighbors([raw_features], n_neighbors=1)
                min_dist = distances[0][0]
                if min_dist <= 0.45:
                    prediction = classifier.predict([raw_features])[0]
                    proba = np.max(classifier.predict_proba([raw_features]))
                    confidence = proba * 100
                    detected_gesture = prediction
                else:
                    detected_gesture = "UNKNOWN"
            else:
                detected_gesture = "NEED TRAINING (R)"

        # 3. Фейдеры
        elif active_mode in ["FADER_SYSTEM", "FADER_SPOTIFY"]:
            if is_lmb_down:
                win32_lmb_up()
                is_lmb_down = False

            x1, y1 = landmark_points[4]
            x2, y2 = landmark_points[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            pinch_dist = math.hypot(x2 - x1, y2 - y1)
            fader_color = COLOR_SPOTIFY if active_mode == "FADER_SPOTIFY" else COLOR_MAGENTA

            if pinch_dist < 35:
                is_pinched = True
                detected_gesture = "DRAGGING"
                target_vol = np.interp(cy, [FADER_TOP, FADER_BOTTOM], [100, 0])
                current_vol = (1 - SMOOTHING) * current_vol + SMOOTHING * target_vol
                clamped_vol = np.clip(current_vol, 0, 100)

                if active_mode == "FADER_SYSTEM":
                    volume_control.SetMasterVolumeLevelScalar(clamped_vol / 100.0, None)
                elif active_mode == "FADER_SPOTIFY":
                    set_spotify_volume(clamped_vol / 100.0)

                cv2.circle(frame, (cx, cy), 9, COLOR_GREEN, -1)
                cv2.line(frame, (cx, cy), (w - 60, cy), fader_color, 2)
            else:
                is_pinched = False
                detected_gesture = "PINCH TO GRAB"
                cv2.circle(frame, (cx, cy), 5, fader_color, 2)

            cv2.line(frame, (x1, y1), (x2, y2), fader_color, 2)

    else:
        if is_lmb_down:
            win32_lmb_up()
            is_lmb_down = False
        is_pinched = False

    # Клавиши
    key = cv2.waitKey(1) & 0xFF
    
    if key in [ord('m'), ord('M')]:
        current_mode_idx = (current_mode_idx + 1) % len(MODES)
        is_pinched = False
        if is_lmb_down:
            win32_lmb_up()
            is_lmb_down = False

    elif key in [ord('r'), ord('R')] and active_mode == "GESTURE_MUTE":
        recording_class = "MUTE_TOGGLE"
    elif key in [ord('n'), ord('N')] and active_mode == "GESTURE_MUTE":
        recording_class = "NONE"
    elif key in [ord('c'), ord('C')]:
        dataset = {"data": [], "labels": []}
        classifier = None
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)
    else:
        if recording_class is not None:
            classifier = train_classifier()
            with open(DATASET_FILE, "wb") as f:
                pickle.dump(dataset, f)
            recording_class = None

    if recording_class and raw_features is not None:
        if curr_time - last_sample_time > SAMPLE_INTERVAL:
            dataset["data"].append(raw_features)
            dataset["labels"].append(recording_class)
            last_sample_time = curr_time

    # Mute Action
    if active_mode == "GESTURE_MUTE" and detected_gesture == "MUTE_TOGGLE":
        if current_stable_gesture != "MUTE_TOGGLE":
            current_stable_gesture = "MUTE_TOGGLE"
            gesture_start_time = curr_time
            action_executed = False
        else:
            elapsed = curr_time - gesture_start_time
            progress = min(elapsed / HOLD_DURATION_TRIGGER, 1.0)
            bar_w = int(progress * 260)
            cv2.rectangle(frame, (20, h - 40), (280, h - 20), COLOR_BG_BOX, -1)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), COLOR_GREEN, -1)

            if elapsed >= HOLD_DURATION_TRIGGER and not action_executed:
                toggle_system_mute()
                action_executed = True
    else:
        current_stable_gesture = None
        action_executed = False

    # UI
    cv2.rectangle(frame, (BTN_RECT[0], BTN_RECT[1]), (BTN_RECT[2], BTN_RECT[3]), COLOR_BUTTON, -1)
    cv2.rectangle(frame, (BTN_RECT[0], BTN_RECT[1]), (BTN_RECT[2], BTN_RECT[3]), COLOR_BUTTON_ACTIVE, 2)
    cv2.putText(frame, f"MODE [M]: {active_mode}", (BTN_RECT[0] + 10, BTN_RECT[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if active_mode in ["FADER_SYSTEM", "FADER_SPOTIFY"]:
        bar_color = COLOR_SPOTIFY if active_mode == "FADER_SPOTIFY" else COLOR_GREEN
        vol_bar_h = np.interp(current_vol, [0, 100], [FADER_BOTTOM, FADER_TOP])
        cv2.rectangle(frame, (w - 60, FADER_TOP), (w - 30, FADER_BOTTOM), COLOR_BG_BOX, -1)
        cv2.rectangle(frame, (w - 60, int(vol_bar_h)), (w - 30, FADER_BOTTOM), bar_color, -1)
        cv2.rectangle(frame, (w - 60, FADER_TOP), (w - 30, FADER_BOTTOM), (150, 150, 150), 2)
        cv2.putText(frame, f"{int(current_vol)}%", (w - 75, FADER_BOTTOM + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

    mute_state = "MUTED" if volume_control.GetMute() else "UNMUTED"
    state_color = COLOR_RED if mute_state == "MUTED" else COLOR_GREEN
    
    cv2.rectangle(frame, (10, 10), (250, 100), COLOR_BG_BOX, -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Audio: {mute_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    cv2.putText(frame, f"Mode: {active_mode}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1)

    cv2.imshow("Hand Vision - Gesture HUD", frame)
    if key == 27:
        break

threaded_cam.stop()
cv2.destroyAllWindows()