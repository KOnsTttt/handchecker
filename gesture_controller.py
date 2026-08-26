import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time
import math
from sklearn.neighbors import KNeighborsClassifier
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

# --- Системное аудио Windows ---
device = AudioUtilities.GetSpeakers()
volume_control = device.EndpointVolume

def toggle_system_mute():
    current_mute = volume_control.GetMute()
    volume_control.SetMute(not current_mute, None)
    print(f"[ACTION] Общий звук: {'MUTED' if not current_mute else 'ACTIVE'}")

# --- Точечное управление аудиосессией Spotify ---
def get_spotify_volume_control():
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if session.Process and session.Process.name().lower() == "spotify.exe":
            return volume
    return None

def set_spotify_volume(volume_scalar):
    spotify_vol = get_spotify_volume_control()
    if spotify_vol:
        spotify_vol.SetMasterVolume(volume_scalar, None)
        return True
    return False

# --- MediaPipe Tasks API (VIDEO MODE) ---
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

# --- Камера ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Три режима работы
MODES = ["GESTURE_MUTE", "MASTER_VOL", "SPOTIFY_VOL"]
current_mode_idx = 0

current_stable_gesture = None
gesture_start_time = 0
HOLD_DURATION_TRIGGER = 0.25
action_executed = False

SMOOTHING = 0.3
current_smoothed_vol = volume_control.GetMasterVolumeLevelScalar() * 100
vol_percent = int(current_smoothed_vol)
vol_bar = np.interp(vol_percent, [0, 100], [380, 120])

prev_time = time.time()
recording_class = None
last_sample_time = 0
SAMPLE_INTERVAL = 0.05

COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_SPOTIFY = (30, 215, 96) # Spotify Green
COLOR_BG_BOX = (25, 25, 25)
COLOR_BUTTON = (50, 50, 50)
COLOR_BUTTON_ACTIVE = (0, 180, 255)

BTN_RECT = [400, 12, 625, 48]

def on_mouse_click(event, x, y, flags, param):
    global current_mode_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_RECT[0] <= x <= BTN_RECT[2] and BTN_RECT[1] <= y <= BTN_RECT[3]:
            current_mode_idx = (current_mode_idx + 1) % len(MODES)
            print(f"[MODE] Режим переключен: {MODES[current_mode_idx]}")

cv2.namedWindow("Hand Vision - Gesture HUD")
cv2.setMouseCallback("Hand Vision - Gesture HUD", on_mouse_click)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 60
    prev_time = curr_time

    frame_timestamp_ms = int(curr_time * 1000)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
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

        # 1. Режим Mute (Жестовый триггер)
        if active_mode == "GESTURE_MUTE":
            if classifier is not None:
                distances, indices = classifier.kneighbors([raw_features], n_neighbors=1)
                min_dist = distances[0][0]
                MAX_DISTANCE_THRESHOLD = 0.45

                if min_dist <= MAX_DISTANCE_THRESHOLD:
                    prediction = classifier.predict([raw_features])[0]
                    proba = np.max(classifier.predict_proba([raw_features]))
                    confidence = proba * 100
                    detected_gesture = prediction
                else:
                    detected_gesture = "UNKNOWN"
                    confidence = 0.0
            else:
                detected_gesture = "NEED TRAINING (R)"

        # 2. Аналоговые режимы (Master или Spotify)
        elif active_mode in ["MASTER_VOL", "SPOTIFY_VOL"]:
            detected_gesture = "PINCH CONTROL"
            x1, y1 = landmark_points[4]
            x2, y2 = landmark_points[8]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            line_color = COLOR_SPOTIFY if active_mode == "SPOTIFY_VOL" else COLOR_MAGENTA
            cv2.circle(frame, (x1, y1), 6, line_color, -1)
            cv2.circle(frame, (x2, y2), 6, line_color, -1)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
            cv2.circle(frame, (cx, cy), 5, COLOR_CYAN, -1)

            length = math.hypot(x2 - x1, y2 - y1)
            target_vol = np.interp(length, [25, 170], [0, 100])
            current_smoothed_vol = (1 - SMOOTHING) * current_smoothed_vol + SMOOTHING * target_vol
            clamped_vol = np.clip(current_smoothed_vol, 0, 100)

            # Отправляем громкость в зависимости от режима
            if active_mode == "MASTER_VOL":
                volume_control.SetMasterVolumeLevelScalar(clamped_vol / 100.0, None)
            elif active_mode == "SPOTIFY_VOL":
                success = set_spotify_volume(clamped_vol / 100.0)
                if not success:
                    cv2.putText(frame, "SPOTIFY NOT RUNNING", (w - 280, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

            vol_percent = int(clamped_vol)
            vol_bar = np.interp(clamped_vol, [0, 100], [380, 120])

    # Клавиатура
    key = cv2.waitKey(1) & 0xFF
    
    if key in [ord('m'), ord('M')]:
        current_mode_idx = (current_mode_idx + 1) % len(MODES)
        print(f"[MODE] Режим переключен: {MODES[current_mode_idx]}")

    elif key in [ord('r'), ord('R')] and active_mode == "GESTURE_MUTE":
        recording_class = "MUTE_TOGGLE"
    elif key in [ord('n'), ord('N')] and active_mode == "GESTURE_MUTE":
        recording_class = "NONE"
    elif key in [ord('c'), ord('C')]:
        dataset = {"data": [], "labels": []}
        classifier = None
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)
        print("[INFO] База данных сброшена.")
    else:
        if recording_class is not None:
            classifier = train_classifier()
            with open(DATASET_FILE, "wb") as f:
                pickle.dump(dataset, f)
            print(f"[INFO] Сохранено. Сэмплов: {len(dataset['data'])}")
            recording_class = None

    # Запись сэмплов
    if recording_class and raw_features is not None:
        if curr_time - last_sample_time > SAMPLE_INTERVAL:
            dataset["data"].append(raw_features)
            dataset["labels"].append(recording_class)
            last_sample_time = curr_time

        count = dataset['labels'].count(recording_class)
        cv2.putText(frame, f"REC [{recording_class}]: {count}", (w - 320, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    # Триггер Mute
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

    # --- UI Индикаторы ---
    # Кнопка переключения режимов
    cv2.rectangle(frame, (BTN_RECT[0], BTN_RECT[1]), (BTN_RECT[2], BTN_RECT[3]), COLOR_BUTTON, -1)
    cv2.rectangle(frame, (BTN_RECT[0], BTN_RECT[1]), (BTN_RECT[2], BTN_RECT[3]), COLOR_BUTTON_ACTIVE, 2)
    cv2.putText(frame, f"MODE [M]: {active_mode}", (BTN_RECT[0] + 10, BTN_RECT[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Вертикальный ползунок (для Master или Spotify)
    if active_mode in ["MASTER_VOL", "SPOTIFY_VOL"]:
        bar_color = COLOR_SPOTIFY if active_mode == "SPOTIFY_VOL" else COLOR_GREEN
        cv2.rectangle(frame, (w - 60, 120), (w - 30, 380), COLOR_BG_BOX, -1)
        cv2.rectangle(frame, (w - 60, int(vol_bar)), (w - 30, 380), bar_color, -1)
        cv2.rectangle(frame, (w - 60, 120), (w - 30, 380), (150, 150, 150), 2)
        cv2.putText(frame, f"{vol_percent}%", (w - 75, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

    # Верхний HUD
    mute_state = "MUTED" if volume_control.GetMute() else "UNMUTED"
    state_color = COLOR_RED if mute_state == "MUTED" else COLOR_GREEN
    
    cv2.rectangle(frame, (10, 10), (250, 100), COLOR_BG_BOX, -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Audio: {mute_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    cv2.putText(frame, f"Mode: {active_mode}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1)

    cv2.imshow("Hand Vision - Gesture HUD", frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()