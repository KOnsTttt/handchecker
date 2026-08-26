# -*- coding: utf-8 -*-
"""
Hand Control HUD — управление системным звуком Windows жестами руки.

Функция: 1 (громкость системы: режимы GESTURE / FADER).
Бинды:   M — режим, R — запись жеста, S — сохранить, C — сброс датасета, ESC — выход.
Фейдер:  ТОЛЬКО верхняя левая четверть кадра. Верх зоны = 100%, низ зоны = 0%.
"""

import os
import sys
import time
import math
import pickle
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.neighbors import KNeighborsClassifier
from pycaw.pycaw import AudioUtilities

# ------------------------------- Конфигурация --------------------------------
MODEL_PATH      = "hand_landmarker.task"
DATASET_FILE    = "gestures_data.pkl"
GESTURE_LABEL   = "MUTE_TOGGLE"     # класс записываемого жеста

HOLD_TRIGGER    = 0.40              # удержание жеста до действия, сек
PREDICT_DIST    = 0.45              # порог расстояния KNN
SAMPLE_INTERVAL = 0.05              # интервал записи сэмплов, сек
SMOOTHING       = 0.35              # сглаживание фейдера

WIN_TITLE = "Hand Control HUD"
WIN_W, WIN_H = 1280, 760

FRAME_X, FRAME_Y = 24, 96           # рамка вебки
FRAME_W, FRAME_H = 900, 506

AI_INPUT_SIZE   = (256, 256)
CAM_SIZE        = (640, 480)

MODES = ["GESTURE", "FADER"]

# Тёмная палитра (BGR)
COL_BG       = (32, 29, 26)
COL_PANEL    = (52, 47, 42)
COL_VIDEO_BG = (14, 13, 12)
COL_BORDER   = (96, 88, 78)
COL_TEXT     = (238, 234, 228)
COL_DIM      = (156, 148, 138)
COL_ACCENT   = (208, 170, 60)
COL_GREEN    = (120, 200, 90)
COL_RED      = (85, 85, 225)
COL_BAR_LOW  = (120, 200, 90)       # зелёный
COL_BAR_MID  = (60, 175, 250)       # янтарный
COL_BAR_HIGH = (85, 85, 225)        # красный


# ------------------------------ Системное аудио ------------------------------
class SystemAudio:
    """Обёртка над pycaw с обработкой ошибок."""

    def __init__(self):
        self._ctrl = None
        try:
            self._ctrl = AudioUtilities.GetSpeakers().EndpointVolume
        except Exception as exc:
            print(f"[AUDIO] Инициализация не удалась: {exc}")

    @property
    def available(self):
        return self._ctrl is not None

    def volume(self):
        """Текущая громкость 0..1."""
        if not self.available:
            return 0.0
        try:
            return float(self._ctrl.GetMasterVolumeLevelScalar())
        except Exception:
            return 0.0

    def set_volume(self, value):
        if not self.available:
            return False
        try:
            self._ctrl.SetMasterVolumeLevelScalar(float(np.clip(value, 0.0, 1.0)), None)
            return True
        except Exception as exc:
            print(f"[AUDIO] set_volume: {exc}")
            return False

    def muted(self):
        if not self.available:
            return False
        try:
            return bool(self._ctrl.GetMute())
        except Exception:
            return False

    def toggle_mute(self):
        if not self.available:
            return
        try:
            self._ctrl.SetMute(not self.muted(), None)
            print(f"[AUDIO] Звук: {'ВЫКЛ' if self.muted() else 'ВКЛ'}")
        except Exception as exc:
            print(f"[AUDIO] toggle_mute: {exc}")


# ---------------------------- Камера (асинхронно) ----------------------------
class ThreadedCamera:
    def __init__(self, src=0, size=CAM_SIZE):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = self.cap.isOpened()
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        if self.ok and not self.started:
            self.started = True
            threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        try:
            self.cap.release()
        except Exception:
            pass


# --------------------------- Датасет и классификатор -------------------------
def load_dataset(path):
    if not os.path.exists(path):
        return {"data": [], "labels": []}
    try:
        with open(path, "rb") as f:
            ds = pickle.load(f)
        if not isinstance(ds, dict) or "data" not in ds or "labels" not in ds:
            raise ValueError("неверный формат файла")
        return ds
    except Exception as exc:
        print(f"[DATA] Ошибка загрузки: {exc}")
        return {"data": [], "labels": []}


def save_dataset(ds, path):
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f"[DATA] Ошибка сохранения: {exc}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def train_classifier(ds):
    data, labels = ds["data"], ds["labels"]
    if len(data) < 5:
        return None
    try:
        knn = KNeighborsClassifier(n_neighbors=min(3, len(labels)), weights="distance")
        knn.fit(data, labels)
        return knn
    except Exception as exc:
        print(f"[ML] Обучение не удалось: {exc}")
        return None


def normalize_landmarks(landmarks):
    """Нормализация относительно запястья (точка 0) + масштаб по макс. отклонению."""
    bx, by, bz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    features = []
    for p in landmarks:
        features.extend((p.x - bx, p.y - by, p.z - bz))
    features = np.asarray(features, dtype=np.float32)
    max_val = np.max(np.abs(features))
    return features / max_val if max_val > 0 else features


# ---------------------------------- UI утилиты -------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, x, y, color=COL_TEXT, scale=0.52, th=1):
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, th, cv2.LINE_AA)


def put_segment(img, text, x, y, color=COL_TEXT, scale=0.55, th=1):
    """Текст + автоматическое продвижение X для статусных строк."""
    put(img, text, x, y, color, scale, th)
    (w, _), _ = cv2.getTextSize(text, FONT, scale, th)
    return x + w + 22


def bar_color(vol):
    stops = [0, 50, 100]
    b = int(np.interp(vol, stops, [COL_BAR_LOW[0], COL_BAR_MID[0], COL_BAR_HIGH[0]]))
    g = int(np.interp(vol, stops, [COL_BAR_LOW[1], COL_BAR_MID[1], COL_BAR_HIGH[1]]))
    r = int(np.interp(vol, stops, [COL_BAR_LOW[2], COL_BAR_MID[2], COL_BAR_HIGH[2]]))
    return (b, g, r)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_hand(canvas, pts):
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(canvas, pts[p1], pts[p2], COL_ACCENT, 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = COL_RED if i in (4, 8, 12, 16, 20) else COL_GREEN
        cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)


# ----------------------------------- Init ------------------------------------
try:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)
except Exception as exc:
    print(f"[MP] Не удалось загрузить модель '{MODEL_PATH}': {exc}")
    sys.exit(1)

audio = SystemAudio()
dataset = load_dataset(DATASET_FILE)
classifier = train_classifier(dataset)

cam = ThreadedCamera().start()
time.sleep(0.4)

cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_TITLE, WIN_W, WIN_H)
cv2.setWindowProperty(WIN_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

BINDS_TEXT = "[M] Mode   [R] Record   [S] Save   [C] Reset   [ESC] Exit"

mode_idx = 0
recording = False
rec_samples = 0
last_sample_t = 0.0
save_msg, save_msg_t = "", 0.0

stable_g, g_start, acted = None, 0.0, False
pinched, drag_vol = False, 0.0

fps = 60.0
prev_t = time.time()

# --------------------------------- Главный цикл -------------------------------
while True:
    ret, bgr = cam.read()
    now = time.time()
    dt = now - prev_t
    prev_t = now
    fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else fps)

    canvas = np.full((WIN_H, WIN_W, 3), COL_BG, dtype=np.uint8)
    active_mode = MODES[mode_idx]

    # ----- Рамка и видео -----
    cv2.rectangle(canvas, (FRAME_X - 2, FRAME_Y - 2),
                  (FRAME_X + FRAME_W + 2, FRAME_Y + FRAME_H + 2), COL_BORDER, 1)
    canvas[FRAME_Y:FRAME_Y + FRAME_H, FRAME_X:FRAME_X + FRAME_W] = COL_VIDEO_BG

    vx, vy, vw, vh, vscale = FRAME_X, FRAME_Y, 0, 0, 1.0
    pts, feats = None, None

    if ret and bgr is not None:
        disp = cv2.flip(bgr, 1)
        fh, fw = disp.shape[:2]
        vscale = min(FRAME_W / fw, FRAME_H / fh)
        vw, vh = int(fw * vscale), int(fh * vscale)
        vx = FRAME_X + (FRAME_W - vw) // 2
        vy = FRAME_Y + (FRAME_H - vh) // 2
        canvas[vy:vy + vh, vx:vx + vw] = cv2.resize(disp, (vw, vh))

        small = cv2.resize(disp, AI_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, int(now * 1000))

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            pts = [(int(vx + p.x * vw), int(vy + p.y * vh)) for p in lm]
            feats = normalize_landmarks(lm)
            draw_hand(canvas, pts)
    elif not cam.ok:
        put(canvas, "CAMERA OFFLINE", FRAME_X + FRAME_W // 2 - 110,
            FRAME_Y + FRAME_H // 2, COL_RED, 0.8, 2)
    else:
        put(canvas, "NO SIGNAL", FRAME_X + FRAME_W // 2 - 70,
            FRAME_Y + FRAME_H // 2, COL_DIM, 0.8, 2)

    gesture_name = "-"

    # ----- Логика режимов -----
    if pts is not None and active_mode == "GESTURE":
        if classifier is None:
            gesture_name = "нет модели: [R] запись -> [S] сохранить"
        else:
            dist, _ = classifier.kneighbors([feats], n_neighbors=1)
            if dist[0][0] <= PREDICT_DIST:
                gesture_name = str(classifier.predict([feats])[0])
            else:
                gesture_name = "unknown"

        # Удержание целевого жеста -> мьют
        holding = (gesture_name == GESTURE_LABEL)
        if holding and stable_g != GESTURE_LABEL:
            stable_g, g_start, acted = GESTURE_LABEL, now, False
        elif not holding:
            stable_g, acted = None, False
        if holding and stable_g == GESTURE_LABEL:
            elapsed = now - g_start
            progress = min(elapsed / HOLD_TRIGGER, 1.0)
            bw = int(progress * 280)
            py_ = FRAME_Y + FRAME_H + 46
            cv2.rectangle(canvas, (FRAME_X, py_), (FRAME_X + 280, py_ + 10), COL_PANEL, -1)
            cv2.rectangle(canvas, (FRAME_X, py_), (FRAME_X + bw, py_ + 10), COL_GREEN, -1)
            if elapsed >= HOLD_TRIGGER and not acted:
                audio.toggle_mute()
                acted = True

    elif pts is not None and active_mode == "FADER":
        # Зона фейдера — верхняя левая четверть кадра. Шкала у правого края зоны.
        zx2, zy2 = vx + vw // 2, vy + vh // 2
        cv2.rectangle(canvas, (vx, vy), (zx2, zy2), COL_ACCENT, 1, cv2.LINE_AA)
        put(canvas, "VOL ZONE", vx + 8, vy + 22, COL_ACCENT, 0.48)

        gx = zx2 - 14
        cv2.line(canvas, (gx, vy + 10), (gx, zy2 - 10), COL_PANEL, 1, cv2.LINE_AA)
        for frac, lbl in ((0.0, "100"), (0.5, "50"), (1.0, "0")):
            gy = int(vy + frac * (zy2 - vy - 20)) + 10
            cv2.line(canvas, (gx - 6, gy), (gx + 6, gy), COL_DIM, 1, cv2.LINE_AA)
            put(canvas, lbl, gx + 9, gy + 4, COL_DIM, 0.38)

        x1, y1 = pts[4]
        x2, y2 = pts[8]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pdist = math.hypot(x2 - x1, y2 - y1)
        thr = max(26, int(36 * vscale))
        inside = vx <= cx <= zx2 and vy <= cy <= zy2

        cv2.line(canvas, (x1, y1), (x2, y2), COL_ACCENT, 2, cv2.LINE_AA)
        if pdist < thr and inside:
            if not pinched:
                pinched, drag_vol = True, audio.volume() * 100
            target = float(np.interp(cy, [vy + 10, zy2 - 10], [100, 0]))
            drag_vol = (1 - SMOOTHING) * drag_vol + SMOOTHING * target
            audio.set_volume(drag_vol / 100.0)
            cv2.circle(canvas, (cx, cy), 9, COL_GREEN, -1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (gx, cy), COL_GREEN, 1, cv2.LINE_AA)
            gesture_name = f"DRAG {int(drag_vol)}%"
        elif pdist < thr:
            # Пинч вне зоны — игнорируем управление громкостью
            pinched = False
            cv2.circle(canvas, (cx, cy), 6, COL_RED, 2, cv2.LINE_AA)
            gesture_name = "OUT OF ZONE"
        else:
            pinched = False
            cv2.circle(canvas, (cx, cy), 6, COL_ACCENT, 2, cv2.LINE_AA)
            gesture_name = f"pinch {pdist:.0f}px"
    else:
        stable_g, acted, pinched = None, False, False

    # ----- Запись сэмплов (в любом режиме) -----
    if recording and feats is not None and now - last_sample_t > SAMPLE_INTERVAL:
        dataset["data"].append(feats)
        dataset["labels"].append(GESTURE_LABEL)
        rec_samples += 1
        last_sample_t = now

    # ----- Клавиши -----
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('m'), ord('M')):
        mode_idx = (mode_idx + 1) % len(MODES)
        stable_g, acted, pinched = None, False, False
    elif key in (ord('r'), ord('R')):
        recording, rec_samples = True, 0
    elif key in (ord('s'), ord('S')):
        recording = False
        classifier = train_classifier(dataset)
        if save_dataset(dataset, DATASET_FILE):
            save_msg = f"saved: {len(dataset['data'])} samples, cls={'ON' if classifier else 'OFF'}"
        else:
            save_msg = "SAVE ERROR (см. консоль)"
        save_msg_t = now
    elif key in (ord('c'), ord('C')):
        dataset = {"data": [], "labels": []}
        classifier = None
        recording, rec_samples = False, 0
        try:
            if os.path.exists(DATASET_FILE):
                os.remove(DATASET_FILE)
            save_msg, save_msg_t = "dataset cleared", now
            print("[DATA] Датасет очищен, файл удалён")
        except OSError as exc:
            save_msg, save_msg_t = f"DELETE ERROR: {exc}", now

    # ----- Шапка рамки: режим + бинды -----
    hy = FRAME_Y - 12
    cv2.rectangle(canvas, (FRAME_X - 2, FRAME_Y - 36),
                  (FRAME_X + FRAME_W + 2, FRAME_Y - 2), COL_PANEL, -1)
    put(canvas, f"MODE: {active_mode}", FRAME_X + 12, hy, COL_ACCENT, 0.62, 2)
    (ts, _), _ = cv2.getTextSize(BINDS_TEXT, FONT, 0.52, 1)
    put(canvas, BINDS_TEXT, FRAME_X + FRAME_W - ts - 12, hy, COL_DIM)

    # ----- Строка статуса под рамкой -----
    sy = FRAME_Y + FRAME_H + 24
    x = FRAME_X
    x = put_segment(canvas, f"FPS {int(fps)}", x, sy, COL_DIM)
    x = put_segment(canvas, f"AUDIO {'N/A' if not audio.available else ('MUTED' if audio.muted() else 'ON')}",
                    x, sy, COL_RED if audio.muted() else COL_GREEN,
                    0.55, 2 if audio.muted() else 1)
    x = put_segment(canvas, f"GESTURE: {gesture_name}", x, sy, COL_TEXT)
    x = put_segment(canvas, f"DB {len(dataset['data'])}", x, sy, COL_DIM)
    x = put_segment(canvas, "CLS READY" if classifier else "CLS NONE",
                    x, sy, COL_GREEN if classifier else COL_RED)

    if recording:
        if int(now * 2) % 2 == 0:
            cv2.circle(canvas, (FRAME_X + FRAME_W - 118, FRAME_Y + 22), 7, COL_RED, -1)
        put(canvas, f"REC {rec_samples}", FRAME_X + FRAME_W - 102, FRAME_Y + 29, COL_RED, 0.62, 2)

    if save_msg and now - save_msg_t < 3.0:
        put(canvas, save_msg, WIN_W - 420, sy, COL_GREEN)

    # ----- Саунд-бар внизу -----
    vol = audio.volume() * 100.0
    bx0, bx1 = FRAME_X, WIN_W - FRAME_X
    by0, by1 = WIN_H - 56, WIN_H - 30
    put(canvas, "SYSTEM VOLUME", bx0, by0 - 10, COL_DIM, 0.5)
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (18, 17, 16), -1)
    fill_w = int(np.interp(vol, [0, 100], [0, bx1 - bx0]))
    cv2.rectangle(canvas, (bx0, by0), (bx0 + fill_w, by1), bar_color(vol), -1)
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), COL_BORDER, 1)
    put(canvas, f"{vol:3.0f}%", bx1 - 74, by1 - 8, COL_TEXT, 0.6, 2)
    if audio.muted():
        put(canvas, "MUTED", (bx0 + bx1) // 2 - 44, by1 - 8, COL_RED, 0.7, 2)

    cv2.imshow(WIN_TITLE, canvas)

cam.stop()
cv2.destroyAllWindows()
print("[EXIT] Приложение завершено.")
