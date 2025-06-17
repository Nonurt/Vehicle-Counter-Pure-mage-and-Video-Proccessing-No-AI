import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from threading import Thread, Event
import json
import os
import numpy as np
from datetime import datetime, time as dt_time, timedelta
import time
import math
from collections import defaultdict
import csv

CONFIG_PATH = "ayarlar.json"
CAMERA_INDEX = 1
DEFAULT_CONFIG = {
    "roi": [],
    "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0},
    "enhance": {"gamma": 1.2, "clipLimit": 2.0, "saturation": 1.0},
    "detection": {"threshold": 25, "min_area": 50, "max_distance": 30, "max_disappeared": 5},
    "schedule": {"start_time": "08:00", "end_time": "17:00", "active_days": [1, 2, 3, 4, 5]}
}


class CentroidTracker:
    def __init__(self, max_distance=30, max_disappeared=5):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.total_count = 0
        self.entered_ids = set()
        self.entry_timestamps = {}
        self.last_positions = {}

    def update(self, centroids):
        updated_objects = {}
        current_time = time.time()

        for cid, c_old in self.objects.items():
            self.disappeared[cid] += 1
            if self.disappeared[cid] > self.max_disappeared:
                if cid in self.entry_timestamps:
                    del self.entry_timestamps[cid]
                if cid in self.last_positions:
                    del self.last_positions[cid]
                continue
            updated_objects[cid] = c_old

        self.objects = updated_objects

        matches = {}
        unmatched = centroids[:]

        for cid, c_old in self.objects.items():
            for c in unmatched:
                distance = np.linalg.norm(np.array(c_old) - np.array(c))
                if distance < self.max_distance:
                    self.objects[cid] = c
                    self.disappeared[cid] = 0
                    matches[cid] = c
                    self.last_positions[cid] = c
                    unmatched.remove(c)
                    break

        for c in unmatched:
            self.objects[self.next_id] = c
            self.disappeared[self.next_id] = 0
            self.entry_timestamps[self.next_id] = current_time
            self.last_positions[self.next_id] = c
            self.next_id += 1

        return list(self.objects.items())

    def mark_entered(self, object_id):
        current_time = time.time()
        if object_id in self.entered_ids:
            last_entry_time = self.entry_timestamps.get(object_id, 0)
            if current_time - last_entry_time < 2.0:
                return False

        self.entered_ids.add(object_id)
        self.entry_timestamps[object_id] = current_time
        self.total_count += 1
        return True

    def has_entered(self, object_id):
        return object_id in self.entered_ids


class AppState:
    def __init__(self):
        self.roi_list = []
        self.distortion = DEFAULT_CONFIG["distortion"].copy()
        self.enhance_params = DEFAULT_CONFIG["enhance"].copy()
        self.detection_params = DEFAULT_CONFIG["detection"].copy()
        self.schedule_params = DEFAULT_CONFIG["schedule"].copy()
        self.stop_event = Event()
        self.is_running = False
        self.auto_enhance = True
        self.show_motion_only = False
        self.full_screen_mode = False
        self.tracking_data = defaultdict(dict)
        self.object_counter = defaultdict(int)
        self.global_tracker = CentroidTracker(
            max_distance=self.detection_params["max_distance"],
            max_disappeared=self.detection_params["max_disappeared"]
        )
        self.start_time = None
        self.elapsed_time = timedelta()
        self.session_data = []


class ROIEditor:
    def __init__(self):
        self.dragging = False
        self.resizing = False
        self.rotating = False
        self.selected_roi_idx = -1
        self.selected_corner = -1
        self.start_x = 0
        self.start_y = 0
        self.start_angle = 0
        self.start_width = 0
        self.start_height = 0


root = tk.Tk()
root.title("Gelişmiş Nesne Sayma Sistemi v3.1")
root.geometry("1300x900")

app_state = AppState()
roi_editor = ROIEditor()
auto_enhance_var = tk.BooleanVar(value=app_state.auto_enhance)
show_motion_var = tk.BooleanVar(value=app_state.show_motion_only)
full_screen_mode_var = tk.BooleanVar(value=app_state.full_screen_mode)


def on_auto_enhance_change(*args):
    app_state.auto_enhance = auto_enhance_var.get()


def on_show_motion_change(*args):
    app_state.show_motion_only = show_motion_var.get()


def on_full_screen_mode_change(*args):
    app_state.full_screen_mode = full_screen_mode_var.get()


auto_enhance_var.trace_add('write', on_auto_enhance_change)
show_motion_var.trace_add('write', on_show_motion_change)
full_screen_mode_var.trace_add('write', on_full_screen_mode_change)


def create_default_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)


def validate_rois():
    valid_rois = []
    for roi in app_state.roi_list:
        try:
            validated = {
                'x': max(0, int(roi.get('x', 0))),
                'y': max(0, int(roi.get('y', 0))),
                'w': max(50, int(roi.get('w', 100))),  # Minimum genişlik 50 piksel
                'h': max(50, int(roi.get('h', 100))),  # Minimum yükseklik 50 piksel
                'angle': float(roi.get('angle', 0)) % 360
            }
            valid_rois.append(validated)
        except Exception as e:
            print(f"Geçersiz ROI düzeltildi: {str(e)}")
    app_state.roi_list = valid_rois


def ensure_int_coordinates(roi):
    roi['x'] = int(round(roi['x']))
    roi['y'] = int(round(roi['y']))
    roi['w'] = int(round(roi['w']))
    roi['h'] = int(round(roi['h']))
    return roi


def ensure_roi_visible(roi, frame_shape):
    h, w = frame_shape[:2]
    roi['w'] = max(50, min(roi['w'], w))  # Minimum 50 piksel
    roi['h'] = max(50, min(roi['h'], h))  # Minimum 50 piksel
    roi['x'] = max(0, min(roi['x'], w - roi['w']))
    roi['y'] = max(0, min(roi['y'], h - roi['h']))
    roi['angle'] = roi.get('angle', 0) % 360
    return roi


def enhance_image(img):
    if not app_state.auto_enhance:
        return img

    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=app_state.enhance_params["clipLimit"],
            tileGridSize=(8, 8)
        )
        l_channel = clahe.apply(l_channel)

        improved_lab = cv2.merge((l_channel, a_channel, b_channel))
        img = cv2.cvtColor(improved_lab, cv2.COLOR_LAB2BGR)

        gamma = max(0.1, min(app_state.enhance_params["gamma"], 3.0))
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype("float32")
        saturation_scale = max(0.1, min(app_state.enhance_params["saturation"], 3.0))
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
        hsv = hsv.astype("uint8")
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img
    except Exception as e:
        print(f"Görüntü iyileştirme hatası: {str(e)}")
        return img


def undistort_image(img):
    h, w = img.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, h, h / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([
        app_state.distortion["k1"],
        app_state.distortion["k2"],
        app_state.distortion["p1"],
        app_state.distortion["p2"]
    ], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


def is_within_schedule():
    now = datetime.now()
    current_time = now.time()
    current_day = now.weekday() + 1

    try:
        start_time = dt_time(*map(int, app_state.schedule_params["start_time"].split(':')))
        end_time = dt_time(*map(int, app_state.schedule_params["end_time"].split(':')))
        active_days = app_state.schedule_params["active_days"]

        return (start_time <= current_time <= end_time and
                current_day in active_days)
    except Exception as e:
        print(f"Zaman kontrol hatası: {str(e)}")
        return True


def handle_mouse_events(event, x, y, flags, param):
    global app_state, roi_editor

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_editor.dragging = False
        roi_editor.resizing = False
        roi_editor.rotating = False
        roi_editor.selected_roi_idx = -1

        for i, roi in enumerate(app_state.roi_list):
            cx, cy = roi['x'] + roi['w'] // 2, roi['y'] + roi['h'] // 2
            angle = math.radians(roi.get('angle', 0))
            w, h = roi['w'], roi['h']

            corners = [
                (cx + (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle),
                 cy + (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)),
                (cx - (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle),
                 cy - (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)),
                (cx - (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle),
                 cy - (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)),
                (cx + (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle),
                 cy + (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle))
            ]

            center_dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if center_dist < 15:
                roi_editor.rotating = True
                roi_editor.selected_roi_idx = i
                roi_editor.start_x = x
                roi_editor.start_y = y
                roi_editor.start_angle = roi.get('angle', 0)
                return

            for j, (corner_x, corner_y) in enumerate(corners):
                if math.sqrt((x - corner_x) ** 2 + (y - corner_y) ** 2) < 15:
                    roi_editor.resizing = True
                    roi_editor.selected_roi_idx = i
                    roi_editor.selected_corner = j
                    roi_editor.start_x = x
                    roi_editor.start_y = y
                    roi_editor.start_width = roi['w']
                    roi_editor.start_height = roi['h']
                    return

            if cv2.pointPolygonTest(np.array(corners, dtype=np.int32), (x, y), False) >= 0:
                roi_editor.dragging = True
                roi_editor.selected_roi_idx = i
                roi_editor.start_x = x - roi['x']
                roi_editor.start_y = y - roi['y']
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_editor.selected_roi_idx >= 0:
            roi = app_state.roi_list[roi_editor.selected_roi_idx]
            cx, cy = roi['x'] + roi['w'] // 2, roi['y'] + roi['h'] // 2

            if roi_editor.dragging:
                app_state.roi_list[roi_editor.selected_roi_idx]['x'] = x - roi_editor.start_x
                app_state.roi_list[roi_editor.selected_roi_idx]['y'] = y - roi_editor.start_y

            elif roi_editor.resizing:
                angle = math.radians(roi.get('angle', 0))
                dx = x - roi_editor.start_x
                dy = y - roi_editor.start_y

                rotated_dx = dx * math.cos(-angle) - dy * math.sin(-angle)
                rotated_dy = dx * math.sin(-angle) + dy * math.cos(-angle)

                if roi_editor.selected_corner == 0:
                    new_w = roi_editor.start_width + rotated_dx * 2
                    new_h = roi_editor.start_height + rotated_dy * 2
                elif roi_editor.selected_corner == 1:
                    new_w = roi_editor.start_width - rotated_dx * 2
                    new_h = roi_editor.start_height + rotated_dy * 2
                elif roi_editor.selected_corner == 2:
                    new_w = roi_editor.start_width - rotated_dx * 2
                    new_h = roi_editor.start_height - rotated_dy * 2
                else:
                    new_w = roi_editor.start_width + rotated_dx * 2
                    new_h = roi_editor.start_height - rotated_dy * 2

                app_state.roi_list[roi_editor.selected_roi_idx]['w'] = max(50, new_w)
                app_state.roi_list[roi_editor.selected_roi_idx]['h'] = max(50, new_h)

            elif roi_editor.rotating:
                angle = math.degrees(math.atan2(y - cy, x - cx))
                app_state.roi_list[roi_editor.selected_roi_idx]['angle'] = angle

            refresh_roi_listbox()

    elif event == cv2.EVENT_LBUTTONUP:
        roi_editor.dragging = False
        roi_editor.resizing = False
        roi_editor.rotating = False
        refresh_roi_listbox()


def is_point_in_roi(point, roi):
    cx, cy = roi['x'] + roi['w'] // 2, roi['y'] + roi['h'] // 2
    angle = math.radians(roi.get('angle', 0))
    w, h = roi['w'], roi['h']

    corners = [
        (cx + (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle),
         cy + (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)),
        (cx - (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle),
         cy - (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)),
        (cx - (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle),
         cy - (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)),
        (cx + (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle),
         cy + (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle))
    ]

    return cv2.pointPolygonTest(np.array(corners, dtype=np.int32), point, False) >= 0


def video_loop():
    app_state.is_running = True
    app_state.start_time = datetime.now()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        app_state.is_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    trackers = []
    prev_frames = []
    debug_mode = False
    prev_global_frame = None

    try:
        while not app_state.stop_event.is_set() and app_state.is_running:
            if not is_within_schedule():
                time.sleep(1)
                update_elapsed_time()
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            frame = undistort_image(frame)
            if app_state.auto_enhance:
                frame = enhance_image(frame)

            motion_frame = None
            if app_state.show_motion_only or app_state.full_screen_mode:
                motion_frame = np.zeros_like(frame)

            # Full screen mode processing
            if app_state.full_screen_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                blurred = cv2.medianBlur(blurred, 3)

                if prev_global_frame is None:
                    prev_global_frame = blurred
                    update_elapsed_time()
                    continue

                if prev_global_frame.shape != blurred.shape:
                    prev_global_frame = blurred
                    update_elapsed_time()
                    continue

                delta = cv2.absdiff(prev_global_frame, blurred)
                _, thresh = cv2.threshold(delta, app_state.detection_params["threshold"], 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                centroids = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > app_state.detection_params["min_area"]:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centroids.append((cx, cy))
                            x1, y1, w1, h1 = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                            if motion_frame is not None:
                                motion_frame[y1:y1 + h1, x1:x1 + w1] = frame[y1:y1 + h1, x1:x1 + w1]

                app_state.global_tracker.max_distance = app_state.detection_params["max_distance"]
                app_state.global_tracker.max_disappeared = app_state.detection_params["max_disappeared"]
                objects = app_state.global_tracker.update(centroids)

                for (object_id, centroid) in objects:
                    text = f"ID {object_id}"
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    if motion_frame is not None:
                        cv2.putText(motion_frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(motion_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                prev_global_frame = blurred.copy()

            # ROI processing
            while len(trackers) < len(app_state.roi_list):
                trackers.append(CentroidTracker(
                    max_distance=app_state.detection_params["max_distance"],
                    max_disappeared=app_state.detection_params["max_disappeared"]
                ))
                prev_frames.append(None)

            while len(trackers) > len(app_state.roi_list):
                trackers.pop()
                prev_frames.pop()

            for i, (roi, tracker) in enumerate(zip(app_state.roi_list, trackers)):
                try:
                    roi = ensure_int_coordinates(roi)
                    roi = ensure_roi_visible(roi, frame.shape)
                    x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
                    angle = roi.get('angle', 0)

                    if y + h > frame.shape[0] or x + w > frame.shape[1]:
                        continue

                    roi_img = frame[y:y + h, x:x + w]
                    if roi_img.size == 0:
                        continue

                    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    blurred = cv2.medianBlur(blurred, 3)

                    if prev_frames[i] is None:
                        prev_frames[i] = blurred
                        continue

                    if prev_frames[i].shape != blurred.shape:
                        prev_frames[i] = blurred
                        continue

                    delta = cv2.absdiff(prev_frames[i], blurred)
                    _, thresh = cv2.threshold(delta, app_state.detection_params["threshold"], 255, cv2.THRESH_BINARY)
                    dilated = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    centroids = []
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > app_state.detection_params["min_area"]:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                centroids.append((cx + x, cy + y))
                                x1, y1, w1, h1 = cv2.boundingRect(c)
                                cv2.rectangle(roi_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                                if motion_frame is not None:
                                    motion_frame[y + y1:y + y1 + h1, x + x1:x + x1 + w1] = frame[y + y1:y + y1 + h1,
                                                                                           x + x1:x + x1 + w1]

                    tracker.max_distance = app_state.detection_params["max_distance"]
                    tracker.max_disappeared = app_state.detection_params["max_disappeared"]
                    objects = tracker.update([(cx - x, cy - y) for (cx, cy) in centroids])

                    # Track object entries
                    current_time = time.time()
                    for (object_id, centroid) in objects:
                        global_centroid = (centroid[0] + x, centroid[1] + y)

                        if is_point_in_roi(global_centroid, roi):
                            if not tracker.has_entered(object_id):
                                if tracker.mark_entered(object_id):
                                    app_state.object_counter[i] += 1
                                    save_count_data(i, app_state.object_counter[i])
                                    app_state.session_data.append({
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "roi_index": i,
                                        "object_id": object_id,
                                        "count": app_state.object_counter[i]
                                    })
                            elif current_time - tracker.entry_timestamps.get(object_id, 0) > 5.0:
                                if tracker.mark_entered(object_id):
                                    app_state.object_counter[i] += 1
                                    save_count_data(i, app_state.object_counter[i])
                                    app_state.session_data.append({
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "roi_index": i,
                                        "object_id": object_id,
                                        "count": app_state.object_counter[i]
                                    })

                        text = f"ID {object_id}"
                        cv2.putText(roi_img, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(roi_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                    prev_frames[i] = blurred.copy()

                    cx, cy = x + w // 2, y + h // 2
                    rect = ((cx, cy), (w, h), angle)
                    box = cv2.boxPoints(rect).astype(np.int32)

                    color = (0, 255, 255) if i == roi_editor.selected_roi_idx else (255, 255, 0)
                    thickness = 3
                    cv2.drawContours(frame, [box], 0, color, thickness)
                    if motion_frame is not None:
                        cv2.drawContours(motion_frame, [box], 0, color, thickness)

                    for pt in box:
                        cv2.circle(frame, tuple(pt), 8, (0, 0, 255), -1)
                        if motion_frame is not None:
                            cv2.circle(motion_frame, tuple(pt), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                    if motion_frame is not None:
                        cv2.circle(motion_frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)

                    count_text = f"Count: {app_state.object_counter[i]}"
                    cv2.putText(frame, count_text, (x + 5, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if motion_frame is not None:
                        cv2.putText(motion_frame, count_text, (x + 5, y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except Exception as e:
                    print(f"ROI {i} işlenirken hata: {str(e)}")
                    continue

            if debug_mode:
                debug_frame = frame.copy()
                for i, roi in enumerate(app_state.roi_list):
                    x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(debug_frame, f"ROI {i}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Debug View", debug_frame)

            display_frame = motion_frame if (app_state.show_motion_only or app_state.full_screen_mode) else frame
            cv2.imshow("Object Counter", display_frame)
            cv2.setMouseCallback("Object Counter", handle_mouse_events, display_frame)

            update_elapsed_time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_config()
            elif key == 0x79:  # F10
                debug_mode = not debug_mode
                if not debug_mode:
                    cv2.destroyWindow("Debug View")

    except Exception as e:
        messagebox.showerror("Hata", f"Video akışında hata: {str(e)}")
    finally:
        save_session_data()
        cap.release()
        cv2.destroyAllWindows()
        app_state.is_running = False
        app_state.stop_event.clear()


def save_count_data(roi_index, count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": timestamp,
        "roi_index": roi_index,
        "count": count,
        "roi_params": app_state.roi_list[roi_index]
    }

    log_file = f"count_log_{roi_index}.json"
    try:
        existing_data = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                existing_data = json.load(f)

        existing_data.append(data)

        with open(log_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"Log kaydedilirken hata: {str(e)}")


def save_session_data():
    if not app_state.session_data:
        return

    filename = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    if not filename:
        return

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["timestamp", "roi_index", "object_id", "count"])
            writer.writeheader()
            writer.writerows(app_state.session_data)
        messagebox.showinfo("Başarılı", f"Oturum verileri {filename} dosyasına kaydedildi")
    except Exception as e:
        messagebox.showerror("Hata", f"CSV kaydedilirken hata: {str(e)}")


def update_elapsed_time():
    if app_state.start_time:
        app_state.elapsed_time = datetime.now() - app_state.start_time
        elapsed_str.set(f"Geçen Süre: {str(app_state.elapsed_time).split('.')[0]}")


def start_video():
    if app_state.is_running:
        return

    validate_rois()
    app_state.session_data = []
    app_state.start_time = datetime.now()
    for i in range(len(app_state.roi_list)):
        app_state.object_counter[i] = 0
    Thread(target=video_loop, daemon=True).start()


def stop_video():
    app_state.is_running = False
    app_state.stop_event.set()
    save_session_data()


def create_roi_panel():
    panel = ttk.LabelFrame(root, text="ROI Yönetimi", width=300)
    panel.pack(side="right", fill="y", padx=5, pady=5)

    list_frame = ttk.Frame(panel)
    list_frame.pack(fill="x", pady=5)
    ttk.Label(list_frame, text="ROI Listesi:").pack()

    roi_listbox = tk.Listbox(list_frame, height=10, exportselection=False)
    roi_listbox.pack(fill="x", expand=True)

    btn_frame = ttk.Frame(panel)
    btn_frame.pack(fill="x", pady=5)

    ttk.Button(btn_frame, text="Yeni ROI Ekle", command=add_roi).pack(side="left", expand=True)
    ttk.Button(btn_frame, text="Seçiliyi Sil",
               command=lambda: delete_selected_roi(roi_listbox)).pack(side="left", expand=True)
    ttk.Button(btn_frame, text="Seçiliyi Düzenle",
               command=lambda: edit_selected_roi(roi_listbox)).pack(side="left", expand=True)

    prop_frame = ttk.LabelFrame(panel, text="ROI Özellikleri")
    prop_frame.pack(fill="x", pady=5)

    props = ["x", "y", "w", "h", "angle"]
    prop_vars = {}
    for i, prop in enumerate(props):
        ttk.Label(prop_frame, text=prop.upper() + ":").grid(row=i, column=0, sticky="e", padx=2)
        var = tk.DoubleVar(value=0)
        spin = ttk.Spinbox(prop_frame, from_=0, to=2000, textvariable=var, width=8)
        spin.grid(row=i, column=1, sticky="ew", padx=2)
        prop_vars[prop] = var

    def on_roi_select(event):
        selection = roi_listbox.curselection()
        if selection:
            idx = selection[0]
            roi = app_state.roi_list[idx]
            for prop in props:
                prop_vars[prop].set(roi.get(prop, 0))

    roi_listbox.bind("<<ListboxSelect>>", on_roi_select)

    def update_roi_property(*args, prop):
        selection = roi_listbox.curselection()
        if selection:
            idx = selection[0]
            app_state.roi_list[idx][prop] = float(prop_vars[prop].get())
            refresh_roi_listbox()

    for prop in props:
        prop_vars[prop].trace_add("write", lambda *args, p=prop: update_roi_property(*args, prop=p))

    return panel, roi_listbox


def create_schedule_panel():
    panel = ttk.LabelFrame(root, text="Zamanlama Ayarları", width=300)
    panel.pack(side="right", fill="y", padx=5, pady=5)

    ttk.Label(panel, text="Başlangıç Saati:").pack(anchor="w", padx=5, pady=(5, 0))
    start_time_var = tk.StringVar(value=app_state.schedule_params["start_time"])
    start_time_entry = ttk.Entry(panel, textvariable=start_time_var, width=8)
    start_time_entry.pack(anchor="w", padx=5)

    ttk.Label(panel, text="Bitiş Saati:").pack(anchor="w", padx=5, pady=(5, 0))
    end_time_var = tk.StringVar(value=app_state.schedule_params["end_time"])
    end_time_entry = ttk.Entry(panel, textvariable=end_time_var, width=8)
    end_time_entry.pack(anchor="w", padx=5)

    ttk.Label(panel, text="Aktif Günler:").pack(anchor="w", padx=5, pady=(5, 0))
    days_frame = ttk.Frame(panel)
    days_frame.pack(fill="x", padx=5)

    day_vars = []
    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    for i, day in enumerate(day_names):
        var = tk.BooleanVar(value=(i + 1) in app_state.schedule_params["active_days"])
        cb = ttk.Checkbutton(days_frame, text=day, variable=var)
        cb.pack(side="left")
        day_vars.append((i + 1, var))

    def save_schedule():
        try:
            start_time = start_time_var.get()
            end_time = end_time_var.get()
            datetime.strptime(start_time, "%H:%M")
            datetime.strptime(end_time, "%H:%M")

            active_days = [day for day, var in day_vars if var.get()]

            app_state.schedule_params = {
                "start_time": start_time,
                "end_time": end_time,
                "active_days": active_days
            }

            messagebox.showinfo("Başarılı", "Zamanlama ayarları kaydedildi")
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz zaman formatı (HH:MM kullanın)")

    ttk.Button(panel, text="Kaydet", command=save_schedule).pack(pady=5)

    return panel


def add_roi():
    default_size = 150
    x = 100 if not app_state.roi_list else app_state.roi_list[-1]['x'] + app_state.roi_list[-1]['w'] + 20
    new_roi = {
        'x': max(0, x),
        'y': 100,
        'w': default_size,
        'h': default_size,
        'angle': 0
    }
    app_state.roi_list.append(new_roi)
    refresh_roi_listbox()
    roi_listbox.selection_clear(0, tk.END)
    roi_listbox.selection_set(tk.END)
    roi_listbox.see(tk.END)


def delete_selected_roi(listbox):
    selection = listbox.curselection()
    if selection:
        idx = selection[0]
        if 0 <= idx < len(app_state.roi_list):
            app_state.roi_list.pop(idx)
            refresh_roi_listbox()
            if roi_editor.selected_roi_idx == idx:
                roi_editor.selected_roi_idx = -1
            elif roi_editor.selected_roi_idx > idx:
                roi_editor.selected_roi_idx -= 1


def edit_selected_roi(listbox):
    selection = listbox.curselection()
    if selection:
        idx = selection[0]
        if 0 <= idx < len(app_state.roi_list):
            roi_editor.selected_roi_idx = idx
            refresh_roi_listbox()


def refresh_roi_listbox():
    roi_listbox.delete(0, tk.END)
    for i, roi in enumerate(app_state.roi_list):
        info = f"ROI {i + 1}: X={roi['x']} Y={roi['y']} W={roi['w']} H={roi['h']} A={roi.get('angle', 0):.1f}°"
        roi_listbox.insert(tk.END, info)

        if i == roi_editor.selected_roi_idx:
            roi_listbox.itemconfig(i, {'bg': 'light yellow'})


def update_param(key, value, param_dict):
    param_dict[key] = float(value)


def save_config():
    config = {
        "roi": app_state.roi_list,
        "distortion": app_state.distortion,
        "enhance": app_state.enhance_params,
        "detection": app_state.detection_params,
        "schedule": app_state.schedule_params
    }
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
        messagebox.showinfo("Başarılı", "Ayarlar başarıyla kaydedildi.")
    except Exception as e:
        messagebox.showerror("Hata", f"Ayarlar kaydedilirken hata: {str(e)}")


def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)

            app_state.roi_list = config.get("roi", [])
            app_state.distortion.update(config.get("distortion", DEFAULT_CONFIG["distortion"]))
            app_state.enhance_params.update(config.get("enhance", DEFAULT_CONFIG["enhance"]))
            app_state.detection_params.update(config.get("detection", DEFAULT_CONFIG["detection"]))
            app_state.schedule_params.update(config.get("schedule", DEFAULT_CONFIG["schedule"]))

            for key in app_state.distortion:
                if key in distortion_sliders:
                    distortion_sliders[key].set(app_state.distortion[key])

            for key in app_state.enhance_params:
                if key in enhance_sliders:
                    enhance_sliders[key].set(app_state.enhance_params[key])

            for key in app_state.detection_params:
                if key in detection_sliders:
                    detection_sliders[key].set(app_state.detection_params[key])

            validate_rois()
            refresh_roi_listbox()
    except Exception as e:
        messagebox.showwarning("Uyarı", f"Config yüklenirken hata: {str(e)}\nVarsayılan ayarlar kullanılacak.")


def reset_config():
    if messagebox.askyesno("Onay", "Tüm ayarları sıfırlamak istediğinize emin misiniz?"):
        app_state.roi_list = []
        app_state.distortion = DEFAULT_CONFIG["distortion"].copy()
        app_state.enhance_params = DEFAULT_CONFIG["enhance"].copy()
        app_state.detection_params = DEFAULT_CONFIG["detection"].copy()
        app_state.schedule_params = DEFAULT_CONFIG["schedule"].copy()

        for key in distortion_sliders:
            distortion_sliders[key].set(DEFAULT_CONFIG["distortion"][key])

        for key in enhance_sliders:
            enhance_sliders[key].set(DEFAULT_CONFIG["enhance"][key])

        for key in detection_sliders:
            detection_sliders[key].set(DEFAULT_CONFIG["detection"][key])

        refresh_roi_listbox()


def create_slider_frame(parent, title, params, param_dict):
    frame = ttk.LabelFrame(parent, text=title)
    frame.pack(padx=10, pady=5, fill="x")

    sliders = {}
    for i, (key, (vmin, vmax, vstep, label)) in enumerate(params.items()):
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)

        ttk.Label(row, text=label, width=12).pack(side="left")
        slider = tk.Scale(row, from_=vmin, to=vmax, resolution=vstep,
                          orient="horizontal", length=200,
                          command=lambda val, k=key: update_param(k, float(val), param_dict))
        slider.set(param_dict[key])
        slider.pack(side="left", fill="x", expand=True)
        sliders[key] = slider

    return sliders


# UI Setup
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = ttk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = ttk.Frame(main_frame)
right_frame.pack(side="right", fill="y")

roi_panel, roi_listbox = create_roi_panel()
schedule_panel = create_schedule_panel()

# Elapsed time display
elapsed_str = tk.StringVar(value="Geçen Süre: 00:00:00")
elapsed_label = ttk.Label(right_frame, textvariable=elapsed_str, font=('Arial', 12))
elapsed_label.pack(pady=10)

control_frame = ttk.Frame(left_frame)
control_frame.pack(pady=10)

ttk.Button(control_frame, text="Başlat", command=start_video).pack(side="left", padx=5)
ttk.Button(control_frame, text="Durdur", command=stop_video).pack(side="left", padx=5)
ttk.Button(control_frame, text="Kaydet", command=save_config).pack(side="left", padx=5)
ttk.Button(control_frame, text="Sıfırla", command=reset_config).pack(side="left", padx=5)
ttk.Button(control_frame, text="CSV'ye Aktar", command=save_session_data).pack(side="left", padx=5)

options_frame = ttk.Frame(left_frame)
options_frame.pack(fill="x", padx=10, pady=5)
tk.Checkbutton(options_frame, text="Otomatik Geliştirme", variable=auto_enhance_var).pack(side="left", padx=5)
tk.Checkbutton(options_frame, text="Sadece Hareketi Göster", variable=show_motion_var).pack(side="left", padx=5)
tk.Checkbutton(options_frame, text="Tüm Ekran Algoritması", variable=full_screen_mode_var).pack(side="left", padx=5)

slider_notebook = ttk.Notebook(left_frame)
slider_notebook.pack(fill="x", padx=10, pady=5)

# Distortion sliders
distortion_params = {
    "k1": (-0.5, 0.5, 0.01, "Radial K1"),
    "k2": (-0.5, 0.5, 0.01, "Radial K2"),
    "p1": (-0.5, 0.5, 0.01, "Tangential P1"),
    "p2": (-0.5, 0.5, 0.01, "Tangential P2")
}
distortion_sliders = create_slider_frame(slider_notebook, "Distortion Ayarları", distortion_params,
                                         app_state.distortion)

# Enhancement sliders
enhance_params_config = {
    "gamma": (0.1, 3.0, 0.1, "Gamma"),
    "clipLimit": (1.0, 5.0, 0.1, "Kontrast"),
    "saturation": (0.1, 3.0, 0.1, "Doygunluk")
}
enhance_sliders = create_slider_frame(slider_notebook, "Görüntü İyileştirme", enhance_params_config,
                                      app_state.enhance_params)

# Detection sliders
detection_params_config = {
    "threshold": (5, 50, 1, "Eşik Değeri"),
    "min_area": (5, 200, 5, "Min Alan"),
    "max_distance": (10, 100, 5, "Max Mesafe"),
    "max_disappeared": (1, 30, 1, "Max Kayıp")
}
detection_sliders = create_slider_frame(slider_notebook, "Algılama Ayarları", detection_params_config,
                                        app_state.detection_params)

create_default_config()
load_config()

root.protocol("WM_DELETE_WINDOW", lambda: (stop_video(), root.destroy()))
root.mainloop()