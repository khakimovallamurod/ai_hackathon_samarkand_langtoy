import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from supervision.draw.color import Color
from telegram import Bot
import time
import asyncio
import config
import text_to_speach
import websocket
import threading
import json

ESP32_WEBSOCKET_URL = "ws://192.168.4.1/ws"  
ws_connection = None
connection_lock = threading.Lock()

DANGEROUS_OBJECTS = [
    'knife', 'scissors', 'oven', 'microwave',
    'fire hydrant', 'motorcycle', 'car', 'train',
    'bus', 'truck'
]

MONITORED_CLASSES = ['person'] + DANGEROUS_OBJECTS + [
    'chair', 'couch', 'bed', 'table', 'tv', 'laptop', 'book',
    'cup', 'bowl', 'food', 'toy', 'backpack', 'cell phone'
]

DANGER_COLOR = Color(255, 0, 0)  
PERSON_COLOR = Color(0, 255, 0)  
KID_COLOR = Color(191, 27, 224)
SAFE_OBJECT_COLOR = Color(15, 219, 209)

def connect_to_esp32():
    global ws_connection
    with connection_lock:
        while True:
            try:
                print("ESP32 ga ulanishga harakat qilinmoqda...")
                ws_connection = websocket.create_connection(ESP32_WEBSOCKET_URL, timeout=10)
                print("ESP32 ga WebSocket orqali muvaffaqiyatli ulanildi!")
                ws_connection.send("child_safety_monitor_connected")
                break 
            except Exception as e:
                print(f"ESP32 ga ulanib bo'lmadi: {e}. 5 soniyadan so'ng qayta urinish.")
                ws_connection = None
                time.sleep(5)

def send_to_esp32(message):
    global ws_connection
    with connection_lock:
        if ws_connection and ws_connection.connected:
            try:
                if isinstance(message, dict):
                    message = json.dumps(message)
                ws_connection.send(message)
                print(f"ESP32 ga yuborildi: {message}")
            except Exception as e:
                print(f"ESP32 ga yuborishda xato: {e}")
                ws_connection = None
                threading.Thread(target=connect_to_esp32, daemon=True).start()
        elif not ws_connection:
            print("ESP32 bilan aloqa yo'q. Qayta ulanishga harakat qilinmoqda...")
            threading.Thread(target=connect_to_esp32, daemon=True).start()

def send_safety_status(child_count, danger_count, dangers_list=None):
    status_data = {
        "type": "safety_status",
        "child_count": child_count,
        "danger_count": danger_count,
        "timestamp": time.time()
    }
    if dangers_list:
        status_data["dangers"] = [
            {
                "danger_class": danger["danger_class"],
                "distance": int(danger["distance"]),
                "person_id": danger["person_id"]
            }
            for danger in dangers_list[:3]  
        ]
    send_to_esp32(status_data)

def send_movement_command(command, duration=2000):
    move_data = {
        "type": "movement",
        "command": command,
        "duration": duration,
        "timestamp": time.time()
    }
    send_to_esp32(move_data)

class Tracker:
    def __init__(self, max_length=20):
        self.trajectories = defaultdict(lambda: deque(maxlen=max_length))
        self.next_id = 1
        self.object_memory = {}
        self.max_distance = 80
    
    def update(self, detections):
        if len(detections.xyxy) == 0:
            return detections
        
        tracker_ids = []
        
        for i, box in enumerate(detections.xyxy):
            center = self.get_center(box)
            class_id = detections.class_id[i]
            
            best_id = None
            min_distance = float('inf')
            
            for obj_id, obj_data in self.object_memory.items():
                if obj_data['class_id'] == class_id:
                    distance = np.linalg.norm(np.array(center) - np.array(obj_data['last_center']))
                    if distance < self.max_distance and distance < min_distance:
                        min_distance = distance
                        best_id = obj_id
            
            if best_id is not None:
                tracker_ids.append(best_id)
                self.object_memory[best_id]['last_center'] = center
                self.trajectories[best_id].append(center)
            else:
                new_id = self.next_id
                self.next_id += 1
                tracker_ids.append(new_id)
                self.object_memory[new_id] = {
                    'class_id': class_id,
                    'last_center': center
                }
                self.trajectories[new_id].append(center)
        
        detections.tracker_id = np.array(tracker_ids)
        return detections
    
    def get_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

class DangerDetector:
    def __init__(self, danger_threshold=120, alert_cooldown=5):
        self.danger_threshold = danger_threshold
        self.alert_cooldown = alert_cooldown
        self.last_alerts = {}
        self.last_esp32_alert = 0
        self.esp32_alert_cooldown = 3  
    def detect_person_danger_proximity(self, person_boxes, dangerous_detections):
        dangers = []
        current_time = time.time()
        
        for person_box, person_id in person_boxes:
            person_center = self.get_box_center(person_box)
            
            for danger_detection in dangerous_detections:
                danger_box = danger_detection['box']
                danger_class = danger_detection['class']
                danger_center = self.get_box_center(danger_box)
                distance = self.calculate_distance(person_center, danger_center)
                
                if distance < self.danger_threshold:
                    alert_key = f"{person_id}_{danger_class}"
                    if (alert_key not in self.last_alerts or 
                        current_time - self.last_alerts[alert_key] > self.alert_cooldown):
                        
                        dangers.append({
                            'person_id': person_id,
                            'person_box': person_box,
                            'danger_box': danger_box,
                            'danger_class': danger_class,
                            'distance': distance
                        })
                        
                        self.last_alerts[alert_key] = current_time
        
        if dangers and (current_time - self.last_esp32_alert > self.esp32_alert_cooldown):
            send_safety_status(
                child_count=len([p for p, _ in person_boxes]),
                danger_count=len(dangers),
                dangers_list=dangers
            )
            
            if len(dangers) > 0:
                send_movement_command("alert_mode", 3000) 
            self.last_esp32_alert = current_time
        return dangers
    
    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def load_models(main_model_path, kid_model_path):
    main_model = YOLO(main_model_path)
    kid_model = YOLO(kid_model_path)
    print(f"Main model classes: {main_model.names}")
    print(f"Kid model classes: {kid_model.names}")
    return main_model, kid_model

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def merge_detections(main_detections, kid_detections, main_model, kid_model):
    if len(main_detections) == 0 and len(kid_detections) == 0:
        return sv.Detections.empty()
    
    all_xyxy = []
    all_confidence = []
    all_class_id = []
    all_class_names = []
    
    person_boxes = []
    if len(main_detections) > 0:
        for i, class_id in enumerate(main_detections.class_id):
            class_name = main_model.names.get(int(class_id), f"Unknown_{class_id}")
            if class_name == 'person':
                person_boxes.append((i, main_detections.xyxy[i]))
    
    used_person_indices = set()
    
    if len(kid_detections) > 0:
        for i, class_id in enumerate(kid_detections.class_id):
            class_name = kid_model.names.get(int(class_id), f"Unknown_{class_id}")
            if class_name == 'Kid':
                kid_box = kid_detections.xyxy[i]
                
                best_overlap = 0
                best_person_idx = -1
                for person_idx, person_box in person_boxes:
                    if person_idx in used_person_indices:
                        continue
                    overlap = calculate_iou(kid_box, person_box)
                    if overlap > 0.3 and overlap > best_overlap:
                        best_overlap = overlap
                        best_person_idx = person_idx
                
                if best_person_idx != -1:
                    used_person_indices.add(best_person_idx)
                
                all_xyxy.append(kid_box)
                all_confidence.append(kid_detections.confidence[i])
                max_class_id = max(main_model.names.keys()) if main_model.names else 0
                kid_class_id = max_class_id + 1
                all_class_id.append(kid_class_id)
                all_class_names.append('Kid')
    
    if len(main_detections) > 0:
        for i, class_id in enumerate(main_detections.class_id):
            class_name = main_model.names.get(int(class_id), f"Unknown_{class_id}")
            
            if class_name == 'person' and i not in used_person_indices:
                all_xyxy.append(main_detections.xyxy[i])
                all_confidence.append(main_detections.confidence[i])
                all_class_id.append(class_id)
                all_class_names.append('person')
            elif class_name != 'person':
                all_xyxy.append(main_detections.xyxy[i])
                all_confidence.append(main_detections.confidence[i])
                all_class_id.append(class_id)
                all_class_names.append(class_name)
    
    if not all_xyxy:
        return sv.Detections.empty()
    
    merged_detections = sv.Detections(
        xyxy=np.array(all_xyxy),
        confidence=np.array(all_confidence),
        class_id=np.array(all_class_id)
    )
    
    merged_detections.class_names = all_class_names
    return merged_detections

def draw_simple_box(frame, x1, y1, x2, y2, color, thickness=2):
    color_bgr = color.as_bgr() if hasattr(color, 'as_bgr') else (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)
    return frame

def draw_text(frame, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def get_class_name_from_detections(detections, index, main_model, kid_model):
    if hasattr(detections, 'class_names') and index < len(detections.class_names):
        return detections.class_names[index]
    
    class_id = detections.class_id[index]
    if class_id in main_model.names:
        return main_model.names[class_id]
    elif class_id in kid_model.names:
        return kid_model.names[class_id]
    else:
        max_main_id = max(main_model.names.keys()) if main_model.names else 0
        if class_id == max_main_id + 1:
            return 'Kid'
        else:
            return f"Unknown_{class_id}"

def annotate_frame(frame, main_model, kid_model, detections, trajectory_tracker, danger_detector):
    annotated_frame = frame.copy()
    person_boxes = []
    child_count = 0
    dangerous_detections = []
    safe_objects = []
    
    if detections.tracker_id is None:
        detections.tracker_id = np.arange(len(detections))
    
    for i, (xyxy, tracker_id, class_id, confidence) in enumerate(zip(
        detections.xyxy, detections.tracker_id, detections.class_id, detections.confidence
    )):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = get_class_name_from_detections(detections, i, main_model, kid_model)
        
        if class_name == 'Kid':
            color = KID_COLOR
            child_count += 1
            person_boxes.append((xyxy, tracker_id))
            draw_text(annotated_frame, "Kid", (x1, y1-10), color=KID_COLOR.as_bgr())
        elif class_name == 'person':
            color = PERSON_COLOR
            person_boxes.append((xyxy, tracker_id))
            draw_text(annotated_frame, "Person", (x1, y1-10), color=PERSON_COLOR.as_bgr())
        elif class_name in DANGEROUS_OBJECTS:
            color = DANGER_COLOR
            dangerous_detections.append({
                'box': xyxy,
                'class': class_name,
                'tracker_id': tracker_id
            })
            draw_text(annotated_frame, f"DANGER: {class_name}", (x1, y1-10), color=DANGER_COLOR.as_bgr())
        else:
            color = SAFE_OBJECT_COLOR
            safe_objects.append({
                'box': xyxy,
                'class': class_name,
                'tracker_id': tracker_id
            })
            draw_text(annotated_frame, class_name, (x1, y1-10), color=SAFE_OBJECT_COLOR.as_bgr())
        
        annotated_frame = draw_simple_box(annotated_frame, x1, y1, x2, y2, color, 2)
        draw_text(annotated_frame, f"ID:{tracker_id}", (x1, y2+15), font_scale=0.4)
        
        if confidence > 0.7 and class_name not in ['Kid', 'person']:
            text_to_speach.text_to_speach_by_lang(text=class_name, filename=f'{class_name}_{class_id}.mp3')

    dangers = danger_detector.detect_person_danger_proximity(person_boxes, dangerous_detections)
    
    for danger in dangers:
        p_box = danger['person_box']
        d_box = danger['danger_box']
        
        p_center = (int((p_box[0] + p_box[2])/2), int((p_box[1] + p_box[3])/2))
        d_center = (int((d_box[0] + d_box[2])/2), int((d_box[1] + d_box[3])/2))
        
        cv2.line(annotated_frame, p_center, d_center, DANGER_COLOR.as_bgr(), 2)
        warning_text = f"DANGER: {danger['danger_class']}"
        draw_text(annotated_frame, warning_text, (p_center[0], p_center[1] - 20), 
                 color=DANGER_COLOR.as_bgr(), font_scale=0.6)
    
    return annotated_frame, child_count, dangers, dangerous_detections, safe_objects

def create_simple_info_panel(frame, child_count, danger_count, safe_count, total_objects, esp32_status):
    panel_height = 80
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    
    stats = [
        f"Kids: {child_count}",
        f"Dangerous: {danger_count}",
        f"Safe: {safe_count}",
        f"Total: {total_objects}"
    ]
    
    for i, stat in enumerate(stats):
        draw_text(panel, stat, (20 + i*120, 25), font_scale=0.5, color=(255, 255, 255))
    
    esp32_color = (0, 255, 0) if esp32_status else (0, 0, 255)
    esp32_text = "ESP32: Connected" if esp32_status else "ESP32: Disconnected"
    draw_text(panel, esp32_text, (20, 50), font_scale=0.4, color=esp32_color)
    
    return panel

async def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    bot = Bot(token=bot_token)
    if image_path:
        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=chat_id, photo=photo, caption=message)
    else:
        await bot.send_message(chat_id=chat_id, text=message)

def get_class_ids(model, class_names):
    class_ids = []
    for name in class_names:
        found = False
        for id, class_name in model.names.items():
            if class_name == name:
                class_ids.append(id)
                found = True
                break
        if not found:
            print(f"⚠️ '{name}' klass modelda topilmadi!")
    return class_ids

def main(camera_index, output_path=None, bot_token=None, chat_id=None):
    MAIN_MODEL_PATH = "models/yolo11n.pt"  
    KID_MODEL_PATH = "models/kid_model.pt"  
    CONFIDENCE_THRESHOLD = 0.4
    NMS_IOU_THRESHOLD = 0.5
    
    threading.Thread(target=connect_to_esp32, daemon=True).start()
    
    main_model, kid_model = load_models(MAIN_MODEL_PATH, KID_MODEL_PATH)
    
    trajectory_tracker = Tracker(max_length=20)
    danger_detector = DangerDetector(danger_threshold=120, alert_cooldown=10)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Kamera ochilmadi!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    main_monitored_class_ids = get_class_ids(main_model, MONITORED_CLASSES)
    kid_class_ids = get_class_ids(kid_model, ['Kid'])

    frame_count = 0
    last_alert_time = 0
    last_status_update = 0
    process_every_n_frames = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame o'qilmadi")
            break
        
        frame_count += 1
        current_time = time.time()
        
        if frame_count % process_every_n_frames != 0:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue
        
        main_results = main_model(frame, imgsz=320, verbose=False, classes=main_monitored_class_ids)[0]
        main_detections = sv.Detections.from_ultralytics(main_results)
        main_detections = main_detections[main_detections.confidence > CONFIDENCE_THRESHOLD]
        main_detections = main_detections.with_nms(NMS_IOU_THRESHOLD)
        
        kid_detections = sv.Detections.empty()
        person_detected = any(main_model.names.get(int(class_id)) == 'person' 
                            for class_id in main_detections.class_id)
        
        if person_detected:
            kid_results = kid_model(frame, imgsz=320, verbose=False, classes=kid_class_ids)[0]
            kid_detections = sv.Detections.from_ultralytics(kid_results)
            kid_detections = kid_detections[kid_detections.confidence > CONFIDENCE_THRESHOLD]
            kid_detections = kid_detections.with_nms(NMS_IOU_THRESHOLD)
        
        merged_detections = merge_detections(main_detections, kid_detections, main_model, kid_model)
        
        if len(merged_detections) > 0:
            merged_detections = trajectory_tracker.update(merged_detections)
        
        annotated_frame, child_count, dangers, dangerous_objects, safe_objects = annotate_frame(
            frame, main_model, kid_model, merged_detections, trajectory_tracker, danger_detector
        )
        
        esp32_connected = ws_connection is not None and ws_connection.connected
        
        if current_time - last_status_update > 10:
            send_safety_status(child_count, len(dangerous_objects))
            last_status_update = current_time
        
        info_panel = create_simple_info_panel(
            annotated_frame, child_count, len(dangerous_objects), 
            len(safe_objects), len(merged_detections), esp32_connected)
        
        final_frame = np.vstack([annotated_frame, info_panel])
        
        if dangers:
            cv2.rectangle(final_frame, (0, 0), (final_frame.shape[1], 5), DANGER_COLOR.as_bgr(), -1)
            draw_text(final_frame, "DANGER DETECTED!", (20, annotated_frame.shape[0] + 60), 
                     font_scale=0.7, color=DANGER_COLOR.as_bgr())
            
            if bot_token and chat_id and (current_time - last_alert_time > 30): 
                alert_msg = f"⚠️ Xavfli holat!\nBolalar: {child_count}\nXavflar: {len(dangers)}"
                
                alert_img_path = "results/alert.jpg"
                cv2.imwrite(alert_img_path, final_frame)
                
                asyncio.run(send_telegram_alert(bot_token, chat_id, alert_msg, alert_img_path))
                last_alert_time = current_time
        
        ret, buffer = cv2.imencode('.jpg', final_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    BOT_TOKEN = config.get_token()
    CHAT_ID = config.get_chat_id()
    
    output_path = 'child_safety_monitoring.mp4'
    camera_idx = 0  
    main(
        camera_index=camera_idx,
        output_path=output_path,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )