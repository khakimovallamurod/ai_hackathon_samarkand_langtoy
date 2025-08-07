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

DANGEROUS_OBJECTS = [
    'knife', 'scissors', 'bottle', 'wine glass', 'oven', 'microwave',
    'sink', 'refrigerator', 'fire hydrant', 'motorcycle', 'car', 'train',
    'bus', 'truck'
]

MONITORED_CLASSES = ['person'] + DANGEROUS_OBJECTS + [
    'chair', 'couch', 'bed', 'table', 'tv', 'laptop', 'book',
    'cup', 'bowl', 'food', 'toy', 'backpack', 'cell phone'
]

DANGER_COLOR = Color(255, 0, 0)  
PERSON_COLOR = Color(0, 255, 0)  
CHILD_COLOR = Color(0, 255, 255)  
SAFE_OBJECT_COLOR = Color(15, 219, 209)  
class Tracker:
    def __init__(self, max_length=30):
        self.trajectories = defaultdict(lambda: deque(maxlen=max_length))
        self.next_id = 1
        self.object_memory = {}  
        self.max_distance = 100 
    
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
    
    def draw_trajectory(self, frame, tracker_id, color=(0, 255, 255), thickness=2):
        if tracker_id not in self.trajectories:
            return frame
        
        trajectory = list(self.trajectories[tracker_id])
        if len(trajectory) < 2:
            return frame
        
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
        
        if trajectory:
            center = (int(trajectory[-1][0]), int(trajectory[-1][1]))
            cv2.circle(frame, center, 3, color, -1)
        
        return frame

class DangerDetector:
    def __init__(self, danger_threshold=150, alert_cooldown=5):
        self.danger_threshold = danger_threshold
        self.alert_cooldown = alert_cooldown
        self.last_alerts = {}
    
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
        return dangers
    
    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_class_colors(model):
    class_colors = {}
    for class_id, class_name in model.names.items():
        if class_name in DANGEROUS_OBJECTS:
            class_colors[class_name] = DANGER_COLOR
        elif class_name == 'person':
            class_colors[class_name] = PERSON_COLOR
        else:
            class_colors[class_name] = SAFE_OBJECT_COLOR
    return class_colors

def load_model(model_path):
    model = YOLO(model_path)
    print(f"Mavjud classlar: {model.names}")
    return model
   
def draw_stable_professional_box(frame, x1, y1, x2, y2, color, thickness=2, corner_length_ratio=0.15):
    width = x2 - x1
    height = y2 - y1
    corner_length = int(min(width, height) * corner_length_ratio)
    corner_length = max(corner_length, 15)
    corner_length = min(corner_length, 50)
    
    color_bgr = color.as_bgr() if hasattr(color, 'as_bgr') else (255, 255, 255)
    
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color_bgr, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color_bgr, thickness)
    
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color_bgr, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color_bgr, thickness)
    
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color_bgr, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color_bgr, thickness)
    
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color_bgr, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color_bgr, thickness)
    
    return frame

def draw_text_with_outline(frame, text, position, font_scale=0.6, thickness=2, 
                          text_color=(255, 255, 255), outline_color=(0, 0, 0), outline_thickness=4):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, text, (int(x), int(y)), font, font_scale, outline_color, outline_thickness)
    cv2.putText(frame, text, (int(x), int(y)), font, font_scale, text_color, thickness)
    

def get_class_name(model, class_id):

    if model and hasattr(model, 'names'):
        return model.names.get(int(class_id), f"Unknown_{class_id}")
    return f"Unknown_{class_id}"

def is_child(detection_box, person_boxes, threshold_ratio=0.65):
    
    current_area = (detection_box[2] - detection_box[0]) * (detection_box[3] - detection_box[1])
    
    total_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes)
    avg_area = total_area / len(person_boxes)
    
    return current_area < (avg_area * threshold_ratio)

def filter_detections_by_classes(detections, model, target_classes):
    if len(detections) == 0:
        return detections
    
    filtered_indices = []
    for i, class_id in enumerate(detections.class_id):
        class_name = get_class_name(model, class_id)
        if class_name in target_classes:
            filtered_indices.append(i)
    
    if not filtered_indices:
        return sv.Detections.empty()
    
    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[filtered_indices],
        confidence=detections.confidence[filtered_indices] if detections.confidence is not None else None,
        class_id=detections.class_id[filtered_indices],
        tracker_id=detections.tracker_id[filtered_indices] if detections.tracker_id is not None else None
    )
    
    return filtered_detections

def annotate_frame(frame, model, detections, class_colors, trajectory_tracker, danger_detector):
    annotated_frame = frame.copy()
    person_boxes = []
    child_boxes = []
    dangerous_detections = []
    safe_objects = []
    
    if detections.tracker_id is None:
        detections.tracker_id = np.arange(len(detections))
    
    for i, (xyxy, tracker_id, class_id) in enumerate(zip(detections.xyxy, detections.tracker_id, detections.class_id)):
        class_name = get_class_name(model, class_id)
        if class_name == 'person':
            person_boxes.append((xyxy, tracker_id))
    
    for i, (xyxy, tracker_id, class_id, confidence) in enumerate(zip(
        detections.xyxy, detections.tracker_id, detections.class_id, detections.confidence
    )):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = get_class_name(model, class_id)
        color = class_colors.get(class_name, SAFE_OBJECT_COLOR)
        
        draw_text_with_outline(annotated_frame, class_name, (x1, y1-10), 
                                font_scale=0.6, text_color=color.as_bgr() if hasattr(color, 'as_bgr') else (255, 255, 255))
        
        if class_name == 'person':
            if len(person_boxes) > 1 and is_child(xyxy, [box[0] for box in person_boxes]):
                child_boxes.append((xyxy, tracker_id))
                annotated_frame = draw_stable_professional_box(
                    annotated_frame, x1, y1, x2, y2, CHILD_COLOR, 4
                )
                draw_text_with_outline(annotated_frame, "CHILD", (x1, y1-30), 
                                        font_scale=0.7, text_color=CHILD_COLOR.as_bgr())
            else:
                annotated_frame = draw_stable_professional_box(
                    annotated_frame, x1, y1, x2, y2, PERSON_COLOR, 3
                )
            
            annotated_frame = trajectory_tracker.draw_trajectory(
                annotated_frame, tracker_id, (0, 255, 255), 2
            )
        
        elif class_name in DANGEROUS_OBJECTS:
            dangerous_detections.append({
                'box': xyxy,
                'class': class_name,
                'tracker_id': tracker_id
            })
            annotated_frame = draw_stable_professional_box(
                annotated_frame, x1, y1, x2, y2, DANGER_COLOR, 4
            )
            draw_text_with_outline(annotated_frame, "DANGER!", (x1, y1-30), 
                                    font_scale=0.6, text_color=DANGER_COLOR.as_bgr())
        else:
            safe_objects.append({
                'box': xyxy,
                'class': class_name,
                'tracker_id': tracker_id
            })
            annotated_frame = draw_stable_professional_box(
                annotated_frame, x1, y1, x2, y2, SAFE_OBJECT_COLOR, 2
            )
        
        label = f"ID:{tracker_id} ({confidence:.2f})"
        draw_text_with_outline(annotated_frame, label, (x1, y2+20), 
                                font_scale=0.4, text_color=(255, 255, 255))
    
    dangers = danger_detector.detect_person_danger_proximity(person_boxes, dangerous_detections)
    
    for danger in dangers:
        p_box = danger['person_box']
        d_box = danger['danger_box']
        
        p_center = (int((p_box[0] + p_box[2])/2), int((p_box[1] + p_box[3])/2))
        d_center = (int((d_box[0] + d_box[2])/2), int((d_box[1] + d_box[3])/2))
        
        cv2.line(annotated_frame, p_center, d_center, DANGER_COLOR.as_bgr(), 2)
        warning_text = f"DANGER: {danger['danger_class']} ({int(danger['distance'])}px)"
        cv2.putText(annotated_frame, warning_text, 
                    (p_center[0], p_center[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DANGER_COLOR.as_bgr(), 2)
    
    return annotated_frame, len(child_boxes), len(person_boxes) - len(child_boxes), dangers, dangerous_detections, safe_objects


def create_info_panel(frame, child_count, adult_count, danger_count, safe_count, total_objects):
    panel_height = 120
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    
    for i in range(panel_height):
        intensity = int(50 * (1 - i / panel_height))
        panel[i, :] = [intensity, intensity, intensity]
    
    stats = [
        f"Children: {child_count}",
        f"Adults: {adult_count}", 
        f"Total people: {child_count + adult_count}",
        f"Dangerous: {danger_count}",
        f"Safe: {safe_count}",
        f"Total objects: {total_objects}"
    ]
    
    for i, stat in enumerate(stats):
        if i < 3:
            draw_text_with_outline(panel, stat, (20, 25 + i*18), 
                                    font_scale=0.5, text_color=(255, 255, 255))
        else:
            draw_text_with_outline(panel, stat, (350, 25 + (i-3)*18), 
                                    font_scale=0.5, text_color=(255, 255, 255))
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
    MODEL_PATH = "models/yolo11n.pt"  
    CONFIDENCE_THRESHOLD = 0.3
    NMS_IOU_THRESHOLD = 0.4
    
    model = load_model(MODEL_PATH)
    
    class_colors = get_class_colors(model)
    trajectory_tracker = Tracker(max_length=30)
    danger_detector = DangerDetector(danger_threshold=150, alert_cooldown=10)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Kamera ochilmadi!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    monitored_class_ids = get_class_ids(model, MONITORED_CLASSES)

    frame_count = 0
    last_alert_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame o'qilmadi")
            break
        
        frame_count += 1
        
        results = model(frame, imgsz=640, verbose=False, classes=monitored_class_ids)[0]
        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections.with_nms(NMS_IOU_THRESHOLD)
        
        if len(detections) > 0:
            detections = trajectory_tracker.update(detections)
        
        annotated_frame, child_count, adult_count, dangers, dangerous_objects, safe_objects = annotate_frame(
            frame, model, detections, class_colors, trajectory_tracker, danger_detector
        )
        
        info_panel = create_info_panel(
            annotated_frame, child_count, adult_count, 
            len(dangerous_objects), len(safe_objects), len(detections))
        
        final_frame = np.vstack([annotated_frame, info_panel])
        
        if dangers:
            cv2.rectangle(final_frame, (0, 0), (final_frame.shape[1], 10), DANGER_COLOR.as_bgr(), -1)
            draw_text_with_outline(final_frame, "DANGER DETECTED!", 
                                    (20, annotated_frame.shape[0] + 60), 
                                    font_scale=0.8, text_color=DANGER_COLOR.as_bgr())
            
            current_time = time.time()
            if bot_token and chat_id and (current_time - last_alert_time > 30): 
                alert_msg = "⚠️ Xavfli holat aniqlindi!\n"
                alert_msg += f"👶 Bolalar soni: {child_count}\n"
                alert_msg += f"👨 Kattalar soni: {adult_count}\n"
                
                for danger in dangers:
                    alert_msg += (f"\n🚨 Xavf: {danger['danger_class']} "
                                f"(Masofa: {int(danger['distance'])}px)")
                
                alert_img_path = "results/alert.jpg"
                cv2.imwrite(alert_img_path, final_frame)
                
                asyncio.run(send_telegram_alert(bot_token, chat_id, alert_msg, alert_img_path))
                last_alert_time = current_time
        
        ret, buffer = cv2.imencode('.jpg', final_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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