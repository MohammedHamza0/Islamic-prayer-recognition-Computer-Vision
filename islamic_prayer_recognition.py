import os
import cv2
from ultralytics import YOLO
import time
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from PIL import ImageFont, ImageDraw, Image

os.chdir(r"F:\Computer Vision\YOLO Projects\IslamicPray")

model = YOLO("IslamicBest.pt")

TheClass = model.model.names
print(TheClass)

def draw_text_with_background(frame, text, position, font_path, font_size, text_color, background_color, border_color, thickness=2, padding=5):
    font = ImageFont.truetype(font_path, font_size)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    text_bbox = draw.textbbox(position, text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x, y = position
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], fill=background_color)
    draw.text((x, y), text, font=font, fill=text_color)
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], outline=border_color, width=thickness)
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    return frame

cap = cv2.VideoCapture("istockphoto-1345393460-640_adpp_is.mp4")

NumberOfRuku = 0
TheFullRuku = 4
was_in_ruku = False
ruku_start_time = 0
ruku_counted = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("The frames have been finished")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        
        text = get_display(reshape(f"الفجر 2 ركعه"))
        frame = draw_text_with_background(frame, 
                    text, 
                    (900, 10), 
                    "arial.ttf",  
                    40, 
                    (0, 0, 0),  
                    (0, 165, 255),  
                    (0, 0, 255)) 
        
        text = get_display(reshape(f"المغرب 3 ركعات"))
        frame = draw_text_with_background(frame, 
                    text, 
                    (850, 60), 
                    "arial.ttf",  
                    40, 
                    (0, 0, 0),  
                    (0, 165, 255),  
                    (0, 0, 255))  
        
        text = get_display(reshape(f"الظهر و العصر و العشاء 4 ركعات"))
        frame = draw_text_with_background(frame, 
                    text, 
                    (630, 110), 
                    "arial.ttf",  
                    40, 
                    (0, 0, 0),  
                    (0, 165, 255),  
                    (0, 0, 255)) 
        
        results = model.predict(frame, conf=0.35)
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confs = result.boxes.conf
            
            for box, cls, conf in zip(boxes, classes, confs):
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                cls = int(cls)
                cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                
                if TheClass[cls] == "raising":
                    text = get_display(reshape(f"قائم"))
                    frame = draw_text_with_background(frame, 
                                        text, 
                                        (x, y), 
                                        "arial.ttf", 
                                        70, 
                                        (255, 255, 255),  
                                        (0, 0, 0),  
                                        (0, 0, 255))  
                if TheClass[cls] == "ruku":
                    text = get_display(reshape(f"راكع"))
                    frame = draw_text_with_background(frame, 
                                        text, 
                                        (x, y), 
                                        "arial.ttf", 
                                        70, 
                                        (255, 255, 255),  
                                        (0, 0, 0),  
                                        (0, 0, 255))  
                    
                if TheClass[cls] == "sujud":
                    text = get_display(reshape(f"ساجد"))
                    frame = draw_text_with_background(frame, 
                                        text, 
                                        (x, y), 
                                        "arial.ttf", 
                                        70, 
                                        (255, 255, 255),  
                                        (0, 0, 0),  
                                        (0, 0, 255))  
                
                if TheClass[cls] == "tashhud":
                    text = get_display(reshape(f"التشهد"))
                    frame = draw_text_with_background(frame, 
                                        text, 
                                        (x, y), 
                                        "arial.ttf", 
                                        70, 
                                        (255, 255, 255),  
                                        (0, 0, 0),  
                                        (0, 0, 255))  
                    
                if TheClass[cls] == "takbeer":
                    text = get_display(reshape(f"تكبير"))
                    frame = draw_text_with_background(frame, 
                                        text, 
                                        (x, y), 
                                        "arial.ttf", 
                                        70, 
                                        (255, 255, 255),  
                                        (0, 0, 0),  
                                        (0, 0, 255))  
             
            current_ruku = False 
            text = get_display(reshape(f"عدد مرات الركوع: {NumberOfRuku}"))
            frame = draw_text_with_background(frame, 
                    text, 
                    (30, 10), 
                    "arial.ttf",  
                    40, 
                    (0, 0, 0),  
                    (165, 0, 255),  
                    (0, 0, 255))  

            if TheClass[cls] == "ruku":
                current_ruku = True
                if not was_in_ruku:
                    ruku_start_time = time.time()  
                    was_in_ruku = True
                    ruku_counted = False  
                elif not ruku_counted and time.time() - ruku_start_time >= 10:  
                    NumberOfRuku += 1
                    ruku_counted = True  

            if not current_ruku:
                was_in_ruku = False
                ruku_counted = False  
            
        cv2.imshow("IslamicPray", frame)
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()
