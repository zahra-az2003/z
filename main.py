import cv2
import numpy as np

# فایل‌های مدل
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
classes_file = "coco.names"

# بارگذاری نام کلاس‌ها
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# بارگذاری مدل
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# لیست کلاس‌های مجاز
allowed_classes = class_names  # اگر خواستی بعضی رو حذف کنی اینجا تغییر بده

# تابع تشخیص شیء
def detect_objects(image, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # کاهش اندازه جعبه برای دقت بیشتر
                shrink_ratio = 0.9  # 90 درصد اندازه اولیه
                w = int(w * shrink_ratio)
                h = int(h * shrink_ratio)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

# تابع رسم جعبه و شمارش اشیا
def draw_boxes(image, boxes, class_ids, indices, class_names, allowed_classes):
    object_counts = {}
    
    for i in indices.flatten():
        label = class_names[class_ids[i]]
        
        if label not in allowed_classes:
            continue  # این شیء رو رد کن

        x, y, w, h = boxes[i]

        object_counts[label] = object_counts.get(label, 0) + 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # نمایش شمارش در گوشه تصویر
    y0 = 30
    for label, count in object_counts.items():
        text = f"{label}: {count}"
        cv2.putText(image, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y0 += 25

    return image, object_counts

# پردازش تصویر
image_path = "image.jpg"
image = cv2.imread(image_path)

boxes, confidences, class_ids, indices = detect_objects(image, net, output_layers)
output_image, object_counts = draw_boxes(image, boxes, class_ids, indices, class_names, allowed_classes)

# ذخیره خروجی
cv2.imwrite("output.jpg", output_image)

print("[✔] Detection and Objectification complete. Output saved in 'output.jpg'.")
print("Detected objects:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")
