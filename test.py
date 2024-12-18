import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

best_model_path = r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\best.pt'
model = YOLO(best_model_path)

image_dir = r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\images\val'
image_files = os.listdir(image_dir)

random_images = random.sample(image_files, 36)

fig, axs = plt.subplots(6, 6, figsize=(27, 24))

for i, image_file in enumerate(random_images):
    row = i // 6
    col = i % 6

    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model.predict(image_rgb)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_rgb, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    axs[row, col].imshow(image_rgb)
    axs[row, col].axis('off')

plt.show()