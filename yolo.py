from ultralytics import  YOLO
import pycocotools
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import os
import yaml
import torch
import shutil
import wandb
from IPython.display import Image, clear_output
import matplotlib.pyplot as plt


train_image_dir = r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\training_images'
test_image_dir =r'.venv\data\data\testing_images'

train_csv_path = r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\train_solution_bounding_boxes (1).csv'
test_csv_path=r'.venv/data/data/sample_submission.csv'

images_dir=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\images'
labels_dir=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\labels'

df = pd.read_csv(train_csv_path)
imgs_list=list(sorted(os.listdir((train_image_dir))))
idxs=list(range(len(imgs_list)))
np.random.shuffle(idxs)
train_idx=idxs[:int(0.8*len(idxs))]
val_idx=idxs[int(0.8*len(idxs)):]

width=676
height=380
df["class"]=0
df.rename(columns={'image':'img_name'}, inplace=True)

df["x_centre"]=(df["xmin"]+df["xmax"])/(2*width)
df["y_centre"]=(df["ymin"]+df["ymax"])/(2*height)
df["width"]=(df["xmax"]-df["xmin"])/width
df["height"]=(df["ymax"]-df["ymin"])/height
df_yolo=df[["img_name","class","x_centre","y_centre","width","height"]]

print(df_yolo.head())
for idx, img_name in enumerate(imgs_list):
    subset = "train"
    if idx in val_idx:
        subset = "val"

    if np.isin(img_name, df_yolo["img_name"]):
        columns = ["class", "x_centre", "y_centre", "width", "height"]
        img_bbox = df_yolo[df_yolo["img_name"] == img_name][columns].values

        label_file_path = os.path.join(labels_dir, subset, img_name[:-4] + ".txt")
        with open(label_file_path, "w+") as f:
            for row in img_bbox:
                text = " ".join(row.astype(str))
                f.write(text)
                f.write("\n")
    old_image_path = os.path.join(train_image_dir, img_name)
    new_image_path = os.path.join(images_dir, subset, img_name)
    shutil.copy(old_image_path, new_image_path)
class_names = ['car']
yolo_format = dict(path=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working',
                   train=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\images\train',
                   val=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\images\val',
                   nc=1,
                   names={0: "car"})

with open(r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\yolo.yaml', 'w') as outfile:
    yaml.dump(yolo_format, outfile, default_flow_style=False)
model=YOLO('yolov8s.pt')
model.train(data=r'C:\Users\vanho\PycharmProjects\pythonProject2\.venv\data\data\working\yolo.yaml',epochs=50,patience=5,batch=16,
                    lr0=0.0005,imgsz=640)
metrics = model.val(split='val')

print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")
print("recall : ",metrics.box.r)
print(metrics.box.f1)
print(metrics.box.maps)