from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model = YOLO("/Users/lixizi/Documents/project/project_file/ultralytics-main-new/predict-lyz/train-small13-600-1089/weights/best.pt")

model.predict("/Users/lixiz i/Documents/project/project_file/ultralytics-main-new/predict-lyz/image/test-day10-night10", save=True)