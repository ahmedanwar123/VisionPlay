from ultralytics import YOLO

model = YOLO("yolov8_n_custom/train_20241227_0947/best.pt")  # Load model
results = model.predict("input_videos/1.mp4", save=True)  # Inference
print(results[0])  # print results
print("======================================")
print("======================================")

for box in results[0].boxes:
    print(box)
