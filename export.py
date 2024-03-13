from ultralytics import YOLO
# Load model
model = YOLO("/root/autodl-tmp/project/ultralytics/runs/segment/train3/weights/best.pt")

# Export model
success = model.export(task="segment", format="onnx", opset=12, imgsz=640, simplify=True)

