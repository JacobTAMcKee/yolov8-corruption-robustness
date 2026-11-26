from ultralytics import YOLO

for name in ["yolov8n", "yolov8s"]:
    model = YOLO(f"{name}.pt")  # downloads weights automatically the first time
    metrics = model.val(data="subset.yaml", imgsz=640, batch=16, device=0)
    print(f"{name}: mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")