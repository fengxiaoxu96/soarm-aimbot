import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

sequence = Path("output/step05/sequence")
mask_dir = Path("output/step05/masks")
mask_dir.mkdir(parents=True, exist_ok=True)
frames = yaml.safe_load((sequence / "timestamps.yaml").read_text(encoding="utf-8"))["frames"]
model = YOLO("yolo11l-seg.pt")
wanted = {"cube", 'laptop'}
confidence_floor = 0.1

for item in frames:
    image = cv2.imread(str(sequence / item["color"]))
    result = model(image, conf=confidence_floor, verbose=False)[0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if result.boxes is not None and len(result.boxes) > 0:
        classes = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()
        if result.masks is not None:
            for index, (class_id, score) in enumerate(zip(classes, scores)):
                if result.names[int(class_id)] not in wanted or score < confidence_floor:
                    continue
                piece = result.masks.data[index].cpu().numpy()
                piece = cv2.resize(
                    piece, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                mask[piece > 0.5] = 255
        else:
            for box, class_id, score in zip(result.boxes.xyxy.tolist(), classes, scores):
                if result.names[int(class_id)] not in wanted or score < confidence_floor:
                    continue
                x1, y1, x2, y2 = (int(round(value)) for value in box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    cv2.imwrite(str(mask_dir / f"frame_{int(item['index']):05d}.png"), mask)