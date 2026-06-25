#!/usr/bin/env python3
"""
FindingKuma - Object detection using YOLOv8 with static-object filtering.

Usage:
    python3 scripts/detect.py <image_path> [--baseline <prev_image>] [--confidence 0.3] [--save-img]

When --baseline is provided, detections that also appeared at the same
position in the baseline image are filtered out (static objects like
buildings, text overlays, and bridge structures).
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
from ultralytics import YOLO
sys.stdout = _real_stdout

os.environ["YOLO_VERBOSE"] = "false"

import cv2

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

CLASS_PRESETS = {
    "bear": [21],
    "person": [0],
    "vehicle": [2, 3, 5, 7],
    "wildlife": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "all": [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # Japan-realistic: excludes giraffe, zebra, elephant (common false positives from text overlays)
    "japan": [0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19, 21],
}

CLASS_COLORS = {
    "person": (0, 200, 0),
    "bear": (0, 0, 255),
    "car": (255, 150, 0),
    "truck": (255, 150, 0),
    "bus": (255, 150, 0),
    "motorcycle": (255, 150, 0),
    "bicycle": (255, 150, 0),
}
DEFAULT_COLOR = (0, 200, 255)

# Static detection filtering parameters
STATIC_IOU_THRESHOLD = 0.5  # IoU threshold to consider two detections as "same position"


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def detect_baseline(baseline_path, confidence, target_classes, model):
    """Run detection on the baseline image and return detections list."""
    results = model(str(baseline_path), conf=confidence, verbose=False)
    baseline_dets = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in target_classes:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            baseline_dets.append({
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2],
            })
    return baseline_dets


def is_static_detection(det_bbox, det_cls_id, baseline_dets):
    """Check if a detection matches a baseline detection (same class, similar position)."""
    for bdet in baseline_dets:
        if bdet["class_id"] == det_cls_id:
            if compute_iou(det_bbox, bdet["bbox"]) >= STATIC_IOU_THRESHOLD:
                return True
    return False


def detect(image_path, baseline_path=None, confidence=0.3, target_classes=None,
           save_img=False, results_dir="results"):
    if target_classes is None:
        target_classes = CLASS_PRESETS["all"]

    model = YOLO("yolov8n.pt")
    results = model(image_path, conf=confidence, verbose=False)

    # Run detection on baseline for static-object filtering
    baseline_dets = None
    if baseline_path and Path(baseline_path).exists():
        baseline_dets = detect_baseline(baseline_path, confidence, target_classes, model)

    detections = []
    filtered_count = 0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in target_classes:
                continue

            cls_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

            # Filter static detections: skip if same class at same position in baseline
            if baseline_dets is not None:
                if is_static_detection([x1, y1, x2, y2], cls_id, baseline_dets):
                    filtered_count += 1
                    continue

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": bbox,
            })

    output = {
        "image_path": str(image_path),
        "detections": detections,
        "detection_count": len(detections),
    }
    if baseline_dets is not None:
        output["diff_filtered"] = filtered_count
        output["baseline_path"] = str(baseline_path)

    if save_img and detections:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            cls_name = det["class"]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        save_path = results_path / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(save_path), img)
        output["saved_image"] = str(save_path)

    return output


def main():
    parser = argparse.ArgumentParser(description="Detect objects in images using YOLOv8")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--baseline", default=None, help="Baseline image for diff filtering")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--classes", default="all", help="Class preset: bear, person, vehicle, wildlife, japan, all")
    parser.add_argument("--save-img", action="store_true", help="Save annotated image to results/")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    target_classes = CLASS_PRESETS.get(args.classes)
    if target_classes is None:
        try:
            target_classes = [int(c) for c in args.classes.split(",")]
        except ValueError:
            print(json.dumps({"error": f"Unknown class preset: {args.classes}"}))
            sys.exit(1)

    output = detect(
        str(image_path),
        baseline_path=args.baseline,
        confidence=args.confidence,
        target_classes=target_classes,
        save_img=args.save_img,
        results_dir=args.results_dir,
    )

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
