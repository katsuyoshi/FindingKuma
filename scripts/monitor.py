#!/usr/bin/env python3
"""
FindingKuma - Periodic monitoring script for bear detection.

Self-contained Python script for GitHub Actions / cron use.
Fetches camera images, runs YOLOv8 detection, and sends
Discord notifications on wildlife detections.

Usage:
    python3 scripts/monitor.py --discord-webhook URL
    python3 scripts/monitor.py --discord-webhook URL --hours 1
    python3 scripts/monitor.py --source all --dry-run

Environment variables:
    DISCORD_WEBHOOK_URL  - Discord webhook URL (alternative to --discord-webhook)
"""

import argparse
import io
import json
import os
import sys
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

_real_stdout = sys.stdout
sys.stdout = io.StringIO()
from ultralytics import YOLO
sys.stdout = _real_stdout

os.environ["YOLO_VERBOSE"] = "false"

import cv2

# Import detection logic from detect.py
sys.path.insert(0, str(Path(__file__).parent))
from detect import CLASS_PRESETS, COCO_NAMES, CLASS_COLORS, DEFAULT_COLOR

JST = timezone(timedelta(hours=9))

# --- Camera sources ---

RIVER_SITES_URL = "https://kasen.pref.akita.lg.jp/pc/data/itv.json"
RIVER_IMAGE_BASE = "https://kasenimg.pref.akita.lg.jp/cameraDataWeb/itv"

ROAD_CAMERAS = {
    "r7c1":  "国道7号 小砂川",
    "r7c2":  "国道7号 竹嶋",
    "r7c3":  "国道7号 三川",
    "r7c4":  "国道7号 勝手",
    "r7c5":  "国道7号 中野",
    "r7c6":  "国道7号 三倉鼻",
    "r7c7":  "国道7号 真坂",
    "r13c1": "国道13号 船沢",
    "r13c2": "国道13号 雨池沢",
    "r13c3": "国道13号 上淀川C",
    "r46c1": "国道46号 刺巻",
    "r46c2": "国道46号 小松",
    "r46c4": "国道46号 荒川",
    "r46c7": "国道46号 木滝沢",
    "r46c10": "国道46号 須神",
    "r46c12": "国道46号 仙岩トンネル岩手側",
}
ROAD_IMAGE_BASE = "https://www.thr.mlit.go.jp/akita/douro/ROMEN/photo"


def fetch_river_sites():
    """Fetch river camera site list from JSON API."""
    req = urllib.request.Request(RIVER_SITES_URL)
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()
    text = raw.decode("shift_jis")
    data = json.loads(text)
    data.pop("date", None)

    sites = []
    for site_id, info in data.items():
        sites.append({
            "id": site_id.zfill(3),
            "name": info.get("an", ""),
            "river": info.get("rn", ""),
            "source": "river",
        })
    return sorted(sites, key=lambda s: s["id"])


def build_river_schedule(site_id, hours):
    """Build list of (timestamp, url) for the past N hours at 5-min intervals."""
    now = datetime.now(JST)
    rounded_min = (now.minute // 5) * 5
    current = now.replace(minute=rounded_min, second=0, microsecond=0)
    sn = site_id.zfill(3)

    schedule = []
    slots = int(hours * 12)  # 12 slots per hour (every 5 min)
    for i in range(slots):
        t = current - timedelta(minutes=i * 5)
        date_str = t.strftime("%Y%m%d")
        time_str = t.strftime("%Y%m%d%H%M")
        url = f"{RIVER_IMAGE_BASE}/{date_str}/{sn}/image_{sn}_{time_str}.jpg"
        schedule.append((t, url))

    return schedule


def download_image(url, dest_path):
    """Download an image from URL. Returns True on success."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                data = resp.read()
                if len(data) > 1000:  # skip tiny error responses
                    with open(dest_path, "wb") as f:
                        f.write(data)
                    return True
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return False


def send_discord_notification(webhook_url, camera_name, detections, image_path,
                              source_type, timestamp=None):
    """Send a Discord webhook notification with detection results."""
    det_list = ", ".join(f"**{d['class']}** ({d['confidence']:.0%})" for d in detections)
    ts_str = timestamp.strftime("%Y/%m/%d %H:%M") if timestamp else datetime.now(JST).strftime("%Y/%m/%d %H:%M")

    content = (
        f"🐻 **検知アラート** - {ts_str} JST\n"
        f"📍 {camera_name} ({source_type})\n"
        f"🔍 {det_list}"
    )

    # Discord webhook with file upload
    boundary = "----FindingKumaBoundary"
    body = []

    # JSON payload part
    payload = json.dumps({"content": content})
    body.append(f"--{boundary}")
    body.append('Content-Disposition: form-data; name="payload_json"')
    body.append("Content-Type: application/json")
    body.append("")
    body.append(payload)

    # File part
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            file_data = f.read()
        body.append(f"--{boundary}")
        body.append(f'Content-Disposition: form-data; name="file"; filename="{Path(image_path).name}"')
        body.append("Content-Type: image/jpeg")
        body.append("")

        # Build multipart body as bytes
        text_part = "\r\n".join(body).encode("utf-8") + b"\r\n"
        end_part = f"\r\n--{boundary}--\r\n".encode("utf-8")
        full_body = text_part + file_data + end_part
    else:
        body.append(f"--{boundary}--")
        full_body = "\r\n".join(body).encode("utf-8")

    req = urllib.request.Request(
        webhook_url,
        data=full_body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 204)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  Discord notification failed: {e}", file=sys.stderr)
        return False


def annotate_image(image_path, detections, save_path):
    """Draw detection boxes on image and save."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

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

    cv2.imwrite(str(save_path), img)


def run_detection(model, image_path, target_classes, confidence):
    """Run YOLO detection on a single image."""
    results = model(str(image_path), conf=confidence, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in target_classes:
                continue
            cls_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    return detections


def main():
    parser = argparse.ArgumentParser(description="FindingKuma periodic monitor")
    parser.add_argument("--discord-webhook", default=os.environ.get("DISCORD_WEBHOOK_URL"),
                        help="Discord webhook URL")
    parser.add_argument("--source", default="river", choices=["river", "road", "all"],
                        help="Camera source (default: river)")
    parser.add_argument("--hours", type=float, default=1,
                        help="Hours of past images to scan (default: 1)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--classes", default="japan",
                        help="Class preset (default: japan)")
    parser.add_argument("--notify-classes", default="bear,wildlife",
                        help="Classes to notify on (comma-separated, default: bear,wildlife)")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save detected images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run detection but don't send notifications")
    args = parser.parse_args()

    if not args.discord_webhook and not args.dry_run:
        print("Error: --discord-webhook or DISCORD_WEBHOOK_URL required", file=sys.stderr)
        sys.exit(1)

    # Resolve class presets
    target_classes = CLASS_PRESETS.get(args.classes)
    if target_classes is None:
        try:
            target_classes = [int(c) for c in args.classes.split(",")]
        except ValueError:
            print(f"Error: Unknown class preset: {args.classes}", file=sys.stderr)
            sys.exit(1)

    # Build set of notification-worthy class IDs
    notify_class_ids = set()
    for preset_name in args.notify_classes.split(","):
        preset_name = preset_name.strip()
        if preset_name in CLASS_PRESETS:
            notify_class_ids.update(CLASS_PRESETS[preset_name])
        else:
            try:
                notify_class_ids.add(int(preset_name))
            except ValueError:
                pass

    results_dir = Path(args.results_dir) if args.results_dir else None
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    now_str = datetime.now(JST).strftime("%Y/%m/%d %H:%M JST")
    slots = int(args.hours * 12)

    scanned = 0
    detected = 0
    notified = 0
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- River cameras: batch scan past N hours ---
        if args.source in ("river", "all"):
            print("Fetching river camera list...")
            try:
                sites = fetch_river_sites()
            except Exception as e:
                print(f"  Failed to fetch river sites: {e}", file=sys.stderr)
                sites = []

            total = len(sites) * slots
            print(f"  {len(sites)} sites x {slots} images = {total} images")
            print(f"  Range: past {args.hours}h (5-min intervals)")
            print()

            count = 0
            for site in sites:
                schedule = build_river_schedule(site["id"], args.hours)
                cam_name = f"{site['river']} {site['name']}"

                for t, url in schedule:
                    count += 1
                    time_label = t.strftime("%H:%M")
                    fname = f"{site['id']}_{t.strftime('%Y%m%d%H%M')}.jpg"
                    img_path = Path(tmpdir) / fname

                    if not download_image(url, str(img_path)):
                        errors += 1
                        continue

                    scanned += 1
                    detections = run_detection(model, str(img_path), target_classes, args.confidence)

                    if not detections:
                        continue

                    detected += 1
                    det_str = ", ".join(f"{d['class']}({d['confidence']})" for d in detections)
                    progress = f"[{count}/{total}]"
                    print(f"  {progress} {site['id']} {cam_name} {time_label}: {det_str}")

                    # Check if any detection is notification-worthy
                    notify_dets = [d for d in detections if d["class_id"] in notify_class_ids]

                    if notify_dets:
                        annotated_path = Path(tmpdir) / f"detected_{fname}"
                        annotate_image(str(img_path), detections, str(annotated_path))

                        if results_dir:
                            save_path = results_dir / f"detected_{fname}"
                            annotate_image(str(img_path), detections, str(save_path))

                        if args.discord_webhook and not args.dry_run:
                            ok = send_discord_notification(
                                args.discord_webhook, cam_name, notify_dets,
                                str(annotated_path), "river", timestamp=t,
                            )
                            if ok:
                                notified += 1
                                print(f"    -> Discord notification sent")
                            else:
                                print(f"    -> Discord notification failed")

                    # Clean up image to save disk
                    if img_path.exists() and not detections:
                        img_path.unlink()

        # --- Road cameras: current image only ---
        if args.source in ("road", "all"):
            print(f"Scanning {len(ROAD_CAMERAS)} road cameras (current image)...")

            for cam_id, cam_name in ROAD_CAMERAS.items():
                url = f"{ROAD_IMAGE_BASE}/{cam_id}.jpg"
                img_path = Path(tmpdir) / f"{cam_id}.jpg"

                if not download_image(url, str(img_path)):
                    errors += 1
                    continue

                scanned += 1
                detections = run_detection(model, str(img_path), target_classes, args.confidence)

                if not detections:
                    continue

                detected += 1
                det_str = ", ".join(f"{d['class']}({d['confidence']})" for d in detections)
                print(f"  {cam_id} {cam_name}: {det_str}")

                notify_dets = [d for d in detections if d["class_id"] in notify_class_ids]

                if notify_dets:
                    annotated_path = Path(tmpdir) / f"detected_{cam_id}.jpg"
                    annotate_image(str(img_path), detections, str(annotated_path))

                    if results_dir:
                        ts = datetime.now(JST).strftime("%Y%m%d%H%M")
                        save_path = results_dir / f"detected_{cam_id}_{ts}.jpg"
                        annotate_image(str(img_path), detections, str(save_path))

                    if args.discord_webhook and not args.dry_run:
                        ok = send_discord_notification(
                            args.discord_webhook, cam_name, notify_dets,
                            str(annotated_path), "road",
                        )
                        if ok:
                            notified += 1

    print()
    print("=" * 50)
    print(f"Scan complete: {now_str}")
    print(f"  Scanned: {scanned}")
    print(f"  Detected: {detected}")
    print(f"  Notified: {notified}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
