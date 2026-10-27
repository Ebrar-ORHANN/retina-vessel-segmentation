
from pathlib import Path
import sys
import csv

import cv2
import numpy as np

# Proje yolu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

IMAGE_DIR = ROOT / "data/raw/sample/images"
MASK_DIR = ROOT / "data/raw/sample/masks"

OUTPUT_DIR = ROOT / "outputs/metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Onceki deneylerle ayni ayarlar
CROP_X = 1250
CROP_Y = 450
CROP_SIZE = 512

SIGMAS = (4, 6, 8, 10)

# 0.03 ile 0.12 arasindaki esikleri test et
THRESHOLDS = tuple(
    round(i / 100, 2)
    for i in range(3, 13)
)


results = []

image_paths = sorted(IMAGE_DIR.glob("*.png"))

if not image_paths:
    raise FileNotFoundError("Retina goruntuleri bulunamadi!")

for image_path in image_paths:

    mask_path = MASK_DIR / image_path.name

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError(
            f"Goruntu veya maske okunamadi: {image_path.name}"
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Boyut hatasi: {image_path.name}")

    if not np.all(np.isin(np.unique(mask), [0, 255])):
        raise ValueError(f"Maske degerleri hatali: {image_path.name}")

    # Yesil kanal
    green = image[:, :, 1]

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green)

    # Sabit kesit
    x, y, size = CROP_X, CROP_Y, CROP_SIZE

    crop = enhanced[y:y+size, x:x+size]
    mask_crop = mask[y:y+size, x:x+size]
    original_crop = image[y:y+size, x:x+size]

    if crop.shape != (size, size):
        raise ValueError(f"Kesit hatasi: {image_path.name}")

    # Yaklasik FOV
    gray = cv2.cvtColor(
        original_crop,
        cv2.COLOR_BGR2GRAY
    )

    valid = gray > 25
    gt = mask_crop == 255

    if not np.any(valid):
        raise ValueError(f"Gecerli alan yok: {image_path.name}")

    if not np.any(gt & valid):
        raise ValueError(f"Damar bulunamadi: {image_path.name}")

    # Frangi yalnizca bir kez hesaplanir
    vesselness = apply_frangi(
        crop,
        sigmas=SIGMAS
    )

    print(f"\n--- {image_path.name} ---")

    for threshold in THRESHOLDS:

        prediction = (vesselness >= threshold) & valid

        tp = int(np.sum(prediction & gt & valid))
        fp = int(np.sum(prediction & ~gt & valid))
        fn = int(np.sum(~prediction & gt & valid))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

        results.append({
            "image": image_path.name,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou
        })

        print(
            f"T={threshold:.2f} | "
            f"Dice={dice:.4f} | "
            f"IoU={iou:.4f} | "
            f"Recall={recall:.4f}"
        )

# CSV dosyasina kaydet

csv_path = OUTPUT_DIR / "frangi_thresholds_extended.csv"


with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "threshold", "precision",
            "recall", "dice", "iou"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

# Goruntuler arasi ozet
print("\n--- 7 GORUNTU UZERINDEN OZET ---")

summary = []

for threshold in THRESHOLDS:

    selected = [
        r for r in results
        if r["threshold"] == threshold
    ]

    dice_values = [r["dice"] for r in selected]
    iou_values = [r["iou"] for r in selected]
    recall_values = [r["recall"] for r in selected]

    median_dice = float(np.median(dice_values))

    summary.append((threshold, median_dice))

    print(
        f"T={threshold:.2f} | "
        f"Medyan Dice={median_dice:.4f} | "
        f"Medyan IoU={np.median(iou_values):.4f} | "
        f"Medyan Recall={np.median(recall_values):.4f}"
    )

best = max(summary, key=lambda item: item[1])

print("\nKesit deneylerinde en yuksek medyan Dice:")
print(f"Threshold: {best[0]:.2f}")
print(f"Medyan Dice: {best[1]:.4f}")

print("\nCSV:", csv_path)
print("Analiz tamamlandi.")
