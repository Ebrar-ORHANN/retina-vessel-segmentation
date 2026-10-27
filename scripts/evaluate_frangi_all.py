
from pathlib import Path
import sys
import csv

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

IMAGE_DIR = ROOT / "data/raw/sample/images"
MASK_DIR = ROOT / "data/raw/sample/masks"

OUTPUT_DIR = ROOT / "outputs/metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sabit deney parametreleri
CROP_X = 1250
CROP_Y = 450
CROP_SIZE = 512

SIGMA_SETS = [
    (1, 2, 3, 4),
    (2, 4, 6, 8),
    (4, 6, 8, 10)
]

results = []

image_paths = sorted(IMAGE_DIR.glob("*.png"))

if not image_paths:
    raise FileNotFoundError("Goruntu bulunamadi!")

for image_path in image_paths:

    mask_path = MASK_DIR / image_path.name

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError(
            f"Goruntu veya maske okunamadi: {image_path.name}"
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Boyut uyusmazligi: {image_path.name}"
        )

    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green)

    y1, y2 = CROP_Y, CROP_Y + CROP_SIZE
    x1, x2 = CROP_X, CROP_X + CROP_SIZE

    crop = enhanced[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    original_crop = image[y1:y2, x1:x2]

    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError("Kesit boyutu hatali!")

    if not np.all(np.isin(np.unique(mask_crop), [0, 255])):
        raise ValueError(
            f"Beklenmeyen maske degeri: {image_path.name}"
        )

    # Yaklasik retina gorus alani
    gray = cv2.cvtColor(
        original_crop,
        cv2.COLOR_BGR2GRAY
    )

    valid_pixels = gray > 25

    y_true = (
        mask_crop[valid_pixels] == 255
    ).astype(np.uint8)

    if np.unique(y_true).size != 2:
        print(
            f"{image_path.name}: Iki sinif bulunamadi, atlandi."
        )
        continue

    vessel_ratio = float(y_true.mean())

    print(f"\n--- {image_path.name} ---")
    print(f"Damar orani: %{vessel_ratio * 100:.2f}")

    for sigmas in SIGMA_SETS:

        response = apply_frangi(
            crop,
            sigmas=sigmas
        )

        scores = response[valid_pixels]

        roc_auc = roc_auc_score(
            y_true, scores
        )

        ap = average_precision_score(
            y_true, scores
        )

        results.append({
            "image": image_path.name,
            "sigmas": str(sigmas),
            "vessel_ratio": round(vessel_ratio, 6),
            "roc_auc": round(float(roc_auc), 6),
            "average_precision": round(float(ap), 6)
        })

        print(
            f"Sigma {sigmas} | "
            f"ROC-AUC: {roc_auc:.4f} | "
            f"AP: {ap:.4f}"
        )

# Sonuclari CSV olarak kaydet
output_path = OUTPUT_DIR / "frangi_sigma_results.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "sigmas",
            "vessel_ratio",
            "roc_auc",
            "average_precision"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

# Goruntuler arasi ozet
print("\n--- GENEL SONUCLAR ---")

for sigmas in SIGMA_SETS:

    method_rows = [
        row for row in results
        if row["sigmas"] == str(sigmas)
    ]

    if not method_rows:
        continue

    auc_values = [
        row["roc_auc"] for row in method_rows
    ]

    ap_values = [
        row["average_precision"] for row in method_rows
    ]

    print(f"\nSigma: {sigmas}")
    print(f"Goruntu sayisi: {len(method_rows)}")
    print(f"Medyan ROC-AUC: {np.median(auc_values):.4f}")
    print(f"Medyan AP: {np.median(ap_values):.4f}")

print("\nAnaliz tamamlandi!")
print("CSV dosyasi:", output_path)
