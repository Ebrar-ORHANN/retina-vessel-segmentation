
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Proje klasoru
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

# Goruntu ve maskeyi oku
image_path = ROOT / "data/raw/sample/images/2_A.png"
mask_path = ROOT / "data/raw/sample/masks/2_A.png"

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Goruntu veya maske bulunamadi!")

if image.shape[:2] != mask.shape:
    raise ValueError("Goruntu ve maske boyutlari farkli!")

# Yesil kanal ve CLAHE
green = image[:, :, 1]

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(green)

# Onceki deneyle ayni kesit
x, y, size = 1250, 450, 512

crop = enhanced[y:y+size, x:x+size]
mask_crop = mask[y:y+size, x:x+size]
original_crop = image[y:y+size, x:x+size]

if crop.shape != (size, size):
    raise ValueError("Kesit boyutu hatali!")

if not np.all(np.isin(np.unique(mask_crop), [0, 255])):
    raise ValueError("Maske 0 ve 255 disinda deger iceriyor!")

# Yaklasik retina gorus alani
gray_crop = cv2.cvtColor(
    original_crop, cv2.COLOR_BGR2GRAY
)

valid_pixels = gray_crop > 25

# Gercek damar etiketleri
y_true = (mask_crop[valid_pixels] == 255).astype(np.uint8)

if np.unique(y_true).size != 2:
    raise ValueError("Hem damar hem arka plan bulunmali!")

print("Gecerli piksel:", len(y_true))
print("Damar orani:", round(float(y_true.mean() * 100), 2), "%")

# Frangi sigma gruplari
sigma_sets = [
    (1, 2, 3, 4),
    (2, 4, 6, 8),
    (4, 6, 8, 10)
]

print("\n--- FRANGI PERFORMANS ANALIZI ---")

for sigmas in sigma_sets:

    # Frangi yanit haritasi
    response = apply_frangi(
        crop,
        sigmas=sigmas
    )

    # Yalnizca retina alanini degerlendir
    scores = response[valid_pixels]

    # Gercek damar ve arka plan skorlari
    vessel_scores = scores[y_true == 1]
    background_scores = scores[y_true == 0]

    # Performans metrikleri
    roc_auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    print(f"\nSigma: {sigmas}")

    print(
        f"Damar ortalama skoru: "
        f"{vessel_scores.mean():.6f}"
    )

    print(
        f"Arka plan ortalama skoru: "
        f"{background_scores.mean():.6f}"
    )

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")

print("\nAnaliz tamamlandi.")
