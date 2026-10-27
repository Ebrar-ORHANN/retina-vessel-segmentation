
from pathlib import Path
import sys
import csv

import cv2
import numpy as np

# ==================================================
# 1. PROJE YOLLARI
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

IMAGE_PATH = ROOT / "data/raw/sample/images/1_A.png"
MASK_PATH = ROOT / "data/raw/sample/masks/1_A.png"

OUTPUT_DIR = ROOT / "outputs/figures"
METRICS_DIR = ROOT / "outputs/metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. GORUNTU VE MASKE OKUMA
# ==================================================

image = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_COLOR
)

mask = cv2.imread(
    str(MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Goruntu bulunamadi: {IMAGE_PATH}"
    )

if mask is None:
    raise FileNotFoundError(
        f"Maske bulunamadi: {MASK_PATH}"
    )

if image.shape[:2] != mask.shape:
    raise ValueError(
        "Goruntu ve maske boyutlari eslesmiyor!"
    )

if not np.all(np.isin(np.unique(mask), [0, 255])):
    raise ValueError(
        "Maske sadece 0 ve 255 degerlerini icermeli!"
    )


# ==================================================
# 3. YESIL KANAL VE CLAHE
# ==================================================

# OpenCV BGR siralamasini kullanir.
green = image[:, :, 1]

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(green)


# ==================================================
# 4. DAMAR YOGUN KESIT
# ==================================================

CROP_X = 1250
CROP_Y = 450
CROP_SIZE = 512

x1 = CROP_X
x2 = CROP_X + CROP_SIZE

y1 = CROP_Y
y2 = CROP_Y + CROP_SIZE

crop = enhanced[y1:y2, x1:x2]

mask_crop = mask[y1:y2, x1:x2]

original_crop = image[y1:y2, x1:x2]

if crop.shape != (CROP_SIZE, CROP_SIZE):
    raise ValueError(
        "Kesit boyutu beklenen 512x512 degil!"
    )


# ==================================================
# 5. RETINA GORUS ALANI (FOV)
# ==================================================

gray = cv2.cvtColor(
    original_crop,
    cv2.COLOR_BGR2GRAY
)

# Yaklasik retina gorus alani
valid = gray > 25

# Gercek damar maskesi
ground_truth = mask_crop == 255

# Sadece gecerli alandaki damarlar
gt_valid = ground_truth & valid

if gt_valid.sum() == 0:
    raise ValueError(
        "Gecerli alanda damar pikseli bulunamadi!"
    )


# ==================================================
# 6. FRANGI FILTRESI
# ==================================================

SIGMAS = (4, 6, 8, 10)

vesselness = apply_frangi(
    crop,
    sigmas=SIGMAS
)

if vesselness.shape != crop.shape:
    raise ValueError(
        "Frangi sonucunun boyutu hatali!"
    )

if not np.all(np.isfinite(vesselness)):
    raise ValueError(
        "Frangi sonucu gecersiz deger iceriyor!"
    )

print("\n--- FRANGI BILGILERI ---")
print("Sigma:", SIGMAS)
print("Minimum:", float(vesselness.min()))
print("Maksimum:", float(vesselness.max()))
print("Ortalama:", float(vesselness.mean()))


# ==================================================
# 7. THRESHOLD DEGERLERI
# ==================================================

# 0.04 ile 0.14 arasinda 0.01 adimlarla
thresholds = [
    round(0.04 + i * 0.01, 2)
    for i in range(11)
]

print("\nTest edilecek esikler:")
print(thresholds)

results = []


# ==================================================
# 8. REFERANS GORUNTULER
# ==================================================

# Frangi sonucu sadece gorsellestirme icin
# ortak 0-255 araligina olcekleniyor.

valid_scores = vesselness[valid]

if valid_scores.size == 0:
    raise ValueError(
        "Gecerli retina pikseli bulunamadi!"
    )

scale = max(
    float(np.percentile(valid_scores, 99.5)),
    1e-8
)

frangi_view = np.clip(
    vesselness / scale * 255,
    0,
    255
).astype(np.uint8)

gt_view = gt_valid.astype(np.uint8) * 255

panels = [
    ("CLAHE", crop),
    ("FRANGI", frangi_view),
    ("GROUND TRUTH", gt_view)
]


# ==================================================
# 9. THRESHOLD TARAMASI
# ==================================================

print("\n--- FRANGI SEGMENTASYON ANALIZI ---")

for threshold in thresholds:

    # Ikili segmentasyon tahmini
    prediction = (
        (vesselness >= threshold) & valid
    )

    # Dogru damar tahminleri
    tp = int(
        np.sum(prediction & gt_valid)
    )

    # Yanlis damar tahminleri
    fp = int(
        np.sum(prediction & ~ground_truth & valid)
    )

    # Kacirilan gercek damarlar
    fn = int(
        np.sum(~prediction & gt_valid)
    )

    # ----------------------------------------------
    # PRECISION
    # ----------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # ----------------------------------------------
    # RECALL
    # ----------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # ----------------------------------------------
    # DICE
    # ----------------------------------------------

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    # ----------------------------------------------
    # IOU
    # ----------------------------------------------

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    # ----------------------------------------------
    # SONUCLARI KAYDET
    # ----------------------------------------------

    results.append({
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "dice": round(dice, 6),
        "iou": round(iou, 6)
    })

    print(f"\nThreshold: {threshold:.2f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")

    # Segmentasyon maskesini gorsellestirme
    prediction_view = (
        prediction.astype(np.uint8) * 255
    )

    panels.append((
        f"THRESHOLD {threshold:.2f}",
        prediction_view
    ))


# ==================================================
# 10. GORSEL PANELLERI HAZIRLA
# ==================================================

labeled_panels = []

PANEL_SIZE = 350
HEADER_HEIGHT = 45

for title, panel in panels:

    # Ikili maskelerde nearest neighbor kullan
    if title.startswith("THRESHOLD") or title == "GROUND TRUTH":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA

    resized = cv2.resize(
        panel,
        (PANEL_SIZE, PANEL_SIZE),
        interpolation=interpolation
    )

    panel_bgr = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2BGR
    )

    # Baslik alani
    labeled = cv2.copyMakeBorder(
        panel_bgr,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    cv2.putText(
        labeled,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    labeled_panels.append(labeled)


# ==================================================
# 11. DINAMIK GORSEL YERLESIM
# ==================================================

COLUMNS = 3

rows = []

for i in range(0, len(labeled_panels), COLUMNS):

    row_panels = labeled_panels[i:i + COLUMNS]

    # Son satir eksikse bos panel ekle
    while len(row_panels) < COLUMNS:

        blank = np.zeros_like(
            labeled_panels[0]
        )

        row_panels.append(blank)

    row_image = cv2.hconcat(row_panels)

    rows.append(row_image)

comparison = cv2.vconcat(rows)


# ==================================================
# 12. KARSILASTIRMA GORSELINI KAYDET
# ==================================================

output_path = (
    OUTPUT_DIR / "1_A_frangi_threshold_sweep.png"
)

success = cv2.imwrite(
    str(output_path),
    comparison
)

if not success:
    raise OSError(
        "Karsilastirma gorseli kaydedilemedi!"
    )


# ==================================================
# 13. CSV SONUCLARINI KAYDET
# ==================================================

csv_path = (
    METRICS_DIR / "1_A_frangi_threshold_sweep.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "threshold",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "dice",
            "iou"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ==================================================
# 14. EN YUKSEK DICE DEGERINI BUL
# ==================================================

best = max(
    results,
    key=lambda result: result["dice"]
)

print("\n================================")
print("   EN YUKSEK DICE SONUCU")
print("================================")

print(f"Threshold : {best['threshold']:.2f}")
print(f"Precision : {best['precision']:.4f}")
print(f"Recall    : {best['recall']:.4f}")
print(f"Dice      : {best['dice']:.4f}")
print(f"IoU       : {best['iou']:.4f}")

print("\n================================")
print("   ANALIZ TAMAMLANDI")
print("================================")

print("Gorsel:", output_path)
print("CSV:", csv_path)
