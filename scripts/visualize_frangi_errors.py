
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

# Dosyalari oku
image = cv2.imread(str(
    ROOT / "data/raw/sample/images/1_A.png"
))

mask = cv2.imread(
    str(ROOT / "data/raw/sample/masks/1_A.png"),
    cv2.IMREAD_GRAYSCALE
)

if image is None or mask is None:
    raise FileNotFoundError("Goruntu veya maske bulunamadi!")

if image.shape[:2] != mask.shape:
    raise ValueError("Goruntu ve maske boyutlari farkli!")

# CLAHE'yi tam goruntuye uygula
green = image[:, :, 1]

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(green)

# Onceki deneydeki ayni kesit
x, y, size = 1250, 450, 512

crop = enhanced[y:y+size, x:x+size]
original_crop = image[y:y+size, x:x+size]
mask_crop = mask[y:y+size, x:x+size]

if crop.shape != (size, size):
    raise ValueError("Kesit boyutu hatali!")

# Frangi damar benzerligi
vesselness = apply_frangi(
    crop,
    sigmas=(4, 6, 8, 10)
)

# Baslangic esigimiz
threshold = 0.08

# Gorus alani
gray = cv2.cvtColor(
    original_crop,
    cv2.COLOR_BGR2GRAY
)

valid = gray > 25

ground_truth = mask_crop == 255
prediction = (vesselness >= threshold) & valid

# Dogru ve yanlis tahminler
tp = prediction & ground_truth & valid
fp = prediction & ~ground_truth & valid
fn = ~prediction & ground_truth & valid

# Siyah zemin uzerine hata haritasi
error_map = np.zeros_like(original_crop)

# OpenCV BGR renk siralamasi
error_map[tp] = (0, 255, 0)    # Yesil
error_map[fp] = (0, 0, 255)    # Kirmizi
error_map[fn] = (255, 0, 0)    # Mavi

# Gercek damar maskesini goster
gt_view = np.zeros_like(original_crop)
gt_view[ground_truth & valid] = (255, 255, 255)

# Panelleri yan yana yerlestir
comparison = cv2.hconcat([
    original_crop,
    gt_view,
    error_map
])

# Sonucu kaydet
output_dir = ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "1_A_frangi_error_map.png"

if not cv2.imwrite(str(output_path), comparison):
    raise OSError("Hata haritasi kaydedilemedi!")

print("Dogru damar (TP):", int(tp.sum()))
print("Yanlis damar (FP):", int(fp.sum()))
print("Kacirilan damar (FN):", int(fn.sum()))
print("Hata haritasi:", output_path)
