
from pathlib import Path
import sys

import cv2
import numpy as np

# Proje yolu
ROOT = Path(__file__).resolve().parents[1]

# Mevcut Frangi fonksiyonumuzu yeniden kullan
sys.path.insert(0, str(ROOT / "src"))

from retina_seg.preprocessing.frangi_filter import apply_frangi

# Goruntuleri oku
image_path = ROOT / "data/raw/sample/images/1_A.png"
mask_path = ROOT / "data/raw/sample/masks/1_A.png"

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Goruntu veya maske bulunamadi!")

if image.shape[:2] != mask.shape:
    raise ValueError("Goruntu ve maske boyutlari farkli!")

# Yesil kanal
green = image[:, :, 1]

# CLAHE ayarimiz sabit
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(green)

# Onceki deneyle ayni merkez kesit
h, w = green.shape
size = 512

y = (h - size) // 2
x = (w - size) // 2

crop = enhanced[y:y+size, x:x+size]
mask_crop = mask[y:y+size, x:x+size]

# Farkli sigma gruplari
sigma_sets = [
    (1, 2, 3, 4),
    (2, 4, 6, 8),
    (4, 6, 8, 10)
]

responses = []

for sigmas in sigma_sets:

    vesselness = apply_frangi(
        crop,
        sigmas=sigmas
    )

    responses.append(vesselness)

    print(
        f"Sigmas: {sigmas} | "
        f"Min: {vesselness.min():.6f} | "
        f"Max: {vesselness.max():.6f} | "
        f"Mean: {vesselness.mean():.6f}"
    )

# Tum haritalar icin ortak gorsellestirme olcegi
scale = max(
    float(np.percentile(r, 99.5))
    for r in responses
)

scale = max(scale, 1e-8)

panels = [
    ("CLAHE", crop)
]

for sigmas, response in zip(sigma_sets, responses):

    view = np.clip(
        response / scale * 255,
        0,
        255
    ).astype(np.uint8)

    title = "SIGMA " + ",".join(map(str, sigmas))

    panels.append((title, view))

panels.append(("GROUND TRUTH", mask_crop))

# Goruntuleri etiketle
labeled_panels = []

for title, panel in panels:

    panel_bgr = cv2.cvtColor(
        panel,
        cv2.COLOR_GRAY2BGR
    )

    labeled = cv2.copyMakeBorder(
        panel_bgr,
        45, 0, 0, 0,
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
        2
    )

    labeled_panels.append(labeled)

comparison = cv2.hconcat(labeled_panels)

# Kaydet
output_dir = ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "1_A_frangi_sigmas.png"

if not cv2.imwrite(str(output_path), comparison):
    raise OSError("Gorsel kaydedilemedi!")

print("\nSigma karsilastirmasi tamamlandi.")
print("Sonuc:", output_path)
