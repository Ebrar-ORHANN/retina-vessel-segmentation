
from pathlib import Path
from statistics import median

import cv2
import numpy as np

# 1. Proje yollari
ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/raw/sample/images/1_A.png"
mask_path = ROOT / "data/raw/sample/masks/1_A.png"

# 2. Goruntu ve maske oku
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Goruntu veya maske bulunamadi!")

if image.shape[:2] != mask.shape:
    raise ValueError("Goruntu ve maske boyutlari farkli!")

if not np.all(np.isin(np.unique(mask), [0, 255])):
    raise ValueError("Maske sadece 0 ve 255 icermeli!")

# 3. Yesil kanal
green = image[:, :, 1]

# 4. Iki CLAHE sonucu
clahe_2 = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
).apply(green)

clahe_3 = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
).apply(green)

variants = {
    "Green": green,
    "CLAHE 2.0": clahe_2,
    "CLAHE 3.0": clahe_3
}

# 5. Retina gorus alani icin yaklasik maske
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fov = gray > 25

# 6. Kucuk bolgelerde analiz
tile_size = 64

results = {name: [] for name in variants}

height, width = green.shape

valid_tiles = 0

for y in range(0, height - tile_size + 1, tile_size):
    for x in range(0, width - tile_size + 1, tile_size):

        ys = slice(y, y + tile_size)
        xs = slice(x, x + tile_size)

        region_fov = fov[ys, xs]

        # Retina disindaki bolgeleri ele
        if np.mean(region_fov) < 0.98:
            continue

        region_mask = mask[ys, xs]

        vessel = (region_mask == 255) & region_fov
        background = (region_mask == 0) & region_fov

        # Yeterli damar ve arka plan bulunmali
        if np.sum(vessel) < 40 or np.sum(background) < 100:
            continue

        valid_tiles += 1

        # Her CLAHE secenegi icin ayni bolgeleri kullan
        for name, processed in variants.items():

            region = processed[ys, xs]

            vessel_pixels = region[vessel].astype(np.float32)
            bg_pixels = region[background].astype(np.float32)

            vessel_mean = np.mean(vessel_pixels)
            bg_mean = np.mean(bg_pixels)

            vessel_std = np.std(vessel_pixels)
            bg_std = np.std(bg_pixels)

            contrast = bg_mean - vessel_mean

            denominator = np.sqrt(
                (vessel_std**2 + bg_std**2) / 2
            )

            cnr = contrast / max(float(denominator), 1e-6)

            results[name].append({
                "contrast": float(contrast),
                "cnr": float(cnr)
            })

# 7. Sonuclari yazdir
print("\n--- CLAHE DAMAR KONTRAST ANALIZI ---")
print("Analiz edilen bolge sayisi:", valid_tiles)

if valid_tiles == 0:
    raise RuntimeError("Analiz icin uygun bolge bulunamadi!")

for name, measurements in results.items():

    contrasts = [m["contrast"] for m in measurements]
    cnrs = [m["cnr"] for m in measurements]

    print(f"\n{name}")
    print(f"Medyan damar kontrasti: {median(contrasts):.4f}")
    print(f"Medyan CNR: {median(cnrs):.4f}")

print("\nAnaliz tamamlandi.")
