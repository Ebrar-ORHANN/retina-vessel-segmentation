
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/raw/sample/images"
MASK_DIR = ROOT / "data/raw/sample/masks"

OUTPUT_DIR = ROOT / "outputs/metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 64

# Karsilastirilacak CLAHE ayarlari
CLIP_LIMITS = [2.0, 3.0]

image_paths = sorted(IMAGE_DIR.glob("*.png"))

if not image_paths:
    raise FileNotFoundError("PNG goruntu bulunamadi!")

# Tum sonuclari saklayacagiz
rows = []

for image_path in image_paths:

    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Maske bulunamadi: {image_path.name}"
        )

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise ValueError(
            f"Goruntu veya maske okunamadi: {image_path.name}"
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Boyut uyusmazligi: {image_path.name}"
        )

    green = image[:, :, 1]

    # Onceki deneyle ayni yaklasik FOV kontrolu
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )
    fov = gray > 25

    # CLAHE varyantlarini olustur
    variants = {"Green": green}

    for clip in CLIP_LIMITS:
        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(8, 8)
        )

        variants[f"CLAHE {clip}"] = clahe.apply(green)

    measurements = {
        name: {"contrast": [], "cnr": []}
        for name in variants
    }

    height, width = green.shape
    valid_tiles = 0

    # Goruntuyu 64x64 bolgelere ayir
    for y in range(0, height - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, width - TILE_SIZE + 1, TILE_SIZE):

            ys = slice(y, y + TILE_SIZE)
            xs = slice(x, x + TILE_SIZE)

            region_fov = fov[ys, xs]

            if np.mean(region_fov) < 0.98:
                continue

            region_mask = mask[ys, xs]

            vessel = (region_mask == 255) & region_fov
            background = (region_mask == 0) & region_fov

            if np.sum(vessel) < 40:
                continue

            if np.sum(background) < 100:
                continue

            valid_tiles += 1

            for name, processed in variants.items():

                region = processed[ys, xs]

                vessel_pixels = region[vessel].astype(np.float32)
                bg_pixels = region[background].astype(np.float32)

                contrast = (
                    np.mean(bg_pixels)
                    - np.mean(vessel_pixels)
                )

                denominator = np.sqrt(
                    (
                        np.var(bg_pixels)
                        + np.var(vessel_pixels)
                    ) / 2
                )

                cnr = contrast / max(float(denominator), 1e-6)

                measurements[name]["contrast"].append(
                    float(contrast)
                )

                measurements[name]["cnr"].append(
                    float(cnr)
                )

    if valid_tiles == 0:
        raise RuntimeError(
            f"Uygun bolge bulunamadi: {image_path.name}"
        )

    print(f"\n--- {image_path.name} ---")
    print("Analiz edilen bolge:", valid_tiles)

    for name, values in measurements.items():

        median_contrast = float(
            np.median(values["contrast"])
        )

        median_cnr = float(
            np.median(values["cnr"])
        )

        rows.append({
            "image": image_path.name,
            "method": name,
            "valid_tiles": valid_tiles,
            "median_contrast": round(median_contrast, 4),
            "median_cnr": round(median_cnr, 4)
        })

        print(
            f"{name:<12} | "
            f"Kontrast: {median_contrast:.4f} | "
            f"CNR: {median_cnr:.4f}"
        )

# Sonuclari UTF-8 CSV olarak kaydet
output_path = OUTPUT_DIR / "clahe_contrast_results.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "method",
            "valid_tiles",
            "median_contrast",
            "median_cnr"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n--- GORUNTULER ARASI OZET ---")

for method in ["Green", "CLAHE 2.0", "CLAHE 3.0"]:

    method_rows = [
        r for r in rows
        if r["method"] == method
    ]

    cnrs = [r["median_cnr"] for r in method_rows]
    contrasts = [r["median_contrast"] for r in method_rows]

    print(
        f"{method:<12} | "
        f"Kontrast medyani: {np.median(contrasts):.4f} | "
        f"CNR medyani: {np.median(cnrs):.4f}"
    )

print("\nAnaliz tamamlandi.")
print("CSV:", output_path)
