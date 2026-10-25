from pathlib import Path
import cv2
import numpy as np

# Proje klasoru
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Retina goruntusu
image_path = PROJECT_ROOT / "data/raw/sample/images/1_A.png"

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")

# Yesil kanal
green = image[:, :, 1]

# Deneyecegimiz CLAHE parametreleri
clip_limits = [1.0, 2.0, 3.0, 4.0]

# Ilk panel: CLAHE uygulanmamis yesil kanal
results = [("GREEN CHANNEL", green)]

# CLAHE deneyleri
for clip in clip_limits:
    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green)
    results.append((f"CLAHE {clip}", enhanced))

    difference = np.mean(
        np.abs(
            enhanced.astype(np.int16) - green.astype(np.int16)
        )
    )

    print(
        f"clipLimit={clip} | "
        f"Ortalama={enhanced.mean():.2f} | "
        f"Std={enhanced.std():.2f} | "
        f"Ortalama fark={difference:.2f}"
    )

# Karsilastirma panelleri
panels = []

for title, result in results:
    small = cv2.resize(
        result,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    panel = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    panel = cv2.copyMakeBorder(
        panel, 45, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    cv2.putText(
        panel,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    panels.append(panel)

# 2 satirli yerlesim:
# 1. satir: Green, CLAHE 1.0, CLAHE 2.0
# 2. satir: CLAHE 3.0, CLAHE 4.0, bos panel
blank_panel = np.zeros_like(panels[0])

top_row = cv2.hconcat(panels[:3])
bottom_row = cv2.hconcat([panels[3], panels[4], blank_panel])

comparison = cv2.vconcat([top_row, bottom_row])

# Sonucu kaydet
output_dir = PROJECT_ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "1_A_clahe_parameters_extended.png"

if not cv2.imwrite(str(output_path), comparison):
    raise OSError("Karsilastirma gorseli kaydedilemedi!")

print("\nKarsilastirma tamamlandi.")
print("Kaydedilen dosya:", output_path)