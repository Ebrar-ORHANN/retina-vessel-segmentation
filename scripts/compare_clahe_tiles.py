from pathlib import Path
import cv2
import numpy as np

# Proje klasoru
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Goruntu yolu
image_path = PROJECT_ROOT / "data/raw/sample/images/1_A.png"

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")

# Yesil kanal
green = image[:, :, 1]

# Sabit clip limit
clip_limit = 2.0

# Denenecek tile boyutlari
tile_sizes = [(4, 4), (8, 8), (16, 16)]

results = [("GREEN CHANNEL", green)]

for tile in tile_sizes:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile
    )

    enhanced = clahe.apply(green)
    results.append((f"TILE {tile[0]}x{tile[1]}", enhanced))

    difference = np.mean(
        np.abs(
            enhanced.astype(np.int16) - green.astype(np.int16)
        )
    )

    print(
        f"tileGridSize={tile} | "
        f"Ortalama={enhanced.mean():.2f} | "
        f"Std={enhanced.std():.2f} | "
        f"Ortalama fark={difference:.2f}"
    )

# Panelleri hazirla
panels = []

for title, result in results:
    small = cv2.resize(
        result,
        (512, 512),
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

# 2x2 yerlesim
top = cv2.hconcat(panels[:2])
bottom = cv2.hconcat(panels[2:])
comparison = cv2.vconcat([top, bottom])

# Kaydet
output_dir = PROJECT_ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "1_A_clahe_tiles.png"

if not cv2.imwrite(str(output_path), comparison):
    raise OSError("Gorsel kaydedilemedi!")

print("\nKarsilastirma tamamlandi.")
print("Kaydedilen dosya:", output_path)