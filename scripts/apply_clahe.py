
from pathlib import Path
import cv2
import numpy as np

# 1. Proje klasorunu belirle
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Retina goruntusunun yolu
image_path = (
    PROJECT_ROOT
    / "data/raw/sample/images/1_A.png"
)

# 3. Goruntuyu oku
image = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR
)

if image is None:
    raise FileNotFoundError(
        f"Goruntu bulunamadi: {image_path}"
    )

# 4. Yesil kanali cikar
green = image[:, :, 1]

# 5. CLAHE algoritmasini olustur
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# 6. CLAHE uygula
green_clahe = clahe.apply(green)

# 7. Karsilastirma icin goruntuleri kucult
original_small = cv2.resize(
    image, (512, 512),
    interpolation=cv2.INTER_AREA
)

green_small = cv2.resize(
    green, (512, 512),
    interpolation=cv2.INTER_AREA
)

clahe_small = cv2.resize(
    green_clahe, (512, 512),
    interpolation=cv2.INTER_AREA
)

# 8. Tek kanalli goruntuleri BGR'ye cevir
green_view = cv2.cvtColor(
    green_small, cv2.COLOR_GRAY2BGR
)

clahe_view = cv2.cvtColor(
    clahe_small, cv2.COLOR_GRAY2BGR
)

# 9. Goruntuleri yan yana birlestir
comparison = cv2.hconcat([
    original_small,
    green_view,
    clahe_view
])

# 10. Cikti klasorunu olustur
output_dir = PROJECT_ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    output_dir / "1_A_clahe_comparison.png"
)

# 11. Sonucu kaydet
success = cv2.imwrite(
    str(output_path),
    comparison
)

if not success:
    raise OSError("Gorsel kaydedilemedi!")

# 12. Sonuclari yazdir
print("Orijinal:", image.shape)
print("Yesil kanal:", green.shape)
print("CLAHE sonucu:", green_clahe.shape)

print("Yesil ortalama:", round(float(np.mean(green)), 2))
print("CLAHE ortalama:", round(float(np.mean(green_clahe)), 2))

print("Gorsel kaydedildi:", output_path)
