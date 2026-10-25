
from pathlib import Path
import cv2

# 1. Projenin ana klasorunu bul
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Goruntu ve maske yollarini belirle
image_path = PROJECT_ROOT / "data/raw/sample/images/1_A.png"
mask_path = PROJECT_ROOT / "data/raw/sample/masks/1_A.png"

# 3. Goruntuleri oku
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")

if mask is None:
    raise FileNotFoundError(f"Maske bulunamadi: {mask_path}")

if image.shape[:2] != mask.shape:
    raise ValueError("Goruntu ve maske boyutlari uyusmuyor!")

# 4. OpenCV BGR siralamasini kullanir
blue, green, red = cv2.split(image)

# 5. Kanallari karsilastirma icin 3 kanalli hale getir
red_view = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
green_view = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
blue_view = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 6. Karsilastirilacak goruntuler
panels = [
    ("ORIGINAL", image),
    ("RED", red_view),
    ("GREEN", green_view),
    ("BLUE", blue_view),
    ("GROUND TRUTH", mask_view),
]

# 7. Goruntuleri yan yana yerlestir
panel_images = []

for title, panel in panels:

    # Maskede sinif degerlerini koru
    interpolation = (
        cv2.INTER_NEAREST
        if title == "GROUND TRUTH"
        else cv2.INTER_AREA
    )

    resized = cv2.resize(
        panel, (512, 512),
        interpolation=interpolation
    )

    # Baslik icin siyah alan ekle
    labeled = cv2.copyMakeBorder(
        resized, 50, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    cv2.putText(
        labeled,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    panel_images.append(labeled)

comparison = cv2.hconcat(panel_images)

# 8. Sonucu kaydet
output_dir = PROJECT_ROOT / "outputs/figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "1_A_channel_comparison.png"

if not cv2.imwrite(str(output_path), comparison):
    raise OSError("Karsilastirma gorseli kaydedilemedi!")

# 9. Bilgileri yazdir
print("Orijinal goruntu:", image.shape)
print("Kirmizi kanal:", red.shape)
print("Yesil kanal:", green.shape)
print("Mavi kanal:", blue.shape)
print("Maske:", mask.shape)
print("Karsilastirma kaydedildi:", output_path)
