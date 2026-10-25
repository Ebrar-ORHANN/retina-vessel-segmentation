
from pathlib import Path
import cv2
import numpy as np

# Projenin ana klasorunu bul
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Goruntu ve maske klasorleri
IMAGE_DIR = PROJECT_ROOT / "data/raw/sample/images"
MASK_DIR = PROJECT_ROOT / "data/raw/sample/masks"

# PNG dosyalarini listele
images = sorted(IMAGE_DIR.glob("*.png"))
masks = sorted(MASK_DIR.glob("*.png"))

print("Goruntu sayisi:", len(images))
print("Maske sayisi:", len(masks))

# Dosya isimlerini kontrol et
image_names = {p.name for p in images}
mask_names = {p.name for p in masks}

if not images:
    raise SystemExit("HATA: Goruntu bulunamadi!")

if image_names != mask_names:
    print("Maskesi olmayan goruntuler:", image_names - mask_names)
    print("Goruntusu olmayan maskeler:", mask_names - image_names)
    raise SystemExit("HATA: Dosyalar eslesmiyor!")

print("\nDosyalar eslesiyor!\n")

# Her goruntuyu incele
for image_path in images:

    mask_path = MASK_DIR / image_path.name

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        raise SystemExit(f"HATA: Dosya okunamadi: {image_path.name}")

    print("-" * 45)
    print("Dosya:", image_path.name)
    print("Goruntu boyutu:", image.shape)
    print("Maske boyutu:", mask.shape)
    print("Goruntu veri tipi:", image.dtype)
    print("Maske veri tipi:", mask.dtype)

    if image.shape[:2] != mask.shape[:2]:
        raise SystemExit("HATA: Goruntu ve maske boyutlari farkli!")

    unique_values = np.unique(mask)

    if len(unique_values) <= 10:
        print("Maske piksel degerleri:", unique_values)
    else:
        print("Maske min/max:", mask.min(), mask.max())
        print("Farkli maske degeri sayisi:", len(unique_values))

print("\nTum goruntu-mask ciftleri incelendi!")



print("\n--- MASKE KANAL KONTROLU ---")

for mask_path in masks:

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if mask.ndim == 3 and mask.shape[2] == 3:

        blue = mask[:, :, 0]
        green = mask[:, :, 1]
        red = mask[:, :, 2]

        channels_equal = (
            np.array_equal(blue, green)
            and np.array_equal(green, red)
        )

        print(mask_path.name, "Kanallar ayni mi:", channels_equal)

        if not channels_equal:
            raise ValueError(
                f"Maske kanallari farkli: {mask_path.name}"
            )

    elif mask.ndim == 2:
        print(mask_path.name, "Zaten tek kanalli")

    else:
        raise ValueError(
            f"Beklenmeyen maske boyutu: {mask.shape}"
        )
