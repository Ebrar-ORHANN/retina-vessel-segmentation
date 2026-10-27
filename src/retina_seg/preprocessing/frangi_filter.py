
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi


# Projenin ana klasoru
ROOT = Path(__file__).resolve().parents[3]


def apply_frangi(gray_image, sigmas=(1, 2, 3, 4)):
    """
    Tek kanalli uint8 goruntuye Frangi uygular.

    Cikti:
        float32 damar benzerligi haritasi
    """

    if gray_image.ndim != 2:
        raise ValueError("Goruntu tek kanalli olmali!")

    if gray_image.dtype != np.uint8:
        raise ValueError("Goruntu uint8 tipinde olmali!")

    # Piksel degerlerini 0-1 araligina getir
    normalized = gray_image.astype(np.float32) / 255.0

    # Koyu damar benzeri yapilari belirginlestir
    vesselness = frangi(
        normalized,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=True
    )

    vesselness = vesselness.astype(np.float32)

    if not np.all(np.isfinite(vesselness)):
        raise ValueError("Frangi gecersiz piksel degeri uretti!")

    return vesselness


def main():

    # 1. Goruntu ve maske yollarini olustur
    image_path = ROOT / "data/raw/sample/images/1_A.png"
    mask_path = ROOT / "data/raw/sample/masks/1_A.png"

    # 2. Dosyalari oku
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise FileNotFoundError("Goruntu veya maske bulunamadi!")

    if image.shape[:2] != mask.shape:
        raise ValueError("Goruntu ve maske boyutlari uyusmuyor!")

    # 3. Yesil kanali al
    green = image[:, :, 1]

    # 4. CLAHE uygula
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green)

    # 5. Merkezden 512x512 kesit al
    h, w = green.shape
    size = 512

    y = (h - size) // 2
    x = (w - size) // 2

    original_crop = image[y:y+size, x:x+size]
    clahe_crop = enhanced[y:y+size, x:x+size]
    mask_crop = mask[y:y+size, x:x+size]

    # 6. Frangi uygula
    vesselness = apply_frangi(clahe_crop)

    # 7. Frangi sonucunu yalnizca gorsellestirmek icin olcekle
    frangi_view = cv2.normalize(
        vesselness,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 8. Panelleri hazirla
    panels = [
        ("ORIGINAL", original_crop),
        ("CLAHE", cv2.cvtColor(
            clahe_crop, cv2.COLOR_GRAY2BGR
        )),
        ("FRANGI", cv2.cvtColor(
            frangi_view, cv2.COLOR_GRAY2BGR
        )),
        ("GROUND TRUTH", cv2.cvtColor(
            mask_crop, cv2.COLOR_GRAY2BGR
        ))
    ]

    labeled_panels = []

    for title, panel in panels:

        labeled = cv2.copyMakeBorder(
            panel, 45, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        cv2.putText(
            labeled,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        labeled_panels.append(labeled)

    # 9. Sonuclari birlestir
    comparison = cv2.hconcat(labeled_panels)

    # 10. Ciktiyi kaydet
    output_dir = ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "1_A_frangi_comparison.png"

    if not cv2.imwrite(str(output_path), comparison):
        raise OSError("Gorsel kaydedilemedi!")

    # 11. Sayisal bilgileri yazdir
    print("Kesit boyutu:", clahe_crop.shape)
    print("Frangi veri tipi:", vesselness.dtype)
    print("Minimum:", float(vesselness.min()))
    print("Maksimum:", float(vesselness.max()))
    print("Ortalama:", float(vesselness.mean()))
    print("Gorsel kaydedildi:", output_path)


if __name__ == "__main__":
    main()
