
import sys
import numpy as np
import cv2

# Python ve kutuphane surumlerini kontrol et
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)

# 100x100 boyutunda renkli goruntu olustur
image = np.zeros((100, 100, 3), dtype=np.uint8)

# Renkli goruntuyu gri tona donustur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sonuclari kontrol et
assert image.shape == (100, 100, 3)
assert gray.shape == (100, 100)

print("Renkli goruntu:", image.shape)
print("Gri goruntu:", gray.shape)
print("Tum kontroller basarili!")
