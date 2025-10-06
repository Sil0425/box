import cv2
import numpy as np

# --- Carica immagine ---
image = cv2.imread("img.jpg")
if image is None:
    print("❌ ERRORE: Immagine non trovata.")
    exit()
print("✅ Immagine caricata.")

height, width = image.shape[:2]

# --- Rilevamento moneta (come prima) ---
offset_x = width // 2
offset_y = 0
crop_height = height // 3
top_right = image[offset_y:offset_y + crop_height, offset_x:width]

gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
gray = cv2.equalizeHist(gray)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=40,
    minRadius=20,
    maxRadius=50
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :1]:
        center_local = (i[0], i[1])
        radius = i[2]
        center_global = (center_local[0] + offset_x, center_local[1])

        print(f"🟢 Centro globale moneta: {center_global}, Raggio: {radius}px")
        diametro_px = radius * 2
        scala_px_per_mm = diametro_px / 23.25
        print(f"📏 Diametro moneta in px: {diametro_px:.2f} -> Scala: {scala_px_per_mm:.2f} px/mm")

        cv2.circle(image, center_global, radius, (0, 255, 0), 2)
        cv2.circle(image, center_global, 3, (0, 0, 255), -1)

else:
    print("⚠️ Nessun cerchio trovato.")
    exit()

# --- Mostra immagine mantenendo proporzioni ---
max_display_height = 800  # massimo in pixel sullo schermo
max_display_width  = 1200

scale = min(max_display_width / width, max_display_height / height)
new_width = int(width * scale)
new_height = int(height * scale)
resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow("Risultato finale", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
