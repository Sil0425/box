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

# --- Preprocessing scatola ---
gray_box = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        # grigio
gray_box = cv2.GaussianBlur(gray_box, (5,5), 0)
gray_box = cv2.bilateralFilter(gray_box, 9, 75, 75)
gray_box = cv2.equalizeHist(gray_box)

# soglia adattiva o normale (non invertita)
_, thresh = cv2.threshold(gray_box, 160, 255, cv2.THRESH_BINARY)

# chiusura per unire i bordi della scatola
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# trova solo il contorno più grande (la scatola)
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    biggest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(closing)
    cv2.drawContours(mask, [biggest], -1, 255, -1)

cv2.imshow("Preprocessing Scatola", closing)

mask = np.zeros_like(closing)
if contours:
    biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
closing = mask





# Ridimensionamento per mostrare sullo schermo
height_box, width_box = closing.shape[:2]
max_display_height = 800
max_display_width  = 1200
scale = min(max_display_width / width_box, max_display_height / height_box)
new_width = int(width_box * scale)
new_height = int(height_box * scale)
closing_resized = cv2.resize(closing, (new_width, new_height))

cv2.imshow("Preprocessing Scatola", closing_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()






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