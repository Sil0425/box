import cv2
import numpy as np

def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Carica immagine
image = cv2.imread("box.jpeg")
if image is None:
    print("❌ ERRORE: Immagine non trovata.")
    exit()
print("✅ Immagine caricata.")

height, width = image.shape[:2]

offset_x = 0
offset_y = 0

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
gray = cv2.equalizeHist(gray)

# Rilevamento cerchi
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=40,
    minRadius=20,
    maxRadius=30
)

#  Rilevamento moneta con filtri + calcolo scala
if circles is not None:
    circles = np.uint16(np.around(circles))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)  # Lab per distanze colore più robuste
    edges = cv2.Canny(gray, 80, 150)

    candidates = []
    for c in circles[0, :]:
        x, y, r = int(c[0]), int(c[1]), int(c[2])

        # salta cerchi troppo vicino ai bordi dell'immagine
        if x - r < 5 or y - r < 5 or x + r > width - 5 or y + r > height - 5:
            continue
        if not (18 < r < 70):
            continue

        # maschera interno del cerchio (escludo il bordo)
        mask_in = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask_in, (x, y), max(1, r - 3), 255, -1)

        # maschera annulus (area di background attorno al cerchio)
        outer = int(min(min(width, height) - 1, r + int(r * 1.4) + 5))
        mask_ann = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask_ann, (x, y), outer, 255, -1)
        cv2.circle(mask_ann, (x, y), r + 4, 0, -1)

        if np.count_nonzero(mask_ann) == 0 or np.count_nonzero(mask_in) == 0:
            continue

        # media colore (Lab) dentro e nell'annulus
        mean_in = np.array(cv2.mean(lab, mask=mask_in)[:3])
        mean_ann = np.array(cv2.mean(lab, mask=mask_ann)[:3])
        color_dist = np.linalg.norm(mean_in - mean_ann)  # distanza colore medio

        # distanza pixel-per-pixel dal colore medio dell'annulus -> fill ratio
        inside_pixels = lab[mask_in == 255][:, :3]
        if inside_pixels.size == 0:
            continue
        dists = np.linalg.norm(inside_pixels - mean_ann, axis=1)
        tol = 12.0            # tolleranza colore
        fill_ratio = np.mean(dists <= tol)  # percentuale di pixel dentro simili al background

        # deviazione dentro il cerchio (moneta -> più varianza)
        std_dev = float(np.std(gray[mask_in == 255]))

        # contorno medio (aiuta se logo ha bordo debole)
        y0, y1 = max(0, y - r), min(height, y + r)
        x0, x1 = max(0, x - r), min(width, x + r)
        perimeter_strength = float(np.mean(edges[y0:y1, x0:x1]))

        candidates.append({
            "x": x, "y": y, "r": r,
            "color_dist": float(color_dist),
            "fill_ratio": float(fill_ratio),
            "std": std_dev,
            "perim": perimeter_strength
        })

    # DECISIONE: preferisco cerchi con fill_ratio basso (cioè NON riempiti come il cartone)
    valid = [c for c in candidates if c["fill_ratio"] < 0.6 and c["color_dist"] > 8]

    # fallback se nessuno passa
    if not valid:
        valid = [c for c in candidates if c["std"] > 14 or c["perim"] > 10]

    if valid:
        # punteggio composito per scegliere il migliore
        def score(c):
            return (1.0 - c["fill_ratio"]) * 2.0 + c["std"] / 30.0 + c["color_dist"] / 30.0 + c["perim"] / 15.0
        best = max(valid, key=score)
        x, y, r = best["x"], best["y"], best["r"]

        # calcolo misure e scala
        # Se stai processando un ritaglio, applica gli offset per ottenere coordinate globali
        center_global = (int(x + offset_x), int(y + offset_y))

        diametro_px = float(r * 2)
        # diametro reale moneta in mm
        scala_px_per_mm = diametro_px / 23.25

        print(f"🟢 Moneta trovata: (x={center_global[0]}, y={center_global[1]}, r={r})")
        print(f"📏 Diametro in px: {diametro_px:.2f}  -> Scala: {scala_px_per_mm:.2f} px/mm")

        # disegno sul frame usando coordinate globali
        cv2.circle(image, center_global, r, (0, 255, 0), 2)
        cv2.circle(image, center_global, 3, (0, 0, 255), -1)


        # SCATOLA
        height, width = image.shape[:2]

        # Conversione HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        #Threshold sul canale Saturation
        _, mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)

        # Operazioni morfologiche
        mask = cv2.dilate(mask,kernel) #okk
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # top
        #mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        # Trova contorni
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Filtro i contorni piccoli
        min_area = 1000 # regolo in base alla dimensione della scatola
        contours = [c for c in contours if cv2.contourArea(c) > min_area] #Prendo contorno principale
        c = max(contours, key=cv2.contourArea)

        #Prendo tutti i punti del contorno
        pts = c.reshape(-1,2)

        #Calcolo l’asse principale con PCA se la scatola non è perfettamente allineata
        pts_centered = pts - pts.mean(axis=0)
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]
        perp_axis = eigvecs[:, np.argmin(eigvals)]

        # Proiezione di tutti i punti sull’asse principale e su quello perpendicolare
        proj_main = pts_centered @ main_axis
        proj_perp = pts_centered @ perp_axis

        width_px = proj_main.max() - proj_main.min()
        height_px = proj_perp.max() - proj_perp.min()







        # Conversione in mm usando scala
        width_mm = width_px / scala_px_per_mm
        height_mm = height_px / scala_px_per_mm

        print(f"Larghezza reale: {width_mm:.2f} mm")
        print(f"Altezza reale: {height_mm:.2f} mm")





        image_box = image.copy()
        cv2.drawContours(image_box, [c], -1, (0,255,0), 2)
        cv2.imshow("Contorno", resize_for_display(image_box))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
        print("⚠️ Nessuna moneta valida trovata ")
