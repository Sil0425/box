import cv2
import numpy as np

CONFIG = {
    "coin_real_diameter_mm": 23.25,
    "hough_param1": 100,
    "hough_param2": 40,
    "hough_min_radius": 20,
    "hough_max_radius": 30,
    "fill_ratio_thresh": 0.6,
    "color_dist_thresh": 8,
    "std_thresh": 14,
    "perim_thresh": 10,
    "mask_sat_thresh": 30,
    "contour_min_area": 1000,
    "color_tol": 12.0
}


def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def detect_coin(img, config):
    """Rileva la moneta e calcola scala px/mm"""
    height, width = img.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)

    # Rilevamento cerchi
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=config["hough_param1"],
        param2=config["hough_param2"],
        minRadius=config["hough_min_radius"],
        maxRadius=config["hough_max_radius"]
    )

    if circles is None:
        print("⚠️ Nessuna moneta rilevata")
        return None, None, None

    circles = np.uint16(np.around(circles))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)  # Lab per distanze colore più robuste
    edges = cv2.Canny(gray, 80, 150)

    candidates = []
    for c in circles[0, :]:
        x, y, r = int(c[0]), int(c[1]), int(c[2])

        # salta cerchi troppo vicino ai bordi dell'immagine
        if x - r < 5 or y - r < 5 or x + r > width - 5 or y + r > height - 5:
            continue
        if not (config["hough_min_radius"] - 2 < r < config["hough_max_radius"] + 40):
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
        tol = config["color_tol"]  # tolleranza colore
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
    valid = [c for c in candidates if c["fill_ratio"] < config["fill_ratio_thresh"] and c["color_dist"] > config["color_dist_thresh"]]

    # fallback se nessuno passa
    if not valid:
        valid = [c for c in candidates if c["std"] > config["std_thresh"] or c["perim"] > config["perim_thresh"]]

    if not valid:
        print("⚠️ Nessuna moneta valida trovata dopo i filtri.")
        return None, None, None

    # punteggio composito per scegliere il migliore
    def score(c):
        return (1.0 - c["fill_ratio"]) * 2.0 + c["std"] / 30.0 + c["color_dist"] / 30.0 + c["perim"] / 15.0

    best = max(valid, key=score)
    x, y, r = best["x"], best["y"], best["r"]

    center_global = (x, y)
    diametro_px = float(r * 2)
    scala_px_per_mm = diametro_px / config["coin_real_diameter_mm"]

    print(f"🟢 Moneta trovata: (x={center_global[0]}, y={center_global[1]}, r={r})")
    print(f"📏 Diametro in px: {diametro_px:.2f}  -> Scala: {scala_px_per_mm:.2f} px/mm")

    # disegno sul frame usando coordinate globali
    cv2.circle(img, center_global, r, (0, 255, 0), 2)
    cv2.circle(img, center_global, 3, (0, 0, 255), -1)

    return scala_px_per_mm, center_global, r


def detect_box(img, scala_px_per_mm, config):
    """Rileva la scatola e ritorna dimensioni reali in mm con debug step-by-step"""
    height, width = img.shape[:2]

    # Conversione HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    cv2.imshow("Debug - HSV Saturation", resize_for_display(sat))
    cv2.waitKey(0)

    # Threshold sul canale Saturation
    _, mask = cv2.threshold(sat, config["mask_sat_thresh"], 255, cv2.THRESH_BINARY)
    cv2.imshow("Debug - Threshold", resize_for_display(mask))
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)

    # DILATE
    mask = cv2.dilate(mask, kernel)
    cv2.imshow("Debug - Dilate", resize_for_display(mask))
    cv2.waitKey(0)

    # MORPH CLOSE
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("Debug - Morph Close", resize_for_display(mask))
    cv2.waitKey(0)

    # MORPH OPEN
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("Debug - Morph Open", resize_for_display(mask))
    cv2.waitKey(0)

    # Trova contorni
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > config["contour_min_area"]]

    if not contours:
        print("⚠️ Nessun contorno valido trovato nella maschera.")
        cv2.imshow("Debug - Mask Finale", resize_for_display(mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None, None

    c = max(contours, key=cv2.contourArea)

    # Contorno finale disegnato
    img_box = img.copy()
    cv2.drawContours(img_box, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Debug - Contorno", resize_for_display(img_box))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # PCA per dimensioni reali
    pts = c.reshape(-1, 2)
    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    perp_axis = eigvecs[:, np.argmin(eigvals)]

    proj_main = pts_centered @ main_axis
    proj_perp = pts_centered @ perp_axis
    width_px = proj_main.max() - proj_main.min()
    height_px = proj_perp.max() - proj_perp.min()
    width_mm = width_px / scala_px_per_mm
    height_mm = height_px / scala_px_per_mm

    print(f"Larghezza reale: {width_mm:.2f} mm")
    print(f"Altezza reale: {height_mm:.2f} mm")

    return width_mm, height_mm, c



if __name__ == "__main__":
    image = cv2.imread("box.jpeg")
    if image is None:
        print("❌ ERRORE: Immagine non trovata.")
        exit()

    scala, center, raggio = detect_coin(image, CONFIG)
    if scala is not None:
        detect_box(image, scala, CONFIG)