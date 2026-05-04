#!/usr/bin/env python3
"""
stereo_calibration.py
Calibrazione stereo per due camere IMX219 su Jetson Orin Nano (CSI-0 / CSI-1).

Utilizzo:
  1. Stampa un pattern a scacchiera 9x6 su carta rigida.
  2. Misura il lato di un quadrato in mm (es. 25mm) e aggiorna SQUARE_SIZE_MM.
  3. Esegui: python3 stereo_calibration.py
  4. Premi SPAZIO per catturare una coppia, ESC per terminare e calcolare.
     Cattura almeno 15-20 coppie da angolazioni diverse.
  5. I risultati vengono salvati in stereo_calib.npz

Dipendenze: opencv-python (con GStreamer e CUDA), numpy
"""

import cv2
import numpy as np
import os
import time

# ─── Parametri configurabili ──────────────────────────────────────────────────
CHECKERBOARD     = (9, 6)          # colonne-1, righe-1 del pattern interno
SQUARE_SIZE_MM   = 25.0            # dimensione reale di un quadrato in mm
CAPTURE_WIDTH    = 1640
CAPTURE_HEIGHT   = 1232
FPS              = 15
SAVE_DIR         = "calib_images"  # cartella dove salvare le coppie catturate
CALIB_FILE       = "stereo_calib.npz"
MIN_CAPTURES     = 15              # coppie minime prima di calcolare
# ─────────────────────────────────────────────────────────────────────────────


def gstreamer_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_cameras():
    cap_l = cv2.VideoCapture(
        gstreamer_pipeline(0, CAPTURE_WIDTH, CAPTURE_HEIGHT, FPS),
        cv2.CAP_GSTREAMER
    )
    cap_r = cv2.VideoCapture(
        gstreamer_pipeline(1, CAPTURE_WIDTH, CAPTURE_HEIGHT, FPS),
        cv2.CAP_GSTREAMER
    )
    if not cap_l.isOpened():
        raise RuntimeError("Impossibile aprire camera sinistra (sensor-id=0)")
    if not cap_r.isOpened():
        raise RuntimeError("Impossibile aprire camera destra (sensor-id=1)")
    print(f"[OK] Entrambe le camere aperte  ({CAPTURE_WIDTH}x{CAPTURE_HEIGHT} @ {FPS}fps)")
    return cap_l, cap_r


def prepare_object_points():
    """Punti 3D del pattern nel sistema di riferimento del mondo (z=0)."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def find_corners(img, pattern):
    """Trova e raffina i corner della scacchiera."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners


def capture_phase(cap_l, cap_r, objp):
    """Fase interattiva di acquisizione coppie immagini."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    obj_points  = []
    pts_l, pts_r = [], []
    img_size    = None
    n           = 0

    print("\n── Istruzioni ──────────────────────────────────────────────────")
    print("  SPAZIO  → cattura coppia se entrambe trovano la scacchiera")
    print("  ESC     → termina acquisizione e calcola calibrazione")
    print("  Almeno 15 coppie, da angolazioni e distanze diverse")
    print("────────────────────────────────────────────────────────────────\n")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            print("[WARN] Frame non ricevuto, riprovo...")
            time.sleep(0.05)
            continue

        if img_size is None:
            img_size = (frame_l.shape[1], frame_l.shape[0])

        # Cerca pattern in entrambe (su copia per display)
        found_l, corners_l = find_corners(frame_l, CHECKERBOARD)
        found_r, corners_r = find_corners(frame_r, CHECKERBOARD)

        disp_l = frame_l.copy()
        disp_r = frame_r.copy()
        cv2.drawChessboardCorners(disp_l, CHECKERBOARD, corners_l, found_l)
        cv2.drawChessboardCorners(disp_r, CHECKERBOARD, corners_r, found_r)

        # Preview affiancato (ridotto per display)
        scale  = 0.4
        dw, dh = int(img_size[0] * scale), int(img_size[1] * scale)
        small_l = cv2.resize(disp_l, (dw, dh))
        small_r = cv2.resize(disp_r, (dw, dh))

        status = f"Catture: {n}/{MIN_CAPTURES}  |  "
        status += "L:OK " if found_l else "L:-- "
        status += "R:OK" if found_r else "R:--"
        combined = np.hstack([small_l, small_r])
        cv2.putText(combined, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if (found_l and found_r) else (0, 0, 255), 2)
        cv2.imshow("Calibrazione Stereo  [SPAZIO=cattura  ESC=calcola]", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPAZIO
            if found_l and found_r:
                obj_points.append(objp)
                pts_l.append(corners_l)
                pts_r.append(corners_r)
                n += 1
                # Salva immagini grezze
                cv2.imwrite(os.path.join(SAVE_DIR, f"left_{n:03d}.png"),  frame_l)
                cv2.imwrite(os.path.join(SAVE_DIR, f"right_{n:03d}.png"), frame_r)
                print(f"[{n:02d}] Coppia salvata")
            else:
                missing = []
                if not found_l: missing.append("sinistra")
                if not found_r: missing.append("destra")
                print(f"[SKIP] Scacchiera non trovata in: {', '.join(missing)}")

    cv2.destroyAllWindows()
    return obj_points, pts_l, pts_r, img_size


def calibrate_stereo(obj_points, pts_l, pts_r, img_size):
    """Esegue calibrazione mono + stereo."""
    if len(obj_points) < MIN_CAPTURES:
        raise RuntimeError(
            f"Coppie insufficienti: {len(obj_points)}/{MIN_CAPTURES}. "
            "Catturare più immagini."
        )

    print(f"\n[→] Calibrazione mono camera sinistra  ({len(obj_points)} coppie)...")
    rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        obj_points, pts_l, img_size, None, None)
    print(f"    RMS sinistra: {rms_l:.4f} px")

    print("[→] Calibrazione mono camera destra...")
    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        obj_points, pts_r, img_size, None, None)
    print(f"    RMS destra:   {rms_r:.4f} px")

    print("[→] Calibrazione stereo...")
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC  # usa le K già calcolate
    )
    rms_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, pts_l, pts_r,
        K_l, D_l, K_r, D_r,
        img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=stereo_flags
    )
    print(f"    RMS stereo:   {rms_s:.4f} px")

    baseline_m = np.linalg.norm(T) / 1000.0  # T è in mm → converti in metri
    print(f"    Baseline:     {baseline_m*100:.2f} cm")

    print("[→] Calcolo mappe di rettifica...")
    R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    return dict(
        K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
        R=R, T=T, E=E, F=F,
        R_l=R_l, R_r=R_r, P_l=P_l, P_r=P_r, Q=Q,
        img_size=np.array(img_size),
        baseline_m=np.array([baseline_m]),
        rms_stereo=np.array([rms_s])
    )


def verify_rectification(cap_l, cap_r, calib: dict):
    """Mostra anteprima rettificata con linee epipolari per verifica visiva."""
    img_size = tuple(calib["img_size"])

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib["K_l"], calib["D_l"], calib["R_l"], calib["P_l"], img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib["K_r"], calib["D_r"], calib["R_r"], calib["P_r"], img_size, cv2.CV_32FC1)

    print("\n[→] Anteprima rettificazione (premi un tasto per chiudere)...")
    for _ in range(60):
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if ret_l and ret_r:
            break

    rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

    scale   = 0.4
    dw, dh  = int(img_size[0] * scale), int(img_size[1] * scale)
    canvas  = np.hstack([
        cv2.resize(rect_l, (dw, dh)),
        cv2.resize(rect_r, (dw, dh))
    ])
    # Linee epipolari orizzontali (devono essere allineate)
    for y in range(0, dh, 40):
        cv2.line(canvas, (0, y), (dw * 2, y), (0, 255, 0), 1)

    cv2.imshow("Verifica rettificazione - le linee verdi devono allinearsi", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_k_txt(calib: dict, path: str = "K.txt"):
    """Salva K.txt nel formato richiesto da Fast-FoundationStereo."""
    K = calib["P_l"][:3, :3]  # usa P_l per la camera rettificata
    baseline = float(calib["baseline_m"][0])
    with open(path, "w") as f:
        f.write(f"{K[0,0]:.6f} {K[0,1]:.6f} {K[0,2]:.6f} ")
        f.write(f"{K[1,0]:.6f} {K[1,1]:.6f} {K[1,2]:.6f} ")
        f.write(f"{K[2,0]:.6f} {K[2,1]:.6f} {K[2,2]:.6f}\n")
        f.write(f"{baseline:.6f}\n")
    print(f"[OK] K.txt salvato in '{path}'")


def main():
    print("=" * 60)
    print("  Calibrazione Stereo IMX219 — Jetson Orin Nano")
    print("=" * 60)

    objp    = prepare_object_points()
    cap_l, cap_r = open_cameras()

    try:
        obj_points, pts_l, pts_r, img_size = capture_phase(cap_l, cap_r, objp)

        if len(obj_points) == 0:
            print("[ABORT] Nessuna coppia catturata.")
            return

        calib = calibrate_stereo(obj_points, pts_l, pts_r, img_size)

        np.savez(CALIB_FILE, **calib)
        print(f"\n[OK] Calibrazione salvata in '{CALIB_FILE}'")

        save_k_txt(calib)
        verify_rectification(cap_l, cap_r, calib)

        print("\n─── Riepilogo ───────────────────────────────────────────")
        print(f"  Coppie usate:  {len(obj_points)}")
        print(f"  RMS stereo:    {float(calib['rms_stereo']):.4f} px  (buono < 1.0)")
        print(f"  Baseline:      {float(calib['baseline_m'])*100:.2f} cm")
        print(f"  File:          {CALIB_FILE}, K.txt")
        print("─────────────────────────────────────────────────────────")

    finally:
        cap_l.release()
        cap_r.release()


if __name__ == "__main__":
    main()
