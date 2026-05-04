#!/usr/bin/env python3
"""
stereo_pipeline.py
Pipeline completa: acquisizione IMX219 → rettifica → Fast-FoundationStereo → depth map.

Prerequisiti:
  1. stereo_calib.npz e K.txt generati da stereo_calibration.py
  2. Fast-FoundationStereo TRT engine già convertito:
       https://github.com/NVlabs/Fast-FoundationStereo
       python scripts/make_single_onnx.py ...
       trtexec --onnx=... --saveEngine=fast_fs.engine --fp16
  3. pip install numpy opencv-python tensorrt pycuda

Utilizzo:
  python3 stereo_pipeline.py
  python3 stereo_pipeline.py --save_depth   # salva ogni depth map come PNG
  python3 stereo_pipeline.py --save_pc      # salva point cloud .ply ogni N frame
"""

import argparse
import time
import numpy as np
import cv2
import os
import struct

# ─── Parametri configurabili ──────────────────────────────────────────────────
CALIB_FILE      = "stereo_calib.npz"
ENGINE_PATH     = "fast_fs.engine"          # TRT engine di Fast-FoundationStereo
INFER_WIDTH     = 640                        # dimensione input al modello
INFER_HEIGHT    = 480
CAPTURE_WIDTH   = 1640                       # risoluzione acquisizione CSI
CAPTURE_HEIGHT  = 1232
FPS             = 15
COLORMAP        = cv2.COLORMAP_MAGMA         # colormap per visualizzazione depth
SAVE_EVERY_N    = 30                         # salva point cloud ogni N frame
MAX_DEPTH_M     = 10.0                       # clamp depth map a X metri
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 1: Acquisizione GStreamer
# ══════════════════════════════════════════════════════════════════════════════

def gstreamer_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


class StereoCapture:
    """Acquisizione sincrona (software) di due camere CSI IMX219."""

    def __init__(self, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=FPS):
        self.cap_l = cv2.VideoCapture(
            gstreamer_pipeline(0, width, height, fps), cv2.CAP_GSTREAMER)
        self.cap_r = cv2.VideoCapture(
            gstreamer_pipeline(1, width, height, fps), cv2.CAP_GSTREAMER)
        if not self.cap_l.isOpened() or not self.cap_r.isOpened():
            raise RuntimeError(
                "Impossibile aprire le camere. Controlla i cavi CSI e i driver IMX219."
            )
        # Pre-flush buffer
        for _ in range(5):
            self.cap_l.read()
            self.cap_r.read()
        print(f"[OK] Camere aperte  ({width}x{height} @ {fps}fps)")

    def read(self):
        """Ritorna (left, right) o (None, None) se errore."""
        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        if ret_l and ret_r:
            return frame_l, frame_r
        return None, None

    def release(self):
        self.cap_l.release()
        self.cap_r.release()


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 2: Calibrazione e rettifica
# ══════════════════════════════════════════════════════════════════════════════

class StereoRectifier:
    """Carica la calibrazione e produce mappe di rettifica."""

    def __init__(self, calib_file: str, out_size: tuple):
        """
        out_size: (width, height) dell'immagine rettificata desiderata
                  (può essere diversa dalla risoluzione di acquisizione).
        """
        if not os.path.exists(calib_file):
            raise FileNotFoundError(
                f"File di calibrazione non trovato: '{calib_file}'\n"
                "Esegui prima stereo_calibration.py"
            )
        calib = np.load(calib_file)

        img_size = tuple(calib["img_size"])  # dimensione usata in calibrazione

        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            calib["K_l"], calib["D_l"], calib["R_l"], calib["P_l"],
            img_size, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            calib["K_r"], calib["D_r"], calib["R_r"], calib["P_r"],
            img_size, cv2.CV_32FC1)

        self.Q          = calib["Q"]
        self.baseline_m = float(calib["baseline_m"][0])
        self.P_l        = calib["P_l"]
        self.out_size   = out_size  # (w, h) per il modello

        # Focal length dalla matrice P_l rettificata
        self.fx = float(calib["P_l"][0, 0])

        print(f"[OK] Calibrazione caricata  baseline={self.baseline_m*100:.1f}cm  fx={self.fx:.1f}px")

    def rectify(self, left: np.ndarray, right: np.ndarray):
        """Rettifica e ridimensiona al formato richiesto dal modello."""
        rect_l = cv2.remap(left,  self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        if self.out_size:
            rect_l = cv2.resize(rect_l, self.out_size)
            rect_r = cv2.resize(rect_r, self.out_size)
        return rect_l, rect_r

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Converte disparity map (pixel) → depth map (metri).
        depth = fx * baseline / disparity
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                disparity > 0,
                self.fx * self.baseline_m / disparity,
                0.0
            )
        depth = np.clip(depth, 0, MAX_DEPTH_M).astype(np.float32)
        return depth


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 3: Inferenza TensorRT (Fast-FoundationStereo)
# ══════════════════════════════════════════════════════════════════════════════

class FoundationStereoTRT:
    """
    Wrapper TensorRT per Fast-FoundationStereo.
    Presuppone un engine con:
      - 2 input:  "left"  e "right"  shape [1, 3, H, W]  float32
      - 1 output: "disp"             shape [1, 1, H, W]  float32
    Se i nomi o le shape del tuo engine sono diversi, adattare _setup_io().
    """

    def __init__(self, engine_path: str, h: int, w: int):
        try:
            import tensorrt as trt
            import pycuda.driver   as cuda
            import pycuda.autoinit  # noqa: F401  inizializza CUDA context
        except ImportError as e:
            raise ImportError(
                f"Dipendenza mancante: {e}\n"
                "Installa: pip install tensorrt pycuda\n"
                "Su Jetson usa i wheel precompilati da JetPack."
            )

        self.trt  = trt
        self.cuda = cuda
        self.h, self.w = h, w

        # Carica engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self._setup_io()
        print(f"[OK] TRT engine caricato  input={w}x{h}")

    def _setup_io(self):
        import pycuda.driver as cuda

        self.bindings  = []
        self.h_buffers = {}
        self.d_buffers = {}

        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = np.float32  # tutti i binding sono float32
            size  = int(np.prod(shape))

            h_buf = cuda.pagelocked_empty(size, dtype)
            d_buf = cuda.mem_alloc(h_buf.nbytes)

            self.bindings.append(int(d_buf))
            self.h_buffers[name] = h_buf
            self.d_buffers[name] = d_buf

        self.stream = self.cuda.Stream()

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → CHW float32 normalizzato [0,1]."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))  # HWC → CHW

    def infer(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        """
        Input:  left_rect, right_rect — BGR uint8, già rettificate e al formato engine
        Output: disparity map float32, shape (H, W)
        """
        import pycuda.driver as cuda

        left_chw  = self._preprocess(left_rect)
        right_chw = self._preprocess(right_rect)

        np.copyto(self.h_buffers["left"],  left_chw.ravel())
        np.copyto(self.h_buffers["right"], right_chw.ravel())

        # H2D
        for name in ("left", "right"):
            cuda.memcpy_htod_async(self.d_buffers[name], self.h_buffers[name], self.stream)

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # D2H
        cuda.memcpy_dtoh_async(self.h_buffers["disp"], self.d_buffers["disp"], self.stream)
        self.stream.synchronize()

        disp = self.h_buffers["disp"].reshape(self.h, self.w)
        return disp


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 4: Fallback CPU (senza engine TRT — debug / prototipazione)
# ══════════════════════════════════════════════════════════════════════════════

class SGBMStereo:
    """
    Stereo Block Matching GPU-accelerato con OpenCV.
    Usato come fallback se l'engine TRT non è disponibile.
    """

    def __init__(self):
        block = 5
        num_disp = 128
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 3 * block ** 2,
            P2=32 * 3 * block ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY
        )
        # WLS filter per raffinare
        self.wls = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
        self.wls.setLambda(8000)
        self.wls.setSigmaColor(1.5)
        print("[WARN] TRT engine non trovato, uso SGBM come fallback.")

    def infer(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        gray_l = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp_l = self.matcher.compute(gray_l, gray_r)
        disp_r = self.right_matcher.compute(gray_r, gray_l)
        disp   = self.wls.filter(disp_l, gray_l, disparity_map_right=disp_r)
        disp   = disp.astype(np.float32) / 16.0  # OpenCV scala x16
        disp[disp < 0] = 0
        return disp


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 5: Utilità di visualizzazione e salvataggio
# ══════════════════════════════════════════════════════════════════════════════

def colorize_depth(depth_m: np.ndarray, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Converte depth (float32, metri) → immagine BGR colorata."""
    norm = np.clip(depth_m / max_depth, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, COLORMAP)


def save_point_cloud(depth_m: np.ndarray, rgb: np.ndarray,
                     Q: np.ndarray, path: str):
    """
    Salva la point cloud in formato ASCII .ply.
    Q: matrice di reproiezione 4x4 da cv2.stereoRectify
    """
    h, w = depth_m.shape
    fx = Q[2, 3]  # -fx dalla Q (convenzione OpenCV: Q[2,3] = -1/Tx * fx non sempre)
    # Approccio più diretto: usa reprojectImageTo3D
    # La disparity equivalente è fx*baseline/depth
    # Usiamo direttamente la depth e le coordinate pixel
    cx = -Q[0, 3]  # principal point x
    cy = -Q[1, 3]  # principal point y
    f  =  Q[2, 3]  # focal length (Q[2,3] = -1/Tx * (−fx) ... usiamo P_l.fx via rectifier)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    valid  = depth_m > 0.1

    Z = depth_m[valid]
    X = (xs[valid] - cx) * Z / f
    Y = (ys[valid] - cy) * Z / f

    colors = rgb[valid][:, ::-1]  # BGR → RGB

    with open(path, "w") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {Z.size}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        ply.write("end_header\n")
        for i in range(Z.size):
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            ply.write(f"{X[i]:.4f} {Y[i]:.4f} {Z[i]:.4f} {r} {g} {b}\n")

    print(f"[OK] Point cloud salvata: {path}  ({Z.size} punti)")


# ══════════════════════════════════════════════════════════════════════════════
#  Sezione 6: Loop principale
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    os.makedirs("output", exist_ok=True)

    # 1. Camere
    stereo_cap = StereoCapture()

    # 2. Rettificatore
    infer_size  = (INFER_WIDTH, INFER_HEIGHT)
    rectifier   = StereoRectifier(CALIB_FILE, out_size=infer_size)

    # 3. Modello di inferenza
    if os.path.exists(ENGINE_PATH):
        model = FoundationStereoTRT(ENGINE_PATH, INFER_HEIGHT, INFER_WIDTH)
    else:
        print(f"[WARN] Engine '{ENGINE_PATH}' non trovato. Uso SGBM come fallback.")
        print("       Per usare Fast-FoundationStereo segui il README di NVlabs/Fast-FoundationStereo")
        model = SGBMStereo()

    # ── Loop ────────────────────────────────────────────────────────────────
    frame_idx  = 0
    fps_times  = []
    print("\n[→] Pipeline avviata.  Premi Q per uscire, S per salvare frame/PC.\n")

    while True:
        t0 = time.perf_counter()

        # Acquisizione
        left_raw, right_raw = stereo_cap.read()
        if left_raw is None:
            print("[WARN] Frame perso, continuo...")
            continue

        # Rettifica
        left_rect, right_rect = rectifier.rectify(left_raw, right_raw)

        # Inferenza disparity
        t_infer = time.perf_counter()
        disparity = model.infer(left_rect, right_rect)
        infer_ms  = (time.perf_counter() - t_infer) * 1000

        # Depth map
        depth_m = rectifier.disparity_to_depth(disparity)

        # Visualizzazione
        depth_color = colorize_depth(depth_m)
        preview_l   = cv2.resize(left_rect,  (INFER_WIDTH // 2, INFER_HEIGHT // 2))
        preview_r   = cv2.resize(right_rect, (INFER_WIDTH // 2, INFER_HEIGHT // 2))
        depth_small = cv2.resize(depth_color,(INFER_WIDTH, INFER_HEIGHT))

        top_row = np.hstack([preview_l, preview_r])
        # pad top_row a larghezza INFER_WIDTH
        if top_row.shape[1] < INFER_WIDTH:
            pad = np.zeros((top_row.shape[0], INFER_WIDTH - top_row.shape[1], 3), np.uint8)
            top_row = np.hstack([top_row, pad])

        canvas = np.vstack([top_row, depth_small])

        # FPS
        fps_times.append(time.perf_counter() - t0)
        if len(fps_times) > 20:
            fps_times.pop(0)
        fps_val = 1.0 / (sum(fps_times) / len(fps_times))

        # Overlay testo
        cv2.putText(canvas, f"FPS: {fps_val:.1f}  Infer: {infer_ms:.0f}ms",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        valid_pts  = np.sum(depth_m > 0)
        mean_depth = float(np.mean(depth_m[depth_m > 0])) if valid_pts else 0
        cv2.putText(canvas, f"Depth medio: {mean_depth:.2f}m  Punti: {valid_pts}",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        cv2.imshow("FoundationStereo Pipeline  [Q=esci  S=salva]", canvas)

        # Salvataggio automatico depth ogni N frame
        if args.save_depth:
            depth_path = f"output/depth_{frame_idx:06d}.png"
            depth_norm = (np.clip(depth_m / MAX_DEPTH_M, 0, 1) * 65535).astype(np.uint16)
            cv2.imwrite(depth_path, depth_norm)

        # Salvataggio automatico point cloud ogni SAVE_EVERY_N frame
        if args.save_pc and frame_idx % SAVE_EVERY_N == 0 and frame_idx > 0:
            pc_path = f"output/cloud_{frame_idx:06d}.ply"
            save_point_cloud(depth_m, left_rect, rectifier.Q, pc_path)

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            # Salvataggio manuale
            tag = f"{frame_idx:06d}"
            cv2.imwrite(f"output/left_{tag}.png",  left_rect)
            cv2.imwrite(f"output/right_{tag}.png", right_rect)
            depth_u16 = (np.clip(depth_m / MAX_DEPTH_M, 0, 1) * 65535).astype(np.uint16)
            cv2.imwrite(f"output/depth_{tag}.png", depth_u16)
            save_point_cloud(depth_m, left_rect, rectifier.Q, f"output/cloud_{tag}.ply")
            print(f"[S] Frame {tag} salvato in output/")

    # Cleanup
    cv2.destroyAllWindows()
    stereo_cap.release()
    print("[→] Pipeline terminata.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Pipeline IMX219 + Fast-FoundationStereo")
    parser.add_argument("--save_depth", action="store_true",
                        help="Salva ogni depth map come PNG 16-bit in output/")
    parser.add_argument("--save_pc",    action="store_true",
                        help=f"Salva point cloud PLY ogni {SAVE_EVERY_N} frame in output/")
    parser.add_argument("--engine",     default=ENGINE_PATH,
                        help=f"Percorso TRT engine (default: {ENGINE_PATH})")
    parser.add_argument("--calib",      default=CALIB_FILE,
                        help=f"File calibrazione (default: {CALIB_FILE})")
    args = parser.parse_args()

    ENGINE_PATH = args.engine
    CALIB_FILE  = args.calib

    run(args)
