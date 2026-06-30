import numpy as np
import cv2

TARGET_SIZE = 900 
PATCH_SIZE = 512 

def _load_full(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img

def _resize_longest(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def _best_patches(gray_full: np.ndarray, n=4, size=PATCH_SIZE):
    h, w = gray_full.shape
    if h < size or w < size:
        size = min(h, w)
    step_y = max(1, (h - size) // 4)
    step_x = max(1, (w - size) // 4)
    candidates = []
    for y in range(0, max(1, h - size + 1), max(1, step_y)):
        for x in range(0, max(1, w - size + 1), max(1, step_x)):
            patch = gray_full[y:y + size, x:x + size]
            if patch.shape[0] != size or patch.shape[1] != size:
                continue
            std = patch.std()
            candidates.append((std, patch))
    if not candidates:
        return [gray_full[:size, :size]]
    candidates.sort(key=lambda t: t[0])
    mid = len(candidates) // 2
    picks = candidates[max(0, mid - n // 2): max(0, mid - n // 2) + n]
    if not picks:
        picks = candidates[-n:]
    return [p for _, p in picks]


def _moire_features(gray_full: np.ndarray) -> dict:
    patches = _best_patches(gray_full, n=4, size=PATCH_SIZE)
    peak_ratios, peakinesses, high_es = [], [], []
    for patch in patches:
        win = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
        f = np.fft.fft2(patch.astype(np.float32) * win)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        mag_log = np.log1p(mag)

        h, w = patch.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        rmax = r.max()

        band_mask = (r >= 0.15 * rmax) & (r < 0.45 * rmax)
        band_vals = mag_log[band_mask]
        band_mean = band_vals.mean() + 1e-8
        band_max = band_vals.max()
        band_std = band_vals.std()

        peak_ratios.append(float(band_max / band_mean))
        peakinesses.append(float(band_std / band_mean))
        total = mag.sum() + 1e-8
        high_es.append(float(mag[r >= 0.45 * rmax].sum() / total))

    return {
        "moire_peak_ratio_max": float(np.max(peak_ratios)),
        "moire_peak_ratio_mean": float(np.mean(peak_ratios)),
        "moire_peakiness_max": float(np.max(peakinesses)),
        "moire_high_e_mean": float(np.mean(high_es)),
    }


def _fft_features(gray: np.ndarray) -> dict:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rmax = r.max()

    low_mask = r < 0.08 * rmax
    mid_mask = (r >= 0.08 * rmax) & (r < 0.35 * rmax)
    high_mask = r >= 0.35 * rmax

    total = mag.sum() + 1e-8
    low_e = mag[low_mask].sum() / total
    mid_e = mag[mid_mask].sum() / total
    high_e = mag[high_mask].sum() / total

    ring_vals = mag_log[high_mask]
    ring_mean = ring_vals.mean() + 1e-8
    ring_max = ring_vals.max()
    peak_ratio = ring_max / ring_mean

    ring_std = ring_vals.std()
    peakiness = ring_std / ring_mean

    return {
        "fft_low_e": float(low_e),
        "fft_mid_e": float(mid_e),
        "fft_high_e": float(high_e),
        "fft_peak_ratio": float(peak_ratio),
        "fft_peakiness": float(peakiness),
    }

def _row_periodicity(gray: np.ndarray) -> float:

    row_mean = gray.mean(axis=1).astype(np.float32)
    row_mean = row_mean - row_mean.mean()
    spec = np.abs(np.fft.rfft(row_mean))
    if len(spec) < 4:
        return 0.0
    spec = spec[2:]  # drop DC / very-low-freq (global gradients/vignetting)
    total = spec.sum() + 1e-8
    peak = spec.max()
    return float(peak / total)

def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _color_features(img_bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    sat_mean = float(s.mean())
    sat_std = float(s.std())

    clip_frac = float((v > 0.97).mean())

    b, g, r = cv2.split(img_bgr.astype(np.float32) / 255.0)
    means = np.array([b.mean(), g.mean(), r.mean()])
    channel_imbalance = float(means.std() / (means.mean() + 1e-8))

    return {
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "clip_frac": clip_frac,
        "channel_imbalance": channel_imbalance,
    }


def extract_features(image_path: str) -> dict:
    img_full = _load_full(image_path)
    gray_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)

    img = _resize_longest(img_full, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = {}
    feats.update(_fft_features(gray))
    feats.update(_moire_features(gray_full))
    feats["row_periodicity"] = _row_periodicity(gray)
    feats["sharpness"] = _sharpness(gray)
    feats.update(_color_features(img))
    return feats


FEATURE_NAMES = [
    "fft_low_e", "fft_mid_e", "fft_high_e", "fft_peak_ratio", "fft_peakiness",
    "moire_peak_ratio_max", "moire_peak_ratio_mean", "moire_peakiness_max", "moire_high_e_mean",
    "row_periodicity", "sharpness", "sat_mean", "sat_std", "clip_frac",
    "channel_imbalance",
]


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
