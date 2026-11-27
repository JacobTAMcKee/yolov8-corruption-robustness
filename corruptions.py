import cv2
import numpy as np


def gaussian_blur(img, k):
    """Blur with a k x k Gaussian kernel (k must be odd). Bigger k = blurrier."""
    return cv2.GaussianBlur(img, (k, k), 0)


def gaussian_noise(img, sigma):
    """Add zero-mean Gaussian noise with standard deviation sigma (in 0-255 units)."""
    # Seed from the image itself so results are reproducible but differ per image
    rng = np.random.default_rng(int(img.sum()) % (2**32))
    noise = rng.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)  # keep valid pixel range


def darken(img, factor):
    """Multiply all pixels by factor (< 1). Smaller factor = darker."""
    dark = img.astype(np.float32) * factor
    return np.clip(dark, 0, 255).astype(np.uint8)


def jpeg_compress(img, quality):
    """Encode as JPEG at the given quality (1-100) and decode back. Lower = more artifacts."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# name -> (function, [mild, medium, severe])
CORRUPTIONS = {
    "blur":   (gaussian_blur,   [5, 11, 21]),
    "noise":  (gaussian_noise,  [10, 25, 50]),
    "dark":   (darken,          [0.7, 0.4, 0.2]),
    "jpeg":   (jpeg_compress,   [50, 20, 5]),
}