"""
Wink-Style Image Enhancement Engine
High-fidelity classical computer vision and DSP pipeline featuring:
- Multi-scale Lanczos-4 Super-Resolution (1x, 2x, 4x)
- Wink-style Portrait Retouching with 70/30 skin texture preservation
- Eye and lip localized contrast & sharpness enhancement
- Contrast-Adaptive Sharpening (CAS) and luminance-isolated unsharp masking
- Dynamic Tone & Vibrance Boost via LAB CLAHE and non-linear saturation
- Dual-domain bilateral denoising filter
"""

import os
import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image


def calculate_tenengrad(image: np.ndarray) -> float:
    """
    Calculate the Tenengrad edge gradient energy metric.
    Higher values correspond to sharper, crisper edges.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobel gradient in x and y directions
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_sq = gx ** 2 + gy ** 2
    
    # Tenengrad thresholding (standard threshold 100)
    mean_energy = float(np.mean(grad_sq))
    return mean_energy


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    """
    if img1.shape != img2.shape:
        # Resize img2 to match img1 if dimensions differ
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)
        
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


# Global cached Haar Cascade classifier so video frame processing never reloads XML from disk per frame
_FACE_CASCADE: Optional[Any] = None


def _get_face_cascade() -> Optional[Any]:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is None:
        return None
    cascade_dir = getattr(cv2, "data", None)
    if cascade_dir and hasattr(cascade_dir, "haarcascades"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            try:
                _FACE_CASCADE = cascade_cls(cascade_path)
            except Exception:
                _FACE_CASCADE = None
    return _FACE_CASCADE


def denoise_image(
    image: np.ndarray,
    strength: float = 0.3,
    denoise_strength: Optional[float] = None,
) -> np.ndarray:
    """
    Edge-preserving dual-domain chrominance and flat-area luminance denoising.
    Cleans up compression artifacts and sensor blotches without softening high-frequency edges or text.
    Supports both `strength` and `denoise_strength` keyword arguments.
    """
    eff_strength = float(denoise_strength if denoise_strength is not None else strength)
    if eff_strength <= 0.0:
        return image
    
    eff_strength = min(1.0, max(0.0, eff_strength))
    
    # 1. Chroma-domain bilateral denoising in YCrCb (removes color blotches without touching edge sharpness)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    d = int(np.clip(3 + 4 * eff_strength, 3, 9))
    sigma_c = int(np.clip(18 + 40 * eff_strength, 10, 65))
    cr_clean = cv2.bilateralFilter(cr, d=d, sigmaColor=sigma_c, sigmaSpace=12)
    cb_clean = cv2.bilateralFilter(cb, d=d, sigmaColor=sigma_c, sigmaSpace=12)
    
    # 2. Edge-guided luminance noise suppression
    # Detect edges on luminance channel so text and contours stay 100% sharp
    sobelx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    # Edge mask: 1.0 on sharp edges/text, 0.0 on flat areas
    edge_mask = np.clip(edge_mag / 24.0, 0.0, 1.0)
    
    # Bilateral smoothing applied only to flat/gradient regions
    y_smooth = cv2.bilateralFilter(y, d=d, sigmaColor=int(20 * eff_strength + 10), sigmaSpace=10)
    y_denoised = (y_smooth.astype(np.float32) * (1.0 - edge_mask) + y.astype(np.float32) * edge_mask).astype(np.uint8)
    
    y_final = cv2.addWeighted(y_denoised, eff_strength, y, float(1.0 - eff_strength), 0)
    merged = cv2.merge([y_final, cr_clean, cb_clean])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def segment_skin(image: np.ndarray) -> np.ndarray:
    """
    Segment skin tone pixels using combined YCbCr and HSV color thresholds.
    Returns a smooth feathered single-channel mask (0.0 to 1.0).
    """
    # 1. YCbCr skin bounds: 77 <= Cb <= 127 and 133 <= Cr <= 173
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _y, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    
    # 2. HSV skin bounds: H in [0, 25] or [165, 180], S in [28, 180], V >= 50
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_hsv = ((h <= 25) | (h >= 165)) & (s >= 28) & (s <= 180) & (v >= 50)
    
    combined = (mask_ycrcb & mask_hsv).astype(np.uint8) * 255
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Feather edges with Gaussian blur for natural seamless blending
    feathered = cv2.GaussianBlur(combined, (9, 9), 3.0)
    return feathered.astype(np.float32) / 255.0


def apply_portrait_retouch(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Wink-style portrait retouching:
    1. Detect faces or verify presence of prominent facial skin cluster
    2. Guard against blurring non-face images (screenshots, UI, text, anime, landscape)
    3. Apply dual-domain bilateral skin smoothing with 70/30 natural pore texture blending
    4. Detect eyes and lips to apply localized contrast and sharpness boost
    """
    if strength <= 0.0:
        return image
        
    h, w = image.shape[:2]
    result = image.copy()
    
    # Detect face bounding boxes if possible
    face_boxes = []
    face_cascade = _get_face_cascade()
    if face_cascade is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        det_scale = 0.5 if max(h, w) > 640 else 1.0
        small_gray = cv2.resize(gray, (0, 0), fx=det_scale, fy=det_scale) if det_scale < 1.0 else gray
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.12, minNeighbors=3, minSize=(20, 20))
        inv = 1.0 / det_scale
        face_boxes = [(int(round(fx * inv)), int(round(fy * inv)), int(round(fw * inv)), int(round(fh * inv))) for (fx, fy, fw, fh) in faces]
        
    # Get skin mask
    skin_mask = segment_skin(image)
    
    # Face & Cluster Guard:
    # If no Haar cascade faces were found, verify whether skin pixels form a genuine face cluster.
    # In screenshots, games, landscapes, or text, skin-tone pixels are tiny scattered fragments (wood, orange buttons, icons).
    if not face_boxes:
        binary_skin = (skin_mask > 0.25).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_skin)
        if num_labels <= 1:
            return image
        max_component_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
        total_pixels = float(h * w)
        rel_area_pct = (max_component_area / total_pixels) * 100.0
        
        # If the largest skin-tone patch is under 4.0% or scattered over many tiny noisy patches, it is NOT a portrait!
        if rel_area_pct < 4.0 or num_labels > 45:
            return image
            
        # For low-res portraits where Haar cascade missed, only smooth the largest skin component
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cluster_mask = (labels == largest_label).astype(np.float32)
        cluster_mask = cv2.GaussianBlur(cluster_mask, (11, 11), 3.0)
        skin_mask = np.clip(skin_mask * cluster_mask, 0.0, 1.0)
    else:
        # Haar faces exist: restrict skin smoothing strictly to elliptical face regions
        face_mask = np.zeros((h, w), dtype=np.float32)
        for (fx, fy, fw, fh) in face_boxes:
            center = (fx + fw // 2, fy + fh // 2)
            axes = (int(fw * 0.45), int(fh * 0.55))
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (15, 15), 5.0)
        skin_mask = np.clip(skin_mask * face_mask, 0.0, 1.0)
        
    # Bilateral smoothing for skin
    smoothed_skin = cv2.bilateralFilter(image, d=9, sigmaColor=65, sigmaSpace=65)
    
    # Natural pore texture preservation:
    # 70% smoothed + 30% original texture (scaled by retouch strength)
    smooth_weight = 0.70 * float(strength)
    original_texture_weight = 1.0 - smooth_weight
    blended_skin = cv2.addWeighted(smoothed_skin, smooth_weight, image, original_texture_weight, 0)
    
    # Apply blended skin only where skin_mask is active
    skin_mask_3c = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2)
    result = (blended_skin * skin_mask_3c + image * (1.0 - skin_mask_3c)).astype(np.uint8)
    
    # Eye and lip localized enhancement
    # If faces were detected, isolate eyes (upper 25%-50%) and lips (lower 65%-85%)
    if face_boxes:
        for (fx, fy, fw, fh) in face_boxes:
            # Eyes region
            ey_start = int(fy + fh * 0.22)
            ey_end = int(fy + fh * 0.52)
            ex_start = int(fx + fw * 0.12)
            ex_end = int(fx + fw * 0.88)
            
            # Lips region
            ly_start = int(fy + fh * 0.65)
            ly_end = int(fy + fh * 0.90)
            lx_start = int(fx + fw * 0.25)
            lx_end = int(fx + fw * 0.75)
            
            for (y1, y2, x1, x2, boost_sharp, boost_contrast) in [
                (ey_start, ey_end, ex_start, ex_end, 1.30, 1.18),  # Eyes
                (ly_start, ly_end, lx_start, lx_end, 1.20, 1.15),  # Lips
            ]:
                if y2 > y1 and x2 > x1 and y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                    roi = result[y1:y2, x1:x2].astype(np.float32)
                    
                    # Contrast boost around mean
                    roi_mean = np.mean(roi, axis=(0, 1), keepdims=True)
                    contrast_factor = 1.0 + (boost_contrast - 1.0) * float(strength)
                    roi_contrasted = np.clip((roi - roi_mean) * contrast_factor + roi_mean, 0, 255)
                    
                    # Crisp unsharp mask for eyes/lips
                    roi_blur = cv2.GaussianBlur(roi_contrasted, (3, 3), 1.0)
                    sharp_factor = (boost_sharp - 1.0) * float(strength) * 1.8
                    roi_enhanced = np.clip(roi_contrasted + sharp_factor * (roi_contrasted - roi_blur), 0, 255)
                    
                    # Feathered blend for ROI
                    roi_mask = np.zeros((y2 - y1, x2 - x1, 1), dtype=np.float32)
                    cv2.ellipse(
                        roi_mask,
                        ((x2 - x1) // 2, (y2 - y1) // 2),
                        (int((x2 - x1) * 0.45), int((y2 - y1) * 0.45)),
                        0, 0, 360, 1.0, -1
                    )
                    roi_mask = cv2.GaussianBlur(roi_mask, (7, 7), 2.0)
                    if roi_mask.ndim == 2:
                        roi_mask = roi_mask[:, :, np.newaxis]
                        
                    result[y1:y2, x1:x2] = (roi_enhanced * roi_mask + result[y1:y2, x1:x2] * (1.0 - roi_mask)).astype(np.uint8)
                    
    return result


def apply_upscale(
    image: np.ndarray,
    scale_factor: float = 2.0,
    upscale_factor: Optional[Any] = None,
) -> np.ndarray:
    """
    High-grade multi-scale Lanczos-4 Super-Resolution upscaling helper.
    Supports both `scale_factor` and `upscale_factor` kwargs.
    Enforces even dimensions for video codec compatibility.
    """
    factor = float(upscale_factor if upscale_factor is not None else scale_factor)
    if factor <= 1.0 or factor <= 0:
        return image
        
    orig_h, orig_w = image.shape[:2]
    new_w = max(2, int(round(orig_w * factor)))
    new_h = max(2, int(round(orig_h * factor)))
    # Ensure even dimensions for video codec compatibility
    new_w += new_w % 2
    new_h += new_h % 2
    
    if factor == 2.0:
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    elif factor == 4.0:
        # 2-stage multi-scale Lanczos-4 upscaling
        mid_w, mid_h = orig_w * 2, orig_h * 2
        mid_w += mid_w % 2
        mid_h += mid_h % 2
        mid_img = cv2.resize(image, (mid_w, mid_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Subtle intermediate edge refinement
        gray_mid = cv2.cvtColor(mid_img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray_mid, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_mid, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_norm = np.clip(grad_mag / (np.max(grad_mag) + 1e-5), 0.0, 1.0)
        grad_mask = np.repeat(grad_norm[:, :, np.newaxis], 3, axis=2)
        
        # Reinforce edges slightly between stages
        mid_blur = cv2.GaussianBlur(mid_img, (3, 3), 0.8)
        mid_refined = np.clip(mid_img.astype(np.float32) + 0.25 * grad_mask * (mid_img.astype(np.float32) - mid_blur.astype(np.float32)), 0, 255).astype(np.uint8)
        
        return cv2.resize(mid_refined, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Arbitrary factor fallback
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def apply_vibrance_and_tone(
    image: np.ndarray,
    strength: float = 0.3,
    vibrance: Optional[float] = None,
) -> np.ndarray:
    """
    Dynamic tone and vibrance boost:
    1. LAB CLAHE on L-channel with adaptive grid size and highlight/shadow protection
       (prevents blowing out white UI cards or crushing dark backgrounds)
    2. Intelligent non-linear saturation curve: boosts muted tones while protecting skin/vibrant areas
    Supports both `strength` and `vibrance` kwargs.
    """
    eff_vibrance = float(vibrance if vibrance is not None else strength)
    if eff_vibrance <= 0.0:
        return image
        
    eff_vibrance = min(1.0, max(0.0, eff_vibrance))
    h, w = image.shape[:2]
    
    # 1. LAB CLAHE on luminance with adaptive grid size
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_float = l_channel.astype(np.float32)
    
    grid_size = 8 if max(h, w) < 256 else (12 if max(h, w) < 1024 else 16)
    clip_limit = 1.0 + 1.8 * eff_vibrance
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe.apply(l_channel).astype(np.float32)
    
    # Highlight & shadow protection to avoid blowing out whites or crushing blacks
    protect = np.clip((l_float - 12.0) / 25.0, 0.0, 1.0) * np.clip((245.0 - l_float) / 25.0, 0.0, 1.0)
    l_toned = l_float * (1.0 - protect * eff_vibrance * 0.8) + l_clahe * (protect * eff_vibrance * 0.8)
    l_final = np.clip(l_toned, 0, 255).astype(np.uint8)
    
    # Recombine LAB
    lab_enhanced = cv2.merge([l_final, a_channel, b_channel])
    bgr_tone = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 2. Intelligent non-linear vibrance curve in HSV
    hsv = cv2.cvtColor(bgr_tone, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Normalize saturation to [0, 1]
    s_norm = s / 255.0
    
    # Non-linear boost: delta_s = 1.0 - s_norm
    # Muted pixels (low s_norm) get maximum relative boost; saturated pixels stay protected
    delta_s = 1.0 - s_norm
    s_boosted = s_norm + (eff_vibrance * 0.45 * delta_s * s_norm)
    s_new = np.clip(s_boosted * 255.0, 0, 255).astype(np.float32)
    
    hsv_boosted = cv2.merge([h, s_new, v])
    result = cv2.cvtColor(hsv_boosted.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Blend with original according to eff_vibrance
    blended = cv2.addWeighted(result, eff_vibrance, image, float(1.0 - eff_vibrance), 0)
    return blended


enhance_tone_vibrance = apply_vibrance_and_tone


def apply_cas_sharpening(
    image: np.ndarray,
    strength: float = 0.5,
    amount: Optional[float] = None,
) -> np.ndarray:
    """
    Scale-adaptive Contrast-Adaptive Sharpening (CAS) and dual-frequency luminance unsharp masking.
    Operates strictly on the L-channel in LAB space to eliminate color fringing.
    Includes edge gradient weighting and soft non-linear saturation to completely eliminate white halos around text.
    Provides crisp edge definition, vector-like lineart, and micro-texture recovery.
    Supports both `strength` and `amount` kwargs.
    """
    eff_amount = float(amount if amount is not None else strength)
    if eff_amount <= 0.0:
        return image
        
    eff_amount = min(1.0, max(0.0, eff_amount))
    h, w = image.shape[:2]
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    
    # Scale-adaptive sigma so high-res images get proportional micro-detail
    scale_ref = max(1.0, min(h, w) / 360.0)
    sigma_fine = max(0.75, 0.65 * min(scale_ref, 4.0))
    sigma_med = max(2.0, 1.6 * min(scale_ref, 4.0))
    
    # Dual-frequency high-pass bands for fine details and structural contours
    blur_fine = cv2.GaussianBlur(l_float, (0, 0), sigma_fine)
    blur_med = cv2.GaussianBlur(l_float, (0, 0), sigma_med)
    hp_fine = l_float - blur_fine
    hp_med = l_float - blur_med
    hp = 0.70 * hp_fine + 0.60 * hp_med
    
    # Sobel gradient for edge awareness (protects flat panels and skies from noise)
    gx = cv2.Sobel(blur_fine, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur_fine, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_weight = np.clip(edge_mag / 20.0, 0.2, 1.4)
    
    # Anti-halo soft non-linear clamping (prevents ugly white halos around text letters & UI)
    sharpen_gain = 1.0 + 3.0 * eff_amount
    clamped_hp = np.tanh((hp * edge_weight * sharpen_gain) / 36.0) * 36.0
    
    l_sharp = np.clip(l_float + clamped_hp, 0, 255).astype(np.uint8)
    
    lab_sharp = cv2.merge([l_sharp, a, b])
    result = cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
    return result


contrast_adaptive_sharpening = apply_cas_sharpening


def enhance_image(
    input_path: str,
    output_path: str,
    scale_factor: float = 2.0,
    denoise_strength: float = 0.25,
    retouch_strength: float = 0.40,
    sharpen_amount: float = 0.45,
    vibrance_boost: float = 0.20,
    clahe_clip: float = 1.5,
    # Keyword aliases used by app/main.py and tests:
    upscale_factor: Optional[int] = None,
    face_retouch: Optional[float] = None,
    sharpen: Optional[float] = None,
    vibrance: Optional[float] = None,
    denoise: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Core Image Enhancement Function.
    Supports both legacy and PRD parameter conventions.
    """
    start_time = time.perf_counter()

    eff_scale = float(upscale_factor if upscale_factor is not None else scale_factor)
    eff_denoise = float(denoise if denoise is not None else denoise_strength)
    eff_retouch = float(face_retouch if face_retouch is not None else retouch_strength)
    eff_sharpen = float(sharpen if sharpen is not None else sharpen_amount)
    eff_vibrance = float(vibrance if vibrance is not None else vibrance_boost)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at: {input_path}")

    h_in, w_in = img.shape[:2]
    orig_tenengrad = calculate_tenengrad(img)

    target_w = max(2, int(round(w_in * eff_scale)))
    target_h = max(2, int(round(h_in * eff_scale)))
    if eff_scale > 1.0:
        target_w += target_w % 2
        target_h += target_h % 2

    baseline_upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    baseline_sharpness = calculate_tenengrad(baseline_upscaled)

    # Step 1: Denoise filter
    current = denoise_image(img, strength=eff_denoise)

    # Step 2: Portrait Retouching & Natural Skin Smoothing
    current = apply_portrait_retouch(current, strength=eff_retouch)

    # Step 3: High-grade Multi-Scale Lanczos-4 Super-Resolution
    if eff_scale != 1.0:
        current = apply_upscale(current, scale_factor=eff_scale)

    # Step 4: Dynamic Tone & Vibrance Boost
    current = apply_vibrance_and_tone(current, strength=eff_vibrance)

    # Step 5: Deblur & Contrast-Adaptive Sharpening (CAS)
    current = apply_cas_sharpening(current, strength=eff_sharpen)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(output_path, current, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    elif ext == ".png":
        cv2.imwrite(output_path, current, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    elif ext == ".webp":
        cv2.imwrite(output_path, current, [int(cv2.IMWRITE_WEBP_QUALITY), 98])
    else:
        success = cv2.imwrite(output_path, current)
        if not success:
            pil_img = Image.fromarray(cv2.cvtColor(current, cv2.COLOR_BGR2RGB))
            pil_img.save(output_path)

    duration_sec = time.perf_counter() - start_time
    processing_time_ms = duration_sec * 1000.0

    final_h, final_w = current.shape[:2]
    enhanced_sharpness = calculate_tenengrad(current)
    sharpness_gain = (
        enhanced_sharpness / baseline_sharpness
        if baseline_sharpness > 0 else 1.0
    )
    psnr_baseline = calculate_psnr(baseline_upscaled, current)
    psnr_val = calculate_psnr(img, current)
    ssim_val = calculate_ssim(img, current)
    tenengrad_gain = round(((enhanced_sharpness - orig_tenengrad) / max(orig_tenengrad, 1e-5)) * 100, 2)

    return {
        "status": "success",
        "input_path": input_path,
        "output_path": output_path,
        "input_resolution": [w_in, h_in],
        "output_resolution": [final_w, final_h],
        "original_resolution": f"{w_in}x{h_in}",
        "scale_factor": eff_scale,
        "processing_time_sec": round(duration_sec, 3),
        "processing_time_ms": round(processing_time_ms, 2),
        "tenengrad_before": round(orig_tenengrad, 2),
        "tenengrad_after": round(enhanced_sharpness, 2),
        "tenengrad_improvement_pct": tenengrad_gain,
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "metrics": {
            "sharpness_gain": round(sharpness_gain, 4),
            "psnr_baseline": round(psnr_baseline, 2),
            "tenengrad_before": round(orig_tenengrad, 2),
            "tenengrad_after": round(enhanced_sharpness, 2),
            "tenengrad_gain": tenengrad_gain,
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
        },
    }
