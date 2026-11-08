"""
Wink HD Enhancer - Overpowered 7-Stage Ultra-HD Image & Video Frame Engine
Featuring:
1. Chroma-Isolated & Edge-Protected Bilateral Denoising
2. 3-Band Frequency Separation Portrait Retouching + Eye/Lip/Hair Micro-Detail Boost
3. Multi-Stage Lanczos-4 + Iterative Back-Projection (IBP) Super-Resolution
4. Sub-Pixel Morphological Edge-Shock Steepening (De-blurring soft edges)
5. 3-Scale Laplacian/DoG Pyramid Detail & Local HDR Clarity Booster (with Tanh Halo Limiter)
6. Overpowered Contrast-Adaptive Sharpening (CAS Pro)
7. Auto Black-Point Dehaze, Cinema S-Curve Tone Mapping & Skin-Safe Vibrance
"""

import os
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np

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
        cascade_path = cascade_dir.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            try:
                _FACE_CASCADE = cascade_cls(cascade_path)
            except Exception:
                _FACE_CASCADE = None
    return _FACE_CASCADE


def calculate_tenengrad(image: np.ndarray) -> float:
    """
    Calculate the Tenengrad edge gradient energy metric.
    Higher values correspond to sharper, crisper edges.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_sq = sobel_x ** 2 + sobel_y ** 2
    return float(np.mean(grad_sq))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


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


def denoise_image(
    image: np.ndarray,
    strength: float = 0.3,
    denoise_strength: Optional[float] = None,
) -> np.ndarray:
    """
    Stage 1: Chroma-Isolated & Edge-Protected Denoising.
    Eliminates color noise & JPEG blocking in YCrCb chroma channels while protecting fine luminance edges
    so sharp details (hair, eyes, textures) are NEVER blurred away before super-resolution.
    """
    eff_strength = float(denoise_strength if denoise_strength is not None else strength)
    if eff_strength <= 0.0:
        return image

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 1. Aggressive chroma denoising (removes color blotches and JPEG chroma artifacts without hurting sharpness)
    chroma_ksize = 5 if eff_strength >= 0.25 else 3
    cr_clean = cv2.GaussianBlur(cr, (chroma_ksize, chroma_ksize), 1.2)
    cb_clean = cv2.GaussianBlur(cb, (chroma_ksize, chroma_ksize), 1.2)

    # 2. Edge-protected Luminance denoising: only smooth flat noisy regions, keep strong edges 100% intact
    d = int(np.clip(5 + 4 * eff_strength, 5, 9))
    sigma_color = int(np.clip(18 + 38 * eff_strength, 15, 60))
    sigma_space = int(np.clip(12 + 25 * eff_strength, 10, 40))
    y_smooth = cv2.bilateralFilter(y, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Compute local gradient magnitude on Y so edges resist smoothing
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    # Normalize edge protection mask: 0.0 on flat areas (smooth noise), 1.0 on edges/textures (preserve detail)
    edge_protect = np.clip(grad_mag / 42.0, 0.0, 1.0)
    blend_y_weight = float(np.clip(eff_strength * 0.75, 0.0, 0.85)) * (1.0 - edge_protect * 0.85)

    y_final = (y_smooth.astype(np.float32) * blend_y_weight + y.astype(np.float32) * (1.0 - blend_y_weight))
    y_final = np.clip(y_final, 0, 255).astype(np.uint8)

    merged = cv2.merge([y_final, cr_clean, cb_clean])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def segment_skin(image: np.ndarray) -> np.ndarray:
    """
    Segment skin tone pixels using combined YCrCb and HSV color thresholds.
    Returns a smooth feathered single-channel mask (0.0 to 1.0).
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _y, cr, cb = cv2.split(ycrcb)
    mask_ycrcb = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_hsv = ((h <= 22) | (h >= 168)) & (s >= 30) & (s <= 175) & (v >= 55)

    combined = (mask_ycrcb & mask_hsv).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    feathered = cv2.GaussianBlur(combined, (9, 9), 3.0)
    return feathered.astype(np.float32) / 255.0


def apply_portrait_retouch(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Stage 2: 3-Band Frequency Separation Portrait Retouching & Facial Feature HD Boost.
    - Separates Low/Mid skin blotches from High-Frequency pore & hair texture so skin looks airbrushed
      yet 100% natural and crisp (never blurry/plastic).
    - Boosts clarity, catchlights, and micro-sharpness on Eyes, Eyebrows, and Lips.
    """
    if strength <= 0.0:
        return image

    h, w = image.shape[:2]
    result = image.copy()

    face_boxes = []
    face_cascade = _get_face_cascade()
    if face_cascade is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        det_scale = 0.5 if max(h, w) > 640 else 1.0
        small_gray = cv2.resize(gray, (0, 0), fx=det_scale, fy=det_scale) if det_scale < 1.0 else gray
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.15, minNeighbors=3, minSize=(24, 24))
        inv = int(round(1.0 / det_scale))
        face_boxes = [(fx * inv, fy * inv, fw * inv, fh * inv) for (fx, fy, fw, fh) in faces]

    skin_mask = segment_skin(image)

    # Restrict strong skin smoothing primarily to detected faces so backgrounds aren't softened
    if face_boxes:
        face_mask = np.zeros((h, w), dtype=np.float32)
        for (fx, fy, fw, fh) in face_boxes:
            center = (fx + fw // 2, fy + fh // 2)
            axes = (int(fw * 0.48), int(fh * 0.58))
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (17, 17), 6.0)
        skin_mask = np.clip(skin_mask * 0.25 + (skin_mask * face_mask) * 0.85, 0.0, 1.0)
    else:
        # Without a detected face, keep skin smoothing subtle to protect background textures
        skin_mask = skin_mask * 0.35

    # Frequency Separation on Luminance:
    # High-frequency texture = original - small_blur (pores, fine hair strands)
    # Mid-frequency blotches smoothed via bilateral filter
    img_f = image.astype(np.float32)
    fine_blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.2)
    high_freq_texture = img_f - fine_blur  # Fine pores & hair strands

    smooth_base = cv2.bilateralFilter(image, d=7, sigmaColor=55, sigmaSpace=55).astype(np.float32)
    # Re-inject 65% of clean high-frequency micro-texture onto the smoothed base
    freq_sep_skin = np.clip(smooth_base + high_freq_texture * 0.65, 0, 255)

    smooth_weight = float(np.clip(strength * 0.72, 0.0, 0.85))
    skin_alpha = (skin_mask * smooth_weight)[:, :, np.newaxis]
    result = np.clip(freq_sep_skin * skin_alpha + img_f * (1.0 - skin_alpha), 0, 255).astype(np.uint8)

    # Localized HD Eye, Eyebrow & Lip Micro-Contrast & Catchlight Enhancement
    if face_boxes:
        for (fx, fy, fw, fh) in face_boxes:
            ey_start, ey_end = int(fy + fh * 0.18), int(fy + fh * 0.52)
            ex_start, ex_end = int(fx + fw * 0.10), int(fx + fw * 0.90)

            ly_start, ly_end = int(fy + fh * 0.64), int(fy + fh * 0.92)
            lx_start, lx_end = int(fx + fw * 0.22), int(fx + fw * 0.78)

            for (y1, y2, x1, x2, sharp_gain, contrast_gain) in [
                (ey_start, ey_end, ex_start, ex_end, 1.85, 1.22),  # Eyes & Eyebrows Ultra-Crisp
                (ly_start, ly_end, lx_start, lx_end, 1.45, 1.15),  # Lips Definition
            ]:
                if y2 > y1 and x2 > x1 and y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                    roi = result[y1:y2, x1:x2].astype(np.float32)
                    mean_val = np.mean(roi, axis=(0, 1), keepdims=True)
                    roi_boosted = (roi - mean_val) * contrast_gain + mean_val

                    blur_fine = cv2.GaussianBlur(roi_boosted, (0, 0), sigmaX=1.0)
                    detail = roi_boosted - blur_fine
                    roi_sharp = roi_boosted + detail * sharp_gain
                    roi_final = np.clip(roi_sharp, 0, 255).astype(np.uint8)

                    alpha = float(np.clip(0.75 * strength, 0.0, 0.9))
                    result[y1:y2, x1:x2] = cv2.addWeighted(roi_final, alpha, result[y1:y2, x1:x2], 1.0 - alpha, 0)

    return result


def _edge_shock_sharpen_luma(l_channel: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Sub-pixel morphological edge steepening (Shock Filter approximation).
    Contracts blurry transition ramps across edges toward the nearest local extremum (dilation/erosion midpoint),
    making upscaled lines, hair, eyes, and text look natively high-resolution rather than interpolated.
    """
    if strength <= 0.0:
        return l_channel

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(l_channel, kernel)
    eroded = cv2.erode(l_channel, kernel)
    midpoint = 0.5 * (dilated.astype(np.float32) + eroded.astype(np.float32))

    l_f = l_channel.astype(np.float32)
    # Push pixel toward dilation if above midpoint, or toward erosion if below midpoint, weighted by edge gradient
    gx = cv2.Sobel(l_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_mask = np.clip((grad - 8.0) / 55.0, 0.0, 1.0)

    shock_target = np.where(l_f >= midpoint, dilated.astype(np.float32), eroded.astype(np.float32))
    alpha = edge_mask * float(np.clip(strength, 0.0, 0.65))
    steepened = l_f * (1.0 - alpha) + shock_target * alpha
    return np.clip(steepened, 0, 255).astype(np.uint8)


def _upscale_stage_ibp(image: np.ndarray, target_w: int, target_h: int, iterations: int = 2) -> np.ndarray:
    """
    Single stage Super-Resolution using Lanczos-4 + Iterative Back-Projection (IBP) + Luma Edge Steepening.
    Reconstructs high-frequency residual lost during forward downsampling model.
    """
    orig_h, orig_w = image.shape[:2]
    upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    orig_f = image.astype(np.float32)

    # Iterative Back-Projection loop
    for _ in range(iterations):
        downprojected = cv2.resize(upscaled, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        residual = orig_f - downprojected
        up_residual = cv2.resize(residual, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        upscaled = np.clip(upscaled + up_residual * 0.65, 0, 255)

    res_u8 = upscaled.astype(np.uint8)
    # Apply subtle sub-pixel edge steepening on Luminance so edges are razor-crisp
    lab = cv2.cvtColor(res_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_steep = _edge_shock_sharpen_luma(l, strength=0.32)
    return cv2.cvtColor(cv2.merge([l_steep, a, b]), cv2.COLOR_LAB2BGR)


def apply_upscale(image: np.ndarray, scale_factor: float = 2.0, fast_mode: bool = False) -> np.ndarray:
    """
    Stage 3: Multi-Stage Progressive Super-Resolution (1x, 2x, 4x).
    For 4x upscaling, steps through 2x -> 4x with Iterative Back-Projection at each octave
    so edges never look soft or pixelated.
    """
    orig_h, orig_w = image.shape[:2]
    if scale_factor <= 1.0:
        # Even at 1x, apply sub-pixel luma edge steepening to de-blur soft original photos
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_steep = _edge_shock_sharpen_luma(l, strength=0.28)
        return cv2.cvtColor(cv2.merge([l_steep, a, b]), cv2.COLOR_LAB2BGR)

    final_w = max(2, int(round(orig_w * scale_factor)))
    final_h = max(2, int(round(orig_h * scale_factor)))
    final_w += final_w % 2
    final_h += final_h % 2

    if fast_mode:
        up = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _edge_shock_sharpen_luma(l, strength=0.25)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Progressive 2x octave stepping for 4x super-resolution
    if scale_factor >= 3.0:
        mid_w = orig_w * 2 + ((orig_w * 2) % 2)
        mid_h = orig_h * 2 + ((orig_h * 2) % 2)
        stage1 = _upscale_stage_ibp(image, mid_w, mid_h, iterations=2)
        return _upscale_stage_ibp(stage1, final_w, final_h, iterations=2)

    return _upscale_stage_ibp(image, final_w, final_h, iterations=2)


def enhance_multiscale_luminance_pyramid(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Stage 4: Overpowered 3-Band Laplacian/DoG Detail Pyramid in LAB Luminance Space.
    Simultaneously boosts:
    1. Ultra-Fine Micro-Textures (sigma=0.75: pores, hair strands, fabric, foliage)
    2. Crisp Structural Edges (sigma=1.8: facial features, text, outlines) with Tanh Anti-Halo Limiter
    3. Local HDR Clarity / Depth (sigma=5.5: 3D micro-contrast punch)
    """
    if amount <= 0.0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_f = l_chan.astype(np.float32)

    # 3 Gaussian scales
    blur_micro = cv2.GaussianBlur(l_f, (0, 0), sigmaX=0.75)
    blur_fine = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.80)
    blur_clarity = cv2.GaussianBlur(l_f, (0, 0), sigmaX=5.50)

    # Band-pass Difference of Gaussians (DoG)
    band_micro = l_f - blur_micro           # Highest frequency micro-detail
    band_fine = blur_micro - blur_fine      # Mid-high edge structure
    band_clarity = blur_fine - blur_clarity # Local contrast / 3D clarity punch

    # Noise gate on micro band so flat dark shadow noise isn't amplified
    micro_mag = np.abs(band_micro)
    noise_gate = np.clip((micro_mag - 1.2) / 4.0, 0.0, 1.0)
    band_micro_gated = band_micro * noise_gate

    # Soft-clipping Tanh limiter prevents bright/dark overshoot halos on high-contrast edges
    strength = float(np.clip(amount, 0.0, 1.5))
    boost_micro = 24.0 * np.tanh((band_micro_gated * (1.80 * strength)) / 24.0)
    boost_fine = 26.0 * np.tanh((band_fine * (1.45 * strength)) / 26.0)
    boost_clarity = 16.0 * np.tanh((band_clarity * (0.60 * strength)) / 16.0)

    l_enhanced = np.clip(l_f + boost_micro + boost_fine + boost_clarity, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def contrast_adaptive_sharpening(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Stage 5: Overpowered Contrast-Adaptive Sharpening (CAS Pro).
    High-power adaptive sharpening kernel (peak up to -0.36) that maximizes local acutance
    without clipping or halo artifacts.
    """
    if amount <= 0.0:
        return image

    img_f = image.astype(np.float32) / 255.0
    padded = cv2.copyMakeBorder(img_f, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # 3x3 cross + diagonal neighborhood
    a = padded[:-2, 1:-1]   # Top
    b = padded[1:-1, :-2]   # Left
    e = padded[1:-1, 1:-1]  # Center
    d = padded[1:-1, 2:]    # Right
    c = padded[2:, 1:-1]    # Bottom

    mn = np.minimum(np.minimum(np.minimum(a, b), np.minimum(c, d)), e)
    mx = np.maximum(np.maximum(np.maximum(a, b), np.maximum(c, d)), e)

    # Overpowered CAS peak weight (-0.34 at amount=1.0, vs old -0.125)
    peak = -0.34 * float(np.clip(amount, 0.0, 1.3))
    diff_max = 1.0 - mx
    diff_min = mn
    amp = np.sqrt(np.clip(np.minimum(diff_min, diff_max) / (mx + 1e-5), 0.0, 1.0))
    w = amp * peak

    num = e + w * (a + b + c + d)
    den = 1.0 + 4.0 * w
    sharpened = np.clip(num / np.maximum(den, 1e-4), 0.0, 1.0)
    return (sharpened * 255.0).astype(np.uint8)


def enhance_luminance_unsharp_mask(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Wrapper into the 3-Scale Laplacian Detail Pyramid for maximum Ultra-HD crispness.
    """
    return enhance_multiscale_luminance_pyramid(image, amount=amount)


def enhance_tone_vibrance(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    vibrance: float = 0.30,
    saturation_boost: float = 0.16,
) -> np.ndarray:
    """
    Stage 6: Auto Black-Point Dehaze, Local HDR CLAHE, S-Curve Dynamic Contrast & Skin-Safe Vibrance.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1. Auto Black-Point / White-Point Dehaze Stretch (only if dynamic range is compressed/hazy)
    p_low, p_high = np.percentile(l, (1.0, 99.2))
    if (p_low > 4.0 or p_high < 251.0) and (p_high - p_low > 20):
        l_stretched = np.clip((l.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        l = cv2.addWeighted(l_stretched, 0.40, l, 0.60, 0)

    # 2. Local HDR CLAHE on Luminance
    if clahe_clip > 0.0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        l = cv2.addWeighted(l_clahe, 0.58, l, 0.42, 0)

    # 3. Subtle Cinema S-Curve for deep blacks and crisp highlights
    l_norm = l.astype(np.float32) / 255.0
    s_curve = l_norm - 0.048 * np.sin(2.0 * np.pi * l_norm)
    l = np.clip(s_curve * 255.0, 0, 255).astype(np.uint8)

    bgr_toned = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Smart Non-Linear Vibrance in HSV space (protects skin tones from orange oversaturation)
    if vibrance > 0.0 or saturation_boost > 0.0:
        hsv = cv2.cvtColor(bgr_toned, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_c, s_c, v_c = cv2.split(hsv)
        s_norm = s_c / 255.0

        vibrance_gain = (1.0 - s_norm) * float(vibrance) * 1.15 + float(saturation_boost) * 0.85
        skin_mask_hsv = ((h_c <= 24) | (h_c >= 166)).astype(np.float32)
        vibrance_gain = vibrance_gain * (1.0 - 0.55 * skin_mask_hsv)

        s_boosted = np.clip(s_c * (1.0 + vibrance_gain), 0, 255)
        bgr_toned = cv2.cvtColor(cv2.merge([h_c, s_boosted, v_c]).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr_toned


def apply_vibrance_and_tone(
    image: np.ndarray,
    vibrance: float = 0.3,
    strength: Optional[float] = None,
) -> np.ndarray:
    """
    Compatibility wrapper used by video_enhancer.py and main.py.
    Supports both `vibrance` and legacy `strength` arguments.
    """
    eff = float(strength if strength is not None else vibrance)
    if eff <= 0.0:
        return image
    return enhance_tone_vibrance(
        image,
        clahe_clip=1.8,
        vibrance=eff,
        saturation_boost=0.12,
    )


def apply_cas_sharpening(
    image: np.ndarray,
    amount: float = 0.5,
    strength: Optional[float] = None,
) -> np.ndarray:
    """
    Compatibility wrapper combining Overpowered CAS Pro + 3-Scale Luminance Pyramid.
    Supports both `amount` and legacy `strength` arguments.
    """
    eff = float(strength if strength is not None else amount)
    if eff <= 0.0:
        return image
    cas_img = contrast_adaptive_sharpening(image, amount=eff * 1.15)
    return enhance_multiscale_luminance_pyramid(cas_img, amount=eff)


def enhance_image(
    input_path: str,
    output_path: str,
    scale_factor: float = 2.0,
    denoise_strength: float = 0.25,
    retouch_strength: float = 0.40,
    sharpen_amount: float = 0.55,
    vibrance_boost: float = 0.25,
    clahe_clip: float = 2.0,
    # Keyword aliases passed by app/main.py:
    upscale_factor: Optional[float] = None,
    face_retouch: Optional[float] = None,
    sharpen: Optional[float] = None,
    vibrance: Optional[float] = None,
    denoise: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full Wink-Style Overpowered 7-Stage Ultra-HD Image Enhancement Pipeline.
    """
    start_time = time.time()

    eff_scale = float(upscale_factor if upscale_factor is not None else scale_factor)
    eff_denoise = float(denoise if denoise is not None else denoise_strength)
    eff_retouch = float(face_retouch if face_retouch is not None else retouch_strength)
    eff_sharpen = float(sharpen if sharpen is not None else sharpen_amount)
    eff_vibrance = float(vibrance if vibrance is not None else vibrance_boost)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    orig_h, orig_w = img.shape[:2]
    orig_tenengrad = calculate_tenengrad(img)

    # Compute baseline upscaled for sharpness comparison
    target_w = max(2, int(round(orig_w * eff_scale)))
    target_h = max(2, int(round(orig_h * eff_scale)))
    if eff_scale > 1.0:
        target_w += target_w % 2
        target_h += target_h % 2
    baseline_upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    baseline_sharpness = calculate_tenengrad(baseline_upscaled)

    # Stage 1: Chroma-Isolated & Edge-Protected Denoising
    working = denoise_image(img, strength=eff_denoise)

    # Stage 2: 3-Band Frequency Separation Portrait Retouch + Eye/Lip Boost
    working = apply_portrait_retouch(working, strength=eff_retouch)

    # Stage 3 & 4: Iterative Back-Projection (IBP) Super-Resolution + Edge-Shock De-blurring
    working = apply_upscale(working, scale_factor=eff_scale, fast_mode=False)

    # Stage 5: Auto Dehaze, HDR CLAHE, Cinema S-Curve & Skin-Safe Vibrance
    working = enhance_tone_vibrance(
        working,
        clahe_clip=clahe_clip,
        vibrance=eff_vibrance,
        saturation_boost=0.14,
    )

    # Stage 6: Overpowered Contrast-Adaptive Sharpening (CAS Pro)
    working = contrast_adaptive_sharpening(working, amount=eff_sharpen * 1.05)

    # Stage 7: 3-Scale Laplacian Pyramid Detail & Local Clarity Booster (Micro + Fine + Clarity)
    working = enhance_multiscale_luminance_pyramid(working, amount=eff_sharpen * 0.95)

    # Save output image with maximum studio quality
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(output_path, working, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
    elif ext == ".png":
        cv2.imwrite(output_path, working, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
    elif ext == ".webp":
        cv2.imwrite(output_path, working, [int(cv2.IMWRITE_WEBP_QUALITY), 99])
    else:
        cv2.imwrite(output_path, working)

    duration = time.time() - start_time
    final_h, final_w = working.shape[:2]
    final_tenengrad = calculate_tenengrad(working)
    psnr_val = calculate_psnr(img, working)
    psnr_baseline = calculate_psnr(baseline_upscaled, working)
    ssim_val = calculate_ssim(img, working)
    tenengrad_gain = round(((final_tenengrad - orig_tenengrad) / max(orig_tenengrad, 1e-5)) * 100, 2)
    sharpness_gain = (final_tenengrad / baseline_sharpness) if baseline_sharpness > 0 else 1.0

    orig_file_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
    out_file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "status": "success",
        "input_path": input_path,
        "output_path": output_path,
        "original_resolution": f"{orig_w}x{orig_h}",
        "input_resolution": [orig_w, orig_h],
        "output_resolution": [final_w, final_h],
        "scale_factor": eff_scale,
        "original_size_bytes": orig_file_size,
        "output_size_bytes": out_file_size,
        "tenengrad_before": round(orig_tenengrad, 2),
        "tenengrad_after": round(final_tenengrad, 2),
        "tenengrad_improvement_pct": tenengrad_gain,
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "processing_time_sec": round(duration, 3),
        "processing_time_ms": round(duration * 1000.0, 2),
        "metrics": {
            "sharpness_gain": round(sharpness_gain, 4),
            "psnr_baseline": round(psnr_baseline, 2),
            "tenengrad_before": round(orig_tenengrad, 2),
            "tenengrad_after": round(final_tenengrad, 2),
            "tenengrad_gain": tenengrad_gain,
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
        },
    }
