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


def denoise_image(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply edge-preserving dual-domain bilateral denoising.
    """
    if strength <= 0.0:
        return image
    
    # Scale parameters by strength (0.0 - 1.0)
    d = int(np.clip(5 + 6 * strength, 3, 11))
    sigma_color = int(np.clip(25 + 50 * strength, 10, 85))
    sigma_space = int(np.clip(15 + 40 * strength, 10, 60))
    
    denoised = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Blend according to strength
    result = cv2.addWeighted(denoised, float(strength), image, float(1.0 - strength), 0)
    return result


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
    1. Detect faces or segment skin regions
    2. Apply dual-domain bilateral skin smoothing with 70/30 natural pore texture blending
    3. Detect eyes and lips to apply localized contrast and sharpness boost
    """
    if strength <= 0.0:
        return image
        
    h, w = image.shape[:2]
    result = image.copy()
    
    # Detect face bounding boxes if possible
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_boxes = []
    if os.path.exists(cascade_path):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        face_boxes = list(faces)
        
    # Get skin mask
    skin_mask = segment_skin(image)
    
    # If face boxes exist, boost mask priority within face regions
    if face_boxes:
        face_mask = np.zeros((h, w), dtype=np.float32)
        for (fx, fy, fw, fh) in face_boxes:
            # Elliptical face region
            center = (fx + fw // 2, fy + fh // 2)
            axes = (int(fw * 0.45), int(fh * 0.55))
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (15, 15), 5.0)
        # Combine: prioritize face region skin
        skin_mask = np.clip(skin_mask * 0.6 + (skin_mask * face_mask) * 0.4, 0.0, 1.0)
        
    # Bilateral smoothing for skin
    smoothed_skin = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
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
                (ey_start, ey_end, ex_start, ex_end, 1.25, 1.15),  # Eyes
                (ly_start, ly_end, lx_start, lx_end, 1.15, 1.12),  # Lips
            ]:
                if y2 > y1 and x2 > x1 and y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w:
                    roi = result[y1:y2, x1:x2].astype(np.float32)
                    
                    # Contrast boost around mean
                    roi_mean = np.mean(roi, axis=(0, 1), keepdims=True)
                    contrast_factor = 1.0 + (boost_contrast - 1.0) * float(strength)
                    roi_contrasted = np.clip((roi - roi_mean) * contrast_factor + roi_mean, 0, 255)
                    
                    # Crisp unsharp mask for eyes/lips
                    roi_blur = cv2.GaussianBlur(roi_contrasted, (3, 3), 1.0)
                    sharp_factor = (boost_sharp - 1.0) * float(strength) * 1.5
                    roi_enhanced = np.clip(roi_contrasted + sharp_factor * (roi_contrasted - roi_blur), 0, 255)
                    
                    # Feathered blend for ROI
                    roi_mask = np.zeros((y2 - y1, x2 - x1, 1), dtype=np.float32)
                    cv2.ellipse(
                        roi_mask,
                        ((x2 - x1) // 2, (y2 - y1) // 2),
                        (int((x2 - x1) * 0.45), int((y2 - y1) * 0.45)),
                        0, 0, 360, 1.0, -1
                    )
                    roi_mask = cv2.GaussianBlur(roi_mask, (7, 7), 2.0)[:, :, np.newaxis] if roi_mask.ndim == 2 else cv2.GaussianBlur(roi_mask, (7, 7), 2.0)
                    if roi_mask.ndim == 2:
                        roi_mask = roi_mask[:, :, np.newaxis]
                        
                    result[y1:y2, x1:x2] = (roi_enhanced * roi_mask + result[y1:y2, x1:x2] * (1.0 - roi_mask)).astype(np.uint8)
                    
    return result


def apply_upscale(image: np.ndarray, upscale_factor: int = 2) -> np.ndarray:
    """
    High-grade multi-scale Lanczos-4 upscaling with gradient edge reinforcement.
    """
    if upscale_factor <= 1:
        return image
        
    h, w = image.shape[:2]
    target_w = w * upscale_factor
    target_h = h * upscale_factor
    
    if upscale_factor == 2:
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    elif upscale_factor == 4:
        # 2-stage multi-scale Lanczos-4 upscaling
        mid_w, mid_h = w * 2, h * 2
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
        
        upscaled = cv2.resize(mid_refined, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Arbitrary factor fallback
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
    return upscaled


def apply_vibrance_and_tone(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Dynamic tone and vibrance boost:
    1. LAB CLAHE on L-channel to expand dynamic range without chromatic shifting
    2. Intelligent non-linear saturation curve: boosts muted tones while protecting skin/vibrant areas
    """
    if strength <= 0.0:
        return image
        
    # 1. LAB CLAHE on luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clip_limit = 1.0 + 2.5 * float(strength)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Recombine LAB
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_tone = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 2. Intelligent non-linear vibrance curve in HSV
    hsv = cv2.cvtColor(bgr_tone, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Normalize saturation to [0, 1]
    s_norm = s / 255.0
    
    # Non-linear boost: delta_s = 1.0 - s_norm
    # Muted pixels (low s_norm) get maximum relative boost; saturated pixels stay protected
    delta_s = 1.0 - s_norm
    s_boosted = s_norm + (float(strength) * 0.55 * delta_s * s_norm)
    s_new = np.clip(s_boosted * 255.0, 0, 255).astype(np.float32)
    
    hsv_boosted = cv2.merge([h, s_new, v])
    result = cv2.cvtColor(hsv_boosted.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Blend with original according to strength
    blended = cv2.addWeighted(result, float(strength), image, float(1.0 - strength), 0)
    return blended


def apply_cas_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Contrast-Adaptive Sharpening (CAS) and dual-frequency luminance unsharp masking.
    Operates strictly on the L-channel in LAB space to eliminate color fringing and halo artifacts.
    Provides crisp edge definition and micro-texture recovery.
    """
    if strength <= 0.0:
        return image
        
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    
    # Dual-frequency high-pass bands for fine details and structural contours
    blur_fine = cv2.GaussianBlur(l_float, (0, 0), 1.0)
    blur_med = cv2.GaussianBlur(l_float, (0, 0), 2.5)
    hp_fine = l_float - blur_fine
    hp_med = l_float - blur_med
    hp = 0.65 * hp_fine + 0.85 * hp_med
    
    # 3x3 local contrast window for adaptive control
    kernel = np.ones((3, 3), np.uint8)
    l_min = cv2.erode(l, kernel).astype(np.float32)
    l_max = cv2.dilate(l, kernel).astype(np.float32)
    contrast = l_max - l_min
    adapt_weight = np.clip(1.0 + (contrast / 128.0), 0.8, 1.5)
    
    # Adaptive sharpening gain
    sharpen_gain = 0.8 + 2.0 * float(strength)
    l_sharp = np.clip(l_float + sharpen_gain * adapt_weight * hp, 0, 255).astype(np.uint8)
    
    lab_sharp = cv2.merge([l_sharp, a, b])
    result = cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
    return result


def enhance_image(
    input_path: str,
    output_path: str,
    upscale_factor: int = 2,
    face_retouch: float = 0.5,
    sharpen: float = 0.5,
    vibrance: float = 0.3,
    denoise: float = 0.3,
) -> Dict[str, Any]:
    """
    Core Image Enhancement Function.
    Adheres strictly to the PROJECT.md interface contract:
    
    Args:
        input_path: Path to input image file (PNG, JPG, WEBP, etc.)
        output_path: Path to save enhanced output image
        upscale_factor: Upscaling multiplier (1, 2, or 4)
        face_retouch: Portrait retouching strength (0.0 to 1.0)
        sharpen: CAS / adaptive unsharp sharpening strength (0.0 to 1.0)
        vibrance: LAB CLAHE and saturation boost strength (0.0 to 1.0)
        denoise: Dual-domain bilateral denoising strength (0.0 to 1.0)
        
    Returns:
        dict matching contract specification:
        {
            "status": "success",
            "input_resolution": [w, h],
            "output_resolution": [w * upscale_factor, h * upscale_factor],
            "processing_time_ms": float,
            "metrics": {"sharpness_gain": float, "psnr_baseline": float}
        }
    """
    start_time = time.perf_counter()
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
        
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at: {input_path}")
        
    h_in, w_in = img.shape[:2]
    
    # Baseline comparison image (standard Lanczos-4 upscaled without enhancements)
    target_w = w_in * upscale_factor
    target_h = h_in * upscale_factor
    baseline_upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    baseline_sharpness = calculate_tenengrad(baseline_upscaled)
    
    # Step 1: Denoise filter
    current = denoise_image(img, strength=denoise)
    
    # Step 2: Portrait Retouching & Natural Skin Smoothing (with pore preservation)
    current = apply_portrait_retouch(current, strength=face_retouch)
    
    # Step 3: High-grade Multi-Scale Lanczos-4 Super-Resolution
    current = apply_upscale(current, upscale_factor=upscale_factor)
    
    # Step 4: Dynamic Tone & Vibrance Boost
    current = apply_vibrance_and_tone(current, strength=vibrance)
    
    # Step 5: Deblur & Contrast-Adaptive Sharpening (CAS)
    current = apply_cas_sharpening(current, strength=sharpen)
    
    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    success = cv2.imwrite(output_path, current)
    if not success:
        # Fallback using PIL
        pil_img = Image.fromarray(cv2.cvtColor(current, cv2.COLOR_BGR2RGB))
        pil_img.save(output_path)
        
    processing_time_ms = (time.perf_counter() - start_time) * 1000.0
    
    # Calculate quality metrics
    enhanced_sharpness = calculate_tenengrad(current)
    sharpness_gain = (
        enhanced_sharpness / baseline_sharpness
        if baseline_sharpness > 0 else 1.0
    )
    psnr_baseline = calculate_psnr(baseline_upscaled, current)
    
    return {
        "status": "success",
        "input_resolution": [w_in, h_in],
        "output_resolution": [target_w, target_h],
        "processing_time_ms": round(processing_time_ms, 2),
        "metrics": {
            "sharpness_gain": round(sharpness_gain, 4),
            "psnr_baseline": round(psnr_baseline, 2),
        },
    }
