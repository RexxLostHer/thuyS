"""
Unit Tests for Wink Image Enhancer Module
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from app.enhancer.image_enhancer import (
    enhance_image,
    calculate_tenengrad,
    calculate_psnr,
    calculate_ssim,
    segment_skin,
    denoise_image,
    apply_cas_sharpening,
    apply_vibrance_and_tone,
)


@pytest.fixture
def sample_test_image(tmp_path):
    """Create a temporary test image with color blocks, skin tone, and geometry."""
    img_path = tmp_path / "test_input.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Background
    img[:] = (30, 20, 40)
    # Skin tone rectangle (BGR: 175, 200, 240)
    img[16:48, 16:48] = (175, 200, 240)
    # High-contrast geometric feature
    img[24:32, 24:32] = (15, 15, 15)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


def test_enhance_image_2x(sample_test_image, tmp_path):
    out_path = str(tmp_path / "out_2x.png")
    res = enhance_image(
        input_path=sample_test_image,
        output_path=out_path,
        upscale_factor=2,
        face_retouch=0.5,
        sharpen=0.5,
        vibrance=0.3,
        denoise=0.2,
    )
    assert res["status"] == "success"
    assert res["input_resolution"] == [64, 64]
    assert res["output_resolution"] == [128, 128]
    assert os.path.exists(out_path)
    
    out_img = cv2.imread(out_path)
    assert out_img.shape == (128, 128, 3)
    assert res["metrics"]["sharpness_gain"] > 1.0
    assert res["metrics"]["psnr_baseline"] > 20.0


def test_enhance_image_4x(sample_test_image, tmp_path):
    out_path = str(tmp_path / "out_4x.png")
    res = enhance_image(
        input_path=sample_test_image,
        output_path=out_path,
        upscale_factor=4,
        face_retouch=0.4,
        sharpen=0.6,
        vibrance=0.2,
        denoise=0.2,
    )
    assert res["status"] == "success"
    assert res["input_resolution"] == [64, 64]
    assert res["output_resolution"] == [256, 256]
    assert os.path.exists(out_path)
    out_img = cv2.imread(out_path)
    assert out_img.shape == (256, 256, 3)


def test_enhance_image_1x_no_upscale(sample_test_image, tmp_path):
    out_path = str(tmp_path / "out_1x.png")
    res = enhance_image(
        input_path=sample_test_image,
        output_path=out_path,
        upscale_factor=1,
        face_retouch=0.3,
        sharpen=0.4,
        vibrance=0.2,
        denoise=0.1,
    )
    assert res["status"] == "success"
    assert res["input_resolution"] == [64, 64]
    assert res["output_resolution"] == [64, 64]
    assert os.path.exists(out_path)


def test_enhance_image_missing_file():
    with pytest.raises(FileNotFoundError):
        enhance_image(
            input_path="non_existent_image_12345.png",
            output_path="out.png"
        )


def test_metrics_identical_images():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    psnr_val = calculate_psnr(img, img)
    ssim_val = calculate_ssim(img, img)
    assert psnr_val >= 99.0  # Perfect reconstruction
    assert ssim_val >= 0.999


def test_metrics_different_images():
    img1 = np.full((50, 50, 3), 50, dtype=np.uint8)
    img2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    psnr_val = calculate_psnr(img1, img2)
    assert psnr_val < 20.0
    assert calculate_ssim(img1, img2) < 0.5


def test_skin_segmentation():
    # Patch of pure skin tone
    skin_patch = np.zeros((20, 20, 3), dtype=np.uint8)
    skin_patch[:] = (175, 200, 240)
    mask = segment_skin(skin_patch)
    assert mask.shape == (20, 20)
    assert np.mean(mask) > 0.5  # Should be recognized as skin
    
    # Patch of pure blue
    blue_patch = np.zeros((20, 20, 3), dtype=np.uint8)
    blue_patch[:] = (255, 0, 0)
    mask_blue = segment_skin(blue_patch)
    assert np.mean(mask_blue) < 0.05  # Should NOT be recognized as skin
