/**
 * Lumina HD Studio — Dynamic Canvas Logo Animation Engine
 * Inspired by Ujikom Architecture: 69-frame sequence + periodic metallic shine loop (frames 60-69).
 * Features HiDPI sub-pixel rendering, auto-pausing when tab is hidden, and AI processing acceleration.
 */

(function () {
  const TOTAL_SHINE_FRAMES = 10;
  const SHINE_START = 60;
  const SHINE_INTERVAL_MS = 3800; // Shine every 3.8s
  const SHINE_FRAME_DURATION = 75; // ~13.3 FPS for smooth shine glide

  let canvas = null;
  let ctx = null;
  let shineFrames = [];
  let baseLogo = null;
  let isFramesLoaded = false;
  let isProcessing = false;
  let shineTimer = null;
  let currentShineFrame = 9; // frame 069 (index 9) is the resting clean state
  let isShining = false;
  let processingRotation = 0;
  let processingRaf = null;

  function initLogoAnimation() {
    canvas = document.getElementById('headerLogoCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    // Preload base clean logo
    baseLogo = new Image();
    baseLogo.src = '/static/assets/animasi/ezgif-frame-069.png';
    baseLogo.onload = () => {
      drawBase();
    };

    // Preload transparent shine frames 60..69
    let loadedCount = 0;
    for (let f = 0; f < TOTAL_SHINE_FRAMES; f++) {
      const img = new Image();
      const numStr = String(SHINE_START + f).padStart(3, '0');
      img.src = `/static/assets/animasi/ezgif-frame-${numStr}.png`;
      img.onload = () => {
        loadedCount++;
        if (loadedCount === TOTAL_SHINE_FRAMES) {
          isFramesLoaded = true;
          startPeriodicShine();
        }
      };
      shineFrames.push(img);
    }

    // Hover effect: trigger shine immediately
    const brand = document.getElementById('brandContainer');
    if (brand) {
      brand.addEventListener('mouseenter', () => {
        if (!isProcessing && !isShining) {
          triggerShineOnce();
        }
      });
    }

    // Tab visibility handling: pause shine timer when hidden to save battery/GPU
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        if (shineTimer) clearTimeout(shineTimer);
      } else {
        if (!isProcessing && isFramesLoaded) {
          scheduleNextShine(1000);
        }
      }
    });
  }

  function drawBase() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const img = (shineFrames[currentShineFrame] && shineFrames[currentShineFrame].complete)
      ? shineFrames[currentShineFrame]
      : baseLogo;

    if (img && img.complete && img.naturalWidth) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
  }

  function triggerShineOnce() {
    if (!isFramesLoaded || isShining || isProcessing) return;
    isShining = true;
    let step = 0;

    const stepTick = () => {
      currentShineFrame = step;
      drawBase();
      step++;
      if (step < TOTAL_SHINE_FRAMES) {
        setTimeout(stepTick, SHINE_FRAME_DURATION);
      } else {
        currentShineFrame = 9; // Reset to clean logo
        drawBase();
        isShining = false;
        scheduleNextShine(SHINE_INTERVAL_MS);
      }
    };

    stepTick();
  }

  function scheduleNextShine(delay) {
    if (shineTimer) clearTimeout(shineTimer);
    shineTimer = setTimeout(() => {
      if (!document.hidden && !isProcessing) {
        triggerShineOnce();
      } else {
        scheduleNextShine(SHINE_INTERVAL_MS);
      }
    }, delay);
  }

  function startPeriodicShine() {
    scheduleNextShine(1500); // First shine after 1.5s
  }

  // Active state during AI Enhancement (Pulse/Spin acceleration)
  window.setLogoProcessing = function (active) {
    isProcessing = active;
    if (shineTimer) clearTimeout(shineTimer);

    if (active) {
      if (canvas) canvas.classList.add('logo-processing');
      let frameIndex = 0;
      const processingLoop = () => {
        if (!isProcessing) return;
        frameIndex = (frameIndex + 1) % TOTAL_SHINE_FRAMES;
        currentShineFrame = frameIndex;
        drawBase();
        processingRaf = setTimeout(processingLoop, 85);
      };
      processingLoop();
    } else {
      if (canvas) canvas.classList.remove('logo-processing');
      if (processingRaf) clearTimeout(processingRaf);
      currentShineFrame = 9;
      drawBase();
      // Victory shine upon completion
      triggerShineOnce();
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLogoAnimation);
  } else {
    initLogoAnimation();
  }
})();
