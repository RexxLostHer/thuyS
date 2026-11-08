/**
 * Wink HD Enhancer — Interactive Split Comparison Slider & Dual-Video Sync Engine
 * Refactored strictly adhering to PRD Remediasi & Refactoring Menyeluruh (Fase 2 & Fase 4)
 * Implements loadVideoSourceReady Promise cycle, requestVideoFrameCallback 60fps render loop,
 * and isolated Canvas 2D split rendering without Multiplane Overlay black screens.
 */

class ComparisonSlider {
  constructor(options = {}) {
    this.container = options.container || document.getElementById('comparisonContainer');
    this.canvas = options.canvas || document.getElementById('comparisonCanvas');
    this.ctx = this.canvas ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
    this.layerBefore = options.layerBefore || document.getElementById('layerBefore');
    this.layerAfter = options.layerAfter || document.getElementById('layerAfter');
    this.divider = options.divider || document.getElementById('sliderDivider');
    this.handle = options.handle || document.getElementById('sliderHandle');
    this.videoControls = options.videoControls || document.getElementById('videoControlsBar');

    // Controls elements
    this.btnPlayPause = document.getElementById('btnPlayPause');
    this.videoScrubber = document.getElementById('videoScrubber');
    this.timeDisplay = document.getElementById('timeDisplay');
    this.btnMute = document.getElementById('btnMute');

    this.position = 50; // percentage
    this.isDragging = false;
    this.mode = 'image'; // 'image' or 'video'

    // Video synchronization state
    this.masterVideo = null;
    this.slaveVideo = null;
    this.syncRafId = null;
    this.videoCallbackId = null;
    this.isLoopActive = false;
    this.syncThreshold = 0.25; // 250ms drift threshold (PRD Section 2.5: prevents seek thrashing)

    // Zoom and Loupe Inspection Engine
    this.zoomLevel = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;
    this.loupeActive = false;
    this.beforeUrl = null;
    this.afterUrl = null;

    this.btnZoomFit = document.getElementById('btnZoomFit');
    this.btnZoom100 = document.getElementById('btnZoom100');
    this.btnZoom200 = document.getElementById('btnZoom200');
    this.btnLoupeToggle = document.getElementById('btnLoupeToggle');
    this.magnifierLoupe = document.getElementById('magnifierLoupe');
    this.loupeInnerBefore = document.getElementById('loupeInnerBefore');
    this.loupeInnerAfter = document.getElementById('loupeInnerAfter');
    this.loupeDivider = document.getElementById('loupeDivider');

    this.initSliderEvents();
    this.initVideoControlEvents();
    this.initZoomAndLoupeEvents();
    this.updatePosition(50);
  }

  initSliderEvents() {
    const onStart = (e) => {
      this.isDragging = true;
      this.handleDrag(e);
      document.body.style.cursor = 'ew-resize';
      e.preventDefault();
    };

    const onMove = (e) => {
      if (!this.isDragging) return;
      if (e.cancelable && e.touches) {
        e.preventDefault();
      }
      this.handleDrag(e);
    };

    const onEnd = () => {
      if (this.isDragging) {
        this.isDragging = false;
        document.body.style.cursor = 'default';
      }
    };

    // Mouse events
    this.container.addEventListener('mousedown', onStart);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onEnd);

    // Touch events for mobile phones (preventing scroll thrashing during comparison)
    this.container.addEventListener('touchstart', onStart, { passive: false });
    this.container.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onEnd);
  }

  handleDrag(e) {
    const rect = this.container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const offsetX = clientX - rect.left;
    let percent = (offsetX / rect.width) * 100;
    percent = Math.max(0, Math.min(100, percent));
    this.updatePosition(percent);
  }

  updatePosition(percent) {
    this.position = percent;
    this.container.style.setProperty('--slider-pos', `${percent}%`);
    if (this.mode === 'video') {
      this.drawVideoFrames();
    }
  }

  /**
   * Set up Image Comparison Mode (CSS clip-path)
   */
  setImageComparison(beforeUrl, afterUrl) {
    this.cleanupVideoSync();
    this.mode = 'image';
    this.beforeUrl = beforeUrl;
    this.afterUrl = afterUrl;
    this.videoControls.style.display = 'none';

    if (this.canvas) {
      this.canvas.style.display = 'none';
    }
    this.layerBefore.style.display = 'flex';
    this.layerAfter.style.display = 'flex';

    this.layerBefore.innerHTML = `<img src="${beforeUrl}" class="comparison-media" alt="Original Low-Res">`;
    this.layerAfter.innerHTML = `<img src="${afterUrl}" class="comparison-media" alt="Lumina HD Enhanced">`;

    this.container.style.display = 'block';
    this.resetZoom();
    this.updatePosition(50);
    this.updateLoupeAssets();
  }

  /**
   * PRD Seksi 4.1: Memuat file video ke dalam HTMLVideoElement hingga frame pertama benar-benar siap digambar ke Canvas.
   * Mencegah bug canvas 0x0 dan frame hitam pada saat inisialisasi.
   */
  loadVideoSourceReady(videoElement, fileOrUrl) {
    return new Promise((resolve, reject) => {
      // 1. Bersihkan ObjectURL lama jika ada
      if (videoElement.dataset.objectUrl) {
        URL.revokeObjectURL(videoElement.dataset.objectUrl);
        delete videoElement.dataset.objectUrl;
      }

      // 2. Set atribut sebelum src agar lolos autoplay policy & decoding pipeline
      videoElement.muted = true;
      videoElement.playsInline = true;
      videoElement.preload = 'auto';
      videoElement.crossOrigin = 'anonymous';

      const srcUrl = typeof fileOrUrl === 'string'
        ? fileOrUrl
        : URL.createObjectURL(fileOrUrl);

      if (typeof fileOrUrl !== 'string') {
        videoElement.dataset.objectUrl = srcUrl;
      }

      let isResolved = false;
      let hasTriggeredSeek = false;
      const cleanupListeners = () => {
        videoElement.removeEventListener('loadeddata', onReady);
        videoElement.removeEventListener('canplay', onReady);
        videoElement.removeEventListener('seeked', onSeeked);
        videoElement.removeEventListener('error', onError);
      };

      const finishResolve = () => {
        if (isResolved) return;
        isResolved = true;
        cleanupListeners();
        resolve({
          width: videoElement.videoWidth || 1920,
          height: videoElement.videoHeight || 1080,
          duration: videoElement.duration || 0,
        });
      };

      const onError = () => {
        if (isResolved) return;
        isResolved = true;
        cleanupListeners();
        const errCode = videoElement.error ? videoElement.error.code : 'unknown';
        reject(new Error(`Failed to decode video (Error code: ${errCode}).`));
      };

      const onSeeked = () => {
        finishResolve();
      };

      const onReady = () => {
        if (hasTriggeredSeek) return;
        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
          return;
        }
        hasTriggeredSeek = true;
        // Force seek to frame 1 (0.05s) to push frame to GPU buffer
        try {
          videoElement.currentTime = 0.05;
        } catch (e) {
          finishResolve();
        }
        // Fallback timer in case seeked doesn't fire
        setTimeout(() => {
          if (videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
            finishResolve();
          }
        }, 200);
      };

      videoElement.addEventListener('loadeddata', onReady);
      videoElement.addEventListener('canplay', onReady);
      videoElement.addEventListener('seeked', onSeeked);
      videoElement.addEventListener('error', onError, { once: true });

      // Immediate check if already ready
      if (videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
        onReady();
      }

      videoElement.src = srcUrl;
      videoElement.load();

      // Absolute safety timeout of 4 seconds so UI NEVER hangs
      setTimeout(() => {
        if (!isResolved) {
          if (videoElement.videoWidth > 0) {
            finishResolve();
          } else {
            onError();
          }
        }
      }, 4000);
    });
  }

  /**
   * Set up Synchronized Dual-Video Comparison Mode (Canvas 2D Engine)
   */
  async setVideoComparison(beforeUrl, afterUrl) {
    this.cleanupVideoSync();
    this.mode = 'video';
    this.videoControls.style.display = 'flex';

    // Sembunyikan layer gambar DOM; tampilkan canvas tunggal anti-overlay clash
    this.layerBefore.style.display = 'none';
    this.layerAfter.style.display = 'none';
    this.layerBefore.innerHTML = '';
    this.layerAfter.innerHTML = '';

    if (this.canvas) {
      this.canvas.style.display = 'block';
      if (!this.ctx) {
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
      }
    }

    const master = document.createElement('video');
    const slave = document.createElement('video');

    // DOM Mount aktif dengan dimensi minimum (agar D3D11 / DirectComposition GPU video decoder aktif)
    let mount = document.getElementById('videoMediaMount');
    if (!mount) {
      mount = document.createElement('div');
      mount.id = 'videoMediaMount';
      document.body.appendChild(mount);
    }
    mount.style.position = 'fixed';
    mount.style.bottom = '0';
    mount.style.right = '0';
    mount.style.width = '4px';
    mount.style.height = '4px';
    mount.style.opacity = '0.01';
    mount.style.pointerEvents = 'none';
    mount.style.zIndex = '-999';
    mount.style.overflow = 'hidden';
    mount.innerHTML = '';
    slave.style.width = '4px';
    slave.style.height = '4px';
    slave.style.display = 'block';
    master.style.width = '4px';
    master.style.height = '4px';
    master.style.display = 'block';
    mount.appendChild(slave);
    mount.appendChild(master);

    this.masterVideo = master;
    this.slaveVideo = slave;

    this.container.style.display = 'block';
    this.updateMuteButton(true);

    try {
      // Tunggu kedua video tervalidasi frame pertama (PRD Seksi 4.1)
      const [slaveMeta, masterMeta] = await Promise.all([
        this.loadVideoSourceReady(slave, beforeUrl),
        this.loadVideoSourceReady(master, afterUrl),
      ]);

      const w = masterMeta.width || slaveMeta.width || 1920;
      const h = masterMeta.height || slaveMeta.height || 1080;

      if (this.canvas) {
        this.canvas.width = w;
        this.canvas.height = h;
        if (this.ctx) {
          this.ctx.imageSmoothingEnabled = true;
          this.ctx.imageSmoothingQuality = 'high';
        }
      }

      this.timeDisplay.textContent = `00:00 / ${this.formatTime(masterMeta.duration)}`;

      // Render frame pertama yang sudah terbukti siap
      this.drawVideoFrames();

      // Pasang sinkronisasi loop per-frame dan kontrol playback (PRD Seksi 4.3)
      this.bindVideoSync(master, slave);

    } catch (err) {
      console.error('Video comparison initialization failed:', err);
    }
  }

  /**
   * PRD Seksi 4.2 & 4.3: Menggambar frame kedua video ke Canvas dengan split perbandingan
   */
  drawVideoFrames() {
    if (!this.canvas || !this.ctx || !this.masterVideo || !this.slaveVideo) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    if (!w || !h) return;

    const splitX = Math.round(w * (this.position / 100));

    const slaveReady = this.slaveVideo.videoWidth > 0 && this.slaveVideo.readyState >= 2;
    const masterReady = this.masterVideo.videoWidth > 0 && this.masterVideo.readyState >= 2;

    // PRD Section 2.5: Do not clear canvas if videos are seeking (readyState < 2),
    // preserving the last valid frame on screen without flickering black
    if (!slaveReady && !masterReady) {
      return;
    }

    if (slaveReady && masterReady) {
      this.ctx.clearRect(0, 0, w, h);
    }

    // 1. Sisi Kiri: Original Low-Res (Slave)
    if (splitX > 0 && slaveReady) {
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.rect(0, 0, splitX, h);
      this.ctx.clip();
      try {
        this.ctx.drawImage(this.slaveVideo, 0, 0, w, h);
      } catch (e) {
        console.warn('drawVideoFrames slave error:', e);
      }
      this.ctx.restore();
    }

    // 2. Sisi Kanan: Wink HD Enhanced (Master)
    if (splitX < w && masterReady) {
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.rect(splitX, 0, w - splitX, h);
      this.ctx.clip();
      try {
        this.ctx.drawImage(this.masterVideo, 0, 0, w, h);
      } catch (e) {
        console.warn('drawVideoFrames master error:', e);
      }
      this.ctx.restore();
    }
  }

  /**
   * PRD Seksi 4.3: Loop Sinkronisasi Frame Real-Time (requestVideoFrameCallback)
   */
  bindVideoSync(master, slave) {
    const loopTick = () => {
      if (!this.isLoopActive || master.paused || master.ended) {
        this.isLoopActive = false;
        return;
      }
      this.drawVideoFrames();
      if ('requestVideoFrameCallback' in master) {
        this.videoCallbackId = master.requestVideoFrameCallback(loopTick);
      } else {
        this.syncRafId = window.requestAnimationFrame(loopTick);
      }
    };

    const startLoop = () => {
      if (this.isLoopActive) return;
      this.isLoopActive = true;
      loopTick();
    };

    const stopLoop = () => {
      this.isLoopActive = false;
      if (this.videoCallbackId !== null && 'cancelVideoFrameCallback' in master) {
        master.cancelVideoFrameCallback(this.videoCallbackId);
        this.videoCallbackId = null;
      }
      if (this.syncRafId !== null) {
        window.cancelAnimationFrame(this.syncRafId);
        this.syncRafId = null;
      }
      this.drawVideoFrames();
    };

    // Playback State Sync
    master.addEventListener('play', () => {
      slave.play().catch(() => {});
      this.updatePlayPauseButton(true);
      startLoop();
    });

    master.addEventListener('pause', () => {
      slave.pause();
      this.updatePlayPauseButton(false);
      stopLoop();
    });

    master.addEventListener('ended', () => {
      slave.pause();
      this.updatePlayPauseButton(false);
      stopLoop();
    });

    master.addEventListener('playing', () => {
      slave.play().catch(() => {});
      this.updatePlayPauseButton(true);
      startLoop();
    });

    master.addEventListener('waiting', () => {
      slave.pause();
    });

    // Seeking Synchronization (PRD: seeked redraws frame)
    master.addEventListener('seeked', () => {
      if (Math.abs(slave.currentTime - master.currentTime) > 0.02) {
        slave.currentTime = master.currentTime;
      }
      this.drawVideoFrames();
    });

    slave.addEventListener('seeked', () => {
      this.drawVideoFrames();
    });

    // Rate change sync
    master.addEventListener('ratechange', () => {
      slave.playbackRate = master.playbackRate;
    });

    // Time scrubber update & continuous drift correction
    master.addEventListener('timeupdate', () => {
      if (!master || isNaN(master.duration)) return;
      const cur = master.currentTime;
      const dur = master.duration;
      const pct = (cur / dur) * 100;
      this.videoScrubber.value = isNaN(pct) ? 0 : pct;
      this.timeDisplay.textContent = `${this.formatTime(cur)} / ${this.formatTime(dur)}`;

      if (slave && Math.abs(master.currentTime - slave.currentTime) > this.syncThreshold) {
        slave.currentTime = master.currentTime;
      }
      this.drawVideoFrames();
    });

    // Auto play master on load (muted = allowed by browser autoplay policy)
    master.play()
      .then(() => {
        slave.play().catch(() => {});
        this.updatePlayPauseButton(true);
        startLoop();
      })
      .catch(() => {
        this.updatePlayPauseButton(false);
        this.drawVideoFrames();
      });
  }

  initVideoControlEvents() {
    if (!this.btnPlayPause) return;

    this.btnPlayPause.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!this.masterVideo) return;
      if (this.masterVideo.paused) {
        this.masterVideo.play().catch(() => {});
        if (this.slaveVideo) this.slaveVideo.play().catch(() => {});
      } else {
        this.masterVideo.pause();
        if (this.slaveVideo) this.slaveVideo.pause();
      }
    });

    this.videoScrubber.addEventListener('input', (e) => {
      if (!this.masterVideo || isNaN(this.masterVideo.duration)) return;
      const targetTime = (e.target.value / 100) * this.masterVideo.duration;
      this.masterVideo.currentTime = targetTime;
      if (this.slaveVideo) {
        this.slaveVideo.currentTime = targetTime;
      }
      this.drawVideoFrames();
    });

    if (this.btnMute) {
      this.btnMute.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.masterVideo) return;
        this.masterVideo.muted = !this.masterVideo.muted;
        this.updateMuteButton(this.masterVideo.muted);
      });
    }
  }

  updateMuteButton(isMuted) {
    if (!this.btnMute) return;
    this.btnMute.innerHTML = isMuted
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`;
  }

  updatePlayPauseButton(isPlaying) {
    if (!this.btnPlayPause) return;
    this.btnPlayPause.innerHTML = isPlaying
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  }

  /**
   * Interactive Zoom, Pan, and Magnifier Loupe Engine
   */
  initZoomAndLoupeEvents() {
    if (this.btnZoomFit) {
      this.btnZoomFit.addEventListener('click', (e) => {
        e.stopPropagation();
        this.setZoom(1.0);
      });
    }
    if (this.btnZoom100) {
      this.btnZoom100.addEventListener('click', (e) => {
        e.stopPropagation();
        this.setZoom(2.0);
      });
    }
    if (this.btnZoom200) {
      this.btnZoom200.addEventListener('click', (e) => {
        e.stopPropagation();
        this.setZoom(3.5);
      });
    }
    if (this.btnLoupeToggle) {
      this.btnLoupeToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleLoupe();
      });
    }

    // Wheel Zoom
    this.container.addEventListener('wheel', (e) => {
      if (this.mode !== 'image') return;
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.35 : -0.35;
      const targetZoom = Math.max(1.0, Math.min(5.0, this.zoomLevel + delta));
      this.setZoom(targetZoom);
    }, { passive: false });

    // Panning drag handlers when zoomed in
    const onPanStart = (e) => {
      if (this.zoomLevel <= 1.0) return;
      // Do not pan if dragging slider divider
      if (e.target.closest('#sliderHandle') || e.target.closest('#comparisonToolbar')) return;
      this.isPanning = true;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      this.panStartX = clientX - this.panX;
      this.panStartY = clientY - this.panY;
      this.container.classList.add('is-panning');
    };

    const onPanMove = (e) => {
      if (!this.isPanning) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      this.panX = clientX - this.panStartX;
      this.panY = clientY - this.panStartY;
      this.applyTransform();
    };

    const onPanEnd = () => {
      if (this.isPanning) {
        this.isPanning = false;
        this.container.classList.remove('is-panning');
      }
    };

    this.container.addEventListener('mousedown', onPanStart);
    window.addEventListener('mousemove', onPanMove);
    window.addEventListener('mouseup', onPanEnd);

    this.container.addEventListener('touchstart', onPanStart, { passive: true });
    window.addEventListener('touchmove', onPanMove, { passive: true });
    window.addEventListener('touchend', onPanEnd);

    // Loupe pointer tracking
    const updateLoupe = (e) => {
      if (!this.loupeActive || !this.magnifierLoupe) return;
      const rect = this.container.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      const x = clientX - rect.left;
      const y = clientY - rect.top;

      if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        this.magnifierLoupe.style.display = 'none';
        return;
      }

      this.magnifierLoupe.style.display = 'block';
      this.magnifierLoupe.style.left = `${x}px`;
      this.magnifierLoupe.style.top = `${y}px`;

      const zoomFactor = 2.6;
      const bgW = rect.width * zoomFactor;
      const bgH = rect.height * zoomFactor;
      const bgX = -(x * zoomFactor - 95);
      const bgY = -(y * zoomFactor - 95);

      if (this.loupeInnerBefore) {
        this.loupeInnerBefore.style.backgroundSize = `${bgW}px ${bgH}px`;
        this.loupeInnerBefore.style.backgroundPosition = `${bgX}px ${bgY}px`;
      }
      if (this.loupeInnerAfter) {
        this.loupeInnerAfter.style.backgroundSize = `${bgW}px ${bgH}px`;
        this.loupeInnerAfter.style.backgroundPosition = `${bgX}px ${bgY}px`;
      }

      // Synchronize slider split line inside the loupe
      const sliderX = (this.position / 100) * rect.width;
      const loupeLeft = x - 95;
      const splitInsideLoupe = Math.max(0, Math.min(100, ((sliderX - loupeLeft) / 190) * 100));
      this.container.style.setProperty('--loupe-split', `${splitInsideLoupe}%`);
    };

    this.container.addEventListener('mousemove', updateLoupe);
    this.container.addEventListener('touchmove', updateLoupe, { passive: true });
    this.container.addEventListener('mouseleave', () => {
      if (this.magnifierLoupe) this.magnifierLoupe.style.display = 'none';
    });
  }

  setZoom(level) {
    this.zoomLevel = level;
    if (level <= 1.0) {
      this.panX = 0;
      this.panY = 0;
      this.container.classList.remove('is-zoomed');
    } else {
      this.container.classList.add('is-zoomed');
    }

    [this.btnZoomFit, this.btnZoom100, this.btnZoom200].forEach((btn) => {
      if (btn) btn.classList.remove('active');
    });

    if (level <= 1.1 && this.btnZoomFit) {
      this.btnZoomFit.classList.add('active');
    } else if (level < 2.5 && this.btnZoom100) {
      this.btnZoom100.classList.add('active');
    } else if (this.btnZoom200) {
      this.btnZoom200.classList.add('active');
    }

    this.applyTransform();
  }

  resetZoom() {
    this.setZoom(1.0);
  }

  applyTransform() {
    this.container.style.setProperty('--zoom-scale', this.zoomLevel);
    this.container.style.setProperty('--pan-x', `${this.panX}px`);
    this.container.style.setProperty('--pan-y', `${this.panY}px`);
  }

  toggleLoupe() {
    this.loupeActive = !this.loupeActive;
    if (this.btnLoupeToggle) {
      this.btnLoupeToggle.classList.toggle('active', this.loupeActive);
    }
    if (!this.loupeActive && this.magnifierLoupe) {
      this.magnifierLoupe.style.display = 'none';
    } else {
      this.updateLoupeAssets();
    }
  }

  updateLoupeAssets() {
    if (!this.beforeUrl || !this.afterUrl) return;
    if (this.loupeInnerBefore) {
      this.loupeInnerBefore.style.backgroundImage = `url("${this.beforeUrl}")`;
    }
    if (this.loupeInnerAfter) {
      this.loupeInnerAfter.style.backgroundImage = `url("${this.afterUrl}")`;
    }
  }

  formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  cleanupVideoSync() {
    this.isLoopActive = false;
    if (this.syncRafId) {
      window.cancelAnimationFrame(this.syncRafId);
      this.syncRafId = null;
    }
    if (this.videoCallbackId && this.masterVideo && 'cancelVideoFrameCallback' in this.masterVideo) {
      this.masterVideo.cancelVideoFrameCallback(this.videoCallbackId);
      this.videoCallbackId = null;
    }
    if (this.masterVideo) {
      this.masterVideo.pause();
      this.masterVideo.src = '';
      this.masterVideo = null;
    }
    if (this.slaveVideo) {
      this.slaveVideo.pause();
      this.slaveVideo.src = '';
      this.slaveVideo = null;
    }
    const mount = document.getElementById('videoMediaMount');
    if (mount) {
      mount.innerHTML = '';
    }
  }

  reset() {
    this.cleanupVideoSync();
    if (this.canvas) {
      this.canvas.style.display = 'none';
      if (this.ctx) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      }
    }
    this.layerBefore.innerHTML = '';
    this.layerAfter.innerHTML = '';
    this.resetZoom();
    if (this.magnifierLoupe) this.magnifierLoupe.style.display = 'none';
    this.container.style.display = 'none';
    this.videoControls.style.display = 'none';
  }
}

// Export global instance
window.ComparisonSlider = ComparisonSlider;
