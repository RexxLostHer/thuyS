/**
 * Wink HD Enhancer — Interactive Split Comparison Slider & Dual-Video Sync Engine
 * High-performance 60fps CSS clip-path comparison slider for images and synchronized dual videos.
 */

class ComparisonSlider {
  constructor(options = {}) {
    this.container = options.container || document.getElementById('comparisonContainer');
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
    this.syncThreshold = 0.04; // 40ms (~1 frame drift threshold)

    this.initSliderEvents();
    this.initVideoControlEvents();
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

    // Touch events
    this.container.addEventListener('touchstart', onStart, { passive: false });
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
  }

  /**
   * Set up Image Comparison Mode
   */
  setImageComparison(beforeUrl, afterUrl) {
    this.cleanupVideoSync();
    this.mode = 'image';
    this.videoControls.style.display = 'none';

    this.layerBefore.innerHTML = `<img src="${beforeUrl}" class="comparison-media" alt="Original Low-Res">`;
    this.layerAfter.innerHTML = `<img src="${afterUrl}" class="comparison-media" alt="Wink HD Enhanced">`;

    this.container.style.display = 'block';
    this.updatePosition(50);
  }

  /**
   * Set up Synchronized Dual-Video Comparison Mode
   */
  setVideoComparison(beforeUrl, afterUrl) {
    this.cleanupVideoSync();
    this.mode = 'video';
    this.videoControls.style.display = 'flex';

    // Master is Enhanced HD (Layer After) - provides audio
    const master = document.createElement('video');
    master.src = afterUrl;
    master.className = 'comparison-media';
    master.playsInline = true;
    master.loop = true;
    master.crossOrigin = 'anonymous';

    // Slave is Original Low-Res (Layer Before) - always muted
    const slave = document.createElement('video');
    slave.src = beforeUrl;
    slave.className = 'comparison-media';
    slave.playsInline = true;
    slave.loop = true;
    slave.muted = true;
    slave.crossOrigin = 'anonymous';

    this.layerBefore.innerHTML = '';
    this.layerBefore.appendChild(slave);

    this.layerAfter.innerHTML = '';
    this.layerAfter.appendChild(master);

    this.masterVideo = master;
    this.slaveVideo = slave;

    this.container.style.display = 'block';
    this.updatePosition(50);

    this.bindVideoSync(master, slave);
  }

  bindVideoSync(master, slave) {
    // 1. Playback State Synchronization
    master.addEventListener('play', () => {
      slave.play().catch(() => {});
      this.updatePlayPauseButton(true);
    });

    master.addEventListener('pause', () => {
      slave.pause();
      this.updatePlayPauseButton(false);
    });

    master.addEventListener('waiting', () => {
      slave.pause();
    });

    master.addEventListener('playing', () => {
      slave.play().catch(() => {});
      this.updatePlayPauseButton(true);
    });

    // 2. Seeking Synchronization
    const syncTime = () => {
      if (Math.abs(slave.currentTime - master.currentTime) > 0.02) {
        slave.currentTime = master.currentTime;
      }
    };
    master.addEventListener('seeking', syncTime);
    master.addEventListener('seeked', syncTime);

    // 3. Playback Rate Synchronization
    master.addEventListener('ratechange', () => {
      slave.playbackRate = master.playbackRate;
    });

    // 4. Scrubber & Time Display Updates
    master.addEventListener('timeupdate', () => {
      if (!this.masterVideo || isNaN(this.masterVideo.duration)) return;
      const cur = this.masterVideo.currentTime;
      const dur = this.masterVideo.duration;
      const pct = (cur / dur) * 100;
      this.videoScrubber.value = isNaN(pct) ? 0 : pct;
      this.timeDisplay.textContent = `${this.formatTime(cur)} / ${this.formatTime(dur)}`;
    });

    master.addEventListener('loadedmetadata', () => {
      this.timeDisplay.textContent = `00:00 / ${this.formatTime(master.duration)}`;
    });

    // 5. Continuous Drift Correction RAF Loop
    const syncLoop = () => {
      if (this.masterVideo && this.slaveVideo && !this.masterVideo.paused) {
        const drift = Math.abs(this.masterVideo.currentTime - this.slaveVideo.currentTime);
        if (drift > this.syncThreshold) {
          this.slaveVideo.currentTime = this.masterVideo.currentTime;
        }
      }
      this.syncRafId = requestAnimationFrame(syncLoop);
    };
    this.syncRafId = requestAnimationFrame(syncLoop);

    // Auto play master on load
    master.play().catch(() => {
      // Browsers may block autoplay with sound; mute if needed or await click
      this.updatePlayPauseButton(false);
    });
  }

  initVideoControlEvents() {
    if (!this.btnPlayPause) return;

    this.btnPlayPause.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!this.masterVideo) return;
      if (this.masterVideo.paused) {
        this.masterVideo.play().catch(() => {});
      } else {
        this.masterVideo.pause();
      }
    });

    this.videoScrubber.addEventListener('input', (e) => {
      if (!this.masterVideo || isNaN(this.masterVideo.duration)) return;
      const targetTime = (e.target.value / 100) * this.masterVideo.duration;
      this.masterVideo.currentTime = targetTime;
      if (this.slaveVideo) {
        this.slaveVideo.currentTime = targetTime;
      }
    });

    if (this.btnMute) {
      this.btnMute.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.masterVideo) return;
        this.masterVideo.muted = !this.masterVideo.muted;
        this.btnMute.innerHTML = this.masterVideo.muted
          ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`
          : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`;
      });
    }
  }

  updatePlayPauseButton(isPlaying) {
    if (!this.btnPlayPause) return;
    this.btnPlayPause.innerHTML = isPlaying
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  }

  formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  cleanupVideoSync() {
    if (this.syncRafId) {
      cancelAnimationFrame(this.syncRafId);
      this.syncRafId = null;
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
  }

  reset() {
    this.cleanupVideoSync();
    this.layerBefore.innerHTML = '';
    this.layerAfter.innerHTML = '';
    this.container.style.display = 'none';
    this.videoControls.style.display = 'none';
  }
}

// Export global instance
window.ComparisonSlider = ComparisonSlider;
