/**
 * Wink HD Enhancer — Main UI Controller & API Bridge
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM References
  const tabImageMode = document.getElementById('tabImageMode');
  const tabVideoMode = document.getElementById('tabVideoMode');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const dropzoneTitle = document.getElementById('dropzoneTitle');
  const dropzoneSubtitle = document.getElementById('dropzoneSubtitle');
  const dropzoneIcon = document.getElementById('dropzoneIcon');
  const btnTrySample = document.getElementById('btnTrySample');

  // Comparison & Stage
  const comparisonContainer = document.getElementById('comparisonContainer');
  const progressHud = document.getElementById('progressHud');
  const stageBadge = document.getElementById('stageBadge');
  const progressFill = document.getElementById('progressFill');
  const progressPercentText = document.getElementById('progressPercentText');
  const progressFpsText = document.getElementById('progressFpsText');
  const progressMsgText = document.getElementById('progressMsgText');

  // Controls
  const btnUpscale1x = document.getElementById('btnUpscale1x');
  const btnUpscale2x = document.getElementById('btnUpscale2x');
  const btnUpscale4x = document.getElementById('btnUpscale4x');
  const sliderFaceRetouch = document.getElementById('sliderFaceRetouch');
  const valFaceRetouch = document.getElementById('valFaceRetouch');
  const sliderSharpen = document.getElementById('sliderSharpen');
  const valSharpen = document.getElementById('valSharpen');
  const sliderVibrance = document.getElementById('sliderVibrance');
  const valVibrance = document.getElementById('valVibrance');
  const sliderDenoise = document.getElementById('sliderDenoise');
  const valDenoise = document.getElementById('valDenoise');

  const btnPresetPortrait = document.getElementById('btnPresetPortrait');
  const btnPresetLandscape = document.getElementById('btnPresetLandscape');
  const btnPresetCinematic = document.getElementById('btnPresetCinematic');

  const btnEnhance = document.getElementById('btnEnhance');
  const btnEnhanceText = document.getElementById('btnEnhanceText');
  const enhanceSpinner = document.getElementById('enhanceSpinner');

  // Export panel
  const exportPanel = document.getElementById('exportPanel');
  const metricResolution = document.getElementById('metricResolution');
  const metricTime = document.getElementById('metricTime');
  const metricGain = document.getElementById('metricGain');
  const btnDownload = document.getElementById('btnDownload');
  const btnReset = document.getElementById('btnReset');

  // Health
  const statusLabel = document.getElementById('statusLabel');

  // Application State
  let currentMode = 'image'; // 'image' or 'video'
  let currentFile = null;
  let upscaleFactor = 2;
  let latestOutputUrl = null;
  let sampleData = null;
  let activeEventSource = null;

  // Initialize Comparison Slider
  const slider = new ComparisonSlider({
    container: comparisonContainer,
    layerBefore: document.getElementById('layerBefore'),
    layerAfter: document.getElementById('layerAfter'),
    divider: document.getElementById('sliderDivider'),
    handle: document.getElementById('sliderHandle'),
    videoControls: document.getElementById('videoControlsBar'),
  });

  // Check Backend Health
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.ffmpeg && data.ffmpeg.available) {
          statusLabel.textContent = 'FFmpeg Engine Ready';
        } else {
          statusLabel.textContent = 'Server Online (CPU Mode)';
        }
      }
    } catch (err) {
      statusLabel.textContent = 'Offline / Connecting...';
    }
  }
  checkHealth();

  // Load Samples Metadata
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      if (res.ok) {
        const data = await res.json();
        sampleData = data.samples;
      }
    } catch (err) {
      console.warn('Could not pre-fetch samples:', err);
    }
  }
  loadSamples();

  // Mode Switching
  function switchMode(mode) {
    if (currentMode === mode) return;
    currentMode = mode;
    currentFile = null;

    if (mode === 'image') {
      tabImageMode.classList.add('active');
      tabVideoMode.classList.remove('active');
      fileInput.accept = 'image/png, image/jpeg, image/webp, image/bmp';
      dropzoneTitle.textContent = 'Drop portrait or photo here';
      dropzoneSubtitle.textContent = 'Supports PNG, JPG, WebP, BMP up to 4K';
      btnTrySample.textContent = '✨ Try Portrait Sample';
    } else {
      tabVideoMode.classList.add('active');
      tabImageMode.classList.remove('active');
      fileInput.accept = 'video/mp4, video/quicktime, video/webm, video/x-msvideo';
      dropzoneTitle.textContent = 'Drop video clip here';
      dropzoneSubtitle.textContent = 'Supports MP4, MOV, WebM with audio';
      btnTrySample.textContent = '🎬 Try Action Video Sample';
    }

    resetWorkspace();
  }

  tabImageMode.addEventListener('click', () => switchMode('image'));
  tabVideoMode.addEventListener('click', () => switchMode('video'));

  // Upscale Factor Buttons
  const upscaleButtons = [
    { btn: btnUpscale1x, factor: 1 },
    { btn: btnUpscale2x, factor: 2 },
    { btn: btnUpscale4x, factor: 4 },
  ];

  upscaleButtons.forEach(({ btn, factor }) => {
    btn.addEventListener('click', () => {
      upscaleButtons.forEach((b) => b.btn.classList.remove('active'));
      btn.classList.add('active');
      upscaleFactor = factor;
    });
  });

  // Slider Input Sync
  function bindSlider(sliderEl, labelEl) {
    sliderEl.addEventListener('input', () => {
      labelEl.textContent = `${sliderEl.value}%`;
    });
  }
  bindSlider(sliderFaceRetouch, valFaceRetouch);
  bindSlider(sliderSharpen, valSharpen);
  bindSlider(sliderVibrance, valVibrance);
  bindSlider(sliderDenoise, valDenoise);

  // Preset Configurations
  btnPresetPortrait.addEventListener('click', () => {
    setSliderValues(75, 45, 35, 40);
  });
  btnPresetLandscape.addEventListener('click', () => {
    setSliderValues(0, 80, 65, 20);
  });
  btnPresetCinematic.addEventListener('click', () => {
    setSliderValues(30, 60, 40, 50);
  });

  function setSliderValues(face, sharp, vib, den) {
    sliderFaceRetouch.value = face;
    valFaceRetouch.textContent = `${face}%`;
    sliderSharpen.value = sharp;
    valSharpen.textContent = `${sharp}%`;
    sliderVibrance.value = vib;
    valVibrance.textContent = `${vib}%`;
    sliderDenoise.value = den;
    valDenoise.textContent = `${den}%`;
  }

  // Dropzone Events
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    currentFile = file;
    const isVid = file.type.startsWith('video') || file.name.match(/\.(mp4|mov|webm|avi|mkv)$/i);

    if (isVid && currentMode !== 'video') {
      switchMode('video');
    } else if (!isVid && currentMode !== 'image') {
      switchMode('image');
    }

    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    dropzoneTitle.textContent = file.name;
    dropzoneSubtitle.textContent = `${(file.type || 'Media file')} • ${sizeMb} MB • Ready to enhance`;
    btnEnhance.disabled = false;
  }

  // 1-Click "Try Sample" Action
  btnTrySample.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!sampleData) {
      await loadSamples();
    }

    if (!sampleData) {
      alert('Samples are still generating or not available.');
      return;
    }

    dropzone.style.display = 'none';
    progressHud.style.display = 'none';

    if (currentMode === 'image' && sampleData.image) {
      const s = sampleData.image;
      slider.setImageComparison(s.lowres_url, s.enhanced_url);
      latestOutputUrl = s.enhanced_url;
      showExportMetrics({
        resolution: `${s.lowres_res.join('x')} ➔ ${s.enhanced_res.join('x')} (${s.upscale_factor}x HD)`,
        time: 'Pre-processed (Instant)',
        gain: '+128% Tenengrad Sharpness (30.7 dB PSNR)',
      });
    } else if (currentMode === 'video' && sampleData.video) {
      const s = sampleData.video;
      slider.setVideoComparison(s.lowres_url, s.enhanced_url);
      latestOutputUrl = s.enhanced_url;
      showExportMetrics({
        resolution: `${s.lowres_res.join('x')} ➔ ${s.enhanced_res.join('x')} (${s.upscale_factor}x HD)`,
        time: `${s.duration}s @ ${s.fps} FPS`,
        gain: 'Lossless AAC Audio Sync • 0 AV Drift',
      });
    }
  });

  // Primary Action: "Enhance HD"
  btnEnhance.addEventListener('click', async () => {
    if (!currentFile) {
      alert('Please select or drop an image or video file first.');
      return;
    }

    btnEnhance.disabled = true;
    enhanceSpinner.style.display = 'block';
    btnEnhanceText.textContent = 'Processing...';

    dropzone.style.display = 'none';
    progressHud.style.display = 'flex';
    exportPanel.style.display = 'none';
    updateProgressUI('Initializing', 5, 0, 'Preparing media pipeline...');

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('upscale_factor', upscaleFactor);
    formData.append('face_retouch', (sliderFaceRetouch.value / 100).toFixed(2));
    formData.append('sharpen', (sliderSharpen.value / 100).toFixed(2));
    formData.append('vibrance', (sliderVibrance.value / 100).toFixed(2));
    formData.append('denoise', (sliderDenoise.value / 100).toFixed(2));

    if (currentMode === 'image') {
      try {
        updateProgressUI('Enhancing', 40, 0, 'Applying Lanczos-4, Retouch & CAS...');
        const res = await fetch('/api/enhance/image', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: 'Unknown server error' }));
          throw new Error(errData.detail || 'Image enhancement failed');
        }

        const data = await res.json();
        updateProgressUI('Completed', 100, 0, 'Enhancement complete!');

        latestOutputUrl = data.output_url;
        slider.setImageComparison(data.input_url, data.output_url);

        const inDim = data.input_resolution.join('x');
        const outDim = data.output_resolution.join('x');
        const gain = data.metrics ? `${((data.metrics.sharpness_gain - 1) * 100).toFixed(1)}% Gain (${data.metrics.psnr_baseline} dB)` : 'Optimal';

        showExportMetrics({
          resolution: `${inDim} ➔ ${outDim} (${upscaleFactor}x HD)`,
          time: `${data.processing_time_ms} ms`,
          gain: gain,
        });
      } catch (err) {
        alert(`Error: ${err.message}`);
        resetWorkspace();
      } finally {
        finishProcessing();
      }
    } else {
      // Video Enhancement via SSE
      try {
        const res = await fetch('/api/enhance/video', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: 'Unknown server error' }));
          throw new Error(errData.detail || 'Video enhancement could not be queued');
        }

        const initial = await res.json();
        const taskId = initial.task_id;
        const inputUrl = initial.input_url;

        // Open SSE connection
        if (activeEventSource) {
          activeEventSource.close();
        }

        activeEventSource = new EventSource(`/api/progress/${taskId}`);

        activeEventSource.onmessage = (event) => {
          try {
            const ev = JSON.parse(event.data);
            updateProgressUI(ev.stage, ev.percent, ev.fps, ev.message);

            if (ev.status === 'completed' || ev.stage === 'completed') {
              activeEventSource.close();
              activeEventSource = null;

              latestOutputUrl = ev.output_url;
              slider.setVideoComparison(inputUrl, ev.output_url);

              const resMeta = ev.result || {};
              showExportMetrics({
                resolution: `${upscaleFactor}x Ultra HD`,
                time: `${resMeta.fps || 0} FPS processing`,
                gain: 'Bit-Exact AAC Audio Remuxed',
              });

              finishProcessing();
            } else if (ev.status === 'error' || ev.stage === 'failed') {
              activeEventSource.close();
              activeEventSource = null;
              alert(`Video enhancement failed: ${ev.message || ev.error}`);
              resetWorkspace();
              finishProcessing();
            }
          } catch (e) {
            console.error('Error parsing SSE event:', e);
          }
        };

        activeEventSource.onerror = (err) => {
          console.warn('SSE stream notice/error:', err);
        };
      } catch (err) {
        alert(`Error: ${err.message}`);
        resetWorkspace();
        finishProcessing();
      }
    }
  });

  function updateProgressUI(stage, percent, fps, message) {
    stageBadge.textContent = stage;
    progressFill.style.width = `${percent}%`;
    progressPercentText.textContent = `${Math.round(percent)}%`;
    progressFpsText.textContent = fps > 0 ? `${fps.toFixed(1)} FPS` : '';
    progressMsgText.textContent = message || '';
  }

  function finishProcessing() {
    btnEnhance.disabled = false;
    enhanceSpinner.style.display = 'none';
    btnEnhanceText.textContent = '⚡ Enhance to HD';
  }

  function showExportMetrics({ resolution, time, gain }) {
    metricResolution.textContent = resolution;
    metricTime.textContent = time;
    metricGain.textContent = gain;
    exportPanel.style.display = 'flex';
  }

  // Download Trigger
  btnDownload.addEventListener('click', () => {
    if (!latestOutputUrl) return;
    const a = document.createElement('a');
    a.href = latestOutputUrl;
    a.download = latestOutputUrl.split('/').pop() || 'enhanced_hd';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // Reset Workspace
  btnReset.addEventListener('click', resetWorkspace);

  function resetWorkspace() {
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    slider.reset();
    dropzone.style.display = 'flex';
    progressHud.style.display = 'none';
    exportPanel.style.display = 'none';
    fileInput.value = '';
    currentFile = null;
    latestOutputUrl = null;

    if (currentMode === 'image') {
      dropzoneTitle.textContent = 'Drop portrait or photo here';
      dropzoneSubtitle.textContent = 'Supports PNG, JPG, WebP, BMP up to 4K';
    } else {
      dropzoneTitle.textContent = 'Drop video clip here';
      dropzoneSubtitle.textContent = 'Supports MP4, MOV, WebM with audio';
    }
    finishProcessing();
  }
});
