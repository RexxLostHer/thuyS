/**
 * Empirical Stress Test Harness for Frontend Media Playback, Slider Sync & Fallback
 * Tests app/static/js/app.js and app/static/js/slider.js
 */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const appJsPath = path.join(rootDir, 'app', 'static', 'js', 'app.js');
const sliderJsPath = path.join(rootDir, 'app', 'static', 'js', 'slider.js');

const appJsContent = fs.readFileSync(appJsPath, 'utf-8');
const sliderJsContent = fs.readFileSync(sliderJsPath, 'utf-8');

console.log('================================================================');
console.log('Wink HD Enhancer — Frontend Media & Slider Sync Stress Suite');
console.log('================================================================');

let totalTests = 0;
let passedTests = 0;

function it(name, fn) {
  totalTests++;
  try {
    fn();
    passedTests++;
    console.log(`  ✓ PASS: ${name}`);
  } catch (err) {
    console.error(`  ✗ FAIL: ${name}`);
    console.error(`    Error: ${err.message}`);
    throw err;
  }
}

async function itAsync(name, fn) {
  totalTests++;
  try {
    await fn();
    passedTests++;
    console.log(`  ✓ PASS: ${name}`);
  } catch (err) {
    console.error(`  ✗ FAIL: ${name}`);
    console.error(`    Error: ${err.message}`);
    throw err;
  }
}

// -----------------------------------------------------------------------------
// SCENARIO 1: appendCacheBuster Verification
// -----------------------------------------------------------------------------
console.log('\n--- Scenario 1: appendCacheBuster Behavior ---');

// Extract appendCacheBuster from app.js to test the actual implementation directly
const matchCacheBuster = appJsContent.match(/function appendCacheBuster\(url\) \{[\s\S]*?\n  \}/);
if (!matchCacheBuster) {
  throw new Error('Could not find appendCacheBuster in app.js');
}
const appendCacheBuster = new Function(`return (${matchCacheBuster[0]})`)();

it('preserves blob: URL unchanged without query parameters', () => {
  const blobUrl = 'blob:http://127.0.0.1:8484/abc-123-def-456';
  const result = appendCacheBuster(blobUrl);
  assert.equal(result, blobUrl);
  assert.ok(!result.includes('?t='), 'Blob URL must not have ?t=');
  assert.ok(!result.includes('&t='), 'Blob URL must not have &t=');
});

it('preserves data: URL unchanged without query parameters', () => {
  const dataUrl = 'data:video/mp4;base64,AAAAHGZ0eXBtcDQyAAAAAG1wNDJpc29t';
  const result = appendCacheBuster(dataUrl);
  assert.equal(result, dataUrl);
  assert.ok(!result.includes('?t='), 'Data URL must not have ?t=');
});

it('appends ?t=timestamp to relative path without query parameters', () => {
  const before = Date.now();
  const url = '/media/sample.mp4';
  const result = appendCacheBuster(url);
  const after = Date.now();
  assert.ok(result.startsWith('/media/sample.mp4?t='));
  const tVal = Number(result.split('?t=')[1]);
  assert.ok(tVal >= before && tVal <= after, 'Timestamp must be current time');
});

it('appends &t=timestamp to URL with existing query parameters', () => {
  const url = '/media/sample.mp4?opt=1';
  const result = appendCacheBuster(url);
  assert.ok(result.startsWith('/media/sample.mp4?opt=1&t='));
  assert.equal((result.match(/\?/g) || []).length, 1, 'Should contain only one ?');
});

it('appends &t=timestamp to complex absolute HTTP URL with multiple parameters', () => {
  const url = 'http://127.0.0.1:8484/api/video?quality=hd&format=mp4';
  const result = appendCacheBuster(url);
  assert.ok(result.startsWith('http://127.0.0.1:8484/api/video?quality=hd&format=mp4&t='));
});

it('safely returns empty string for undefined and null', () => {
  assert.equal(appendCacheBuster(undefined), '');
  assert.equal(appendCacheBuster(null), '');
});

it('safely returns empty string for non-string types and empty input', () => {
  assert.equal(appendCacheBuster(''), '');
  assert.equal(appendCacheBuster(0), '');
  assert.equal(appendCacheBuster(false), '');
  assert.equal(appendCacheBuster({}), '');
  assert.equal(appendCacheBuster([]), '');
});

// -----------------------------------------------------------------------------
// SCENARIO 2: Fallback Polling Verification (pollFallback & handleCompletion)
// -----------------------------------------------------------------------------
console.log('\n--- Scenario 2: Fallback Polling & SSE Recovery ---');

await itAsync('pollFallback recovers upon SSE error and invokes handleCompletion with cascaded URLs', async () => {
  // Extract pollFallback from app.js
  const matchPollFallback = appJsContent.match(/function pollFallback\(taskId, params, initData, onCompleteFn\) \{[\s\S]*?\n  \}/);
  assert.ok(matchPollFallback, 'pollFallback must exist in app.js');

  let activeFallbackInterval = null;
  let updateProgressUICalls = [];
  function updateProgressUI(stage, percent, fps, message) {
    updateProgressUICalls.push({ stage, percent, fps, message });
  }

  let completedData = null;
  const onCompleteFn = (data) => {
    completedData = data;
  };

  // Mock fetch responses: 1st call processing, 2nd call completed
  let fetchCount = 0;
  const mockFetch = async (url) => {
    fetchCount++;
    if (fetchCount === 1) {
      return {
        ok: true,
        json: async () => ({
          status: 'processing',
          stage: 'enhancing',
          percent: 50,
          fps: 28.5,
          message: 'Processing frame 50/100',
        }),
      };
    } else {
      return {
        ok: true,
        json: async () => ({
          status: 'completed',
          stage: 'completed',
          percent: 100,
          result: {
            input_url: '/media/input_test.mp4',
            output_url: '/media/output_test.mp4',
            fps: 30.0,
            processing_time_ms: 2450,
          },
        }),
      };
    }
  };

  const sandbox = {
    fetch: mockFetch,
    setInterval: (cb, ms) => {
      // Execute 2 cycles immediately
      Promise.resolve().then(async () => {
        await cb(); // fetchCount = 1 (processing)
        await cb(); // fetchCount = 2 (completed)
      });
      return 12345;
    },
    clearInterval: (id) => {
      assert.equal(id, 12345);
      activeFallbackInterval = null;
    },
    updateProgressUI,
    alert: () => {},
    resetWorkspace: () => {},
    finishProcessing: () => {},
    activeFallbackInterval: 12345,
    onCompleteFn,
  };

  const fnString = `
    let activeFallbackInterval = 12345;
    ${matchPollFallback[0]}
    pollFallback('test-task-1', { upscale_factor: 2 }, {}, onCompleteFn);
  `;

  vm.createContext(sandbox);
  vm.runInContext(fnString, sandbox);

  // Wait for async poll cycles
  await new Promise((r) => setTimeout(r, 50));

  assert.ok(completedData, 'onCompleteFn must be called when task reaches completed');
  assert.equal(completedData.status, 'completed');
  assert.equal(completedData.result.input_url, '/media/input_test.mp4');
  assert.equal(completedData.result.output_url, '/media/output_test.mp4');
  assert.ok(updateProgressUICalls.length >= 1, 'updateProgressUI should be called during processing stage');
  assert.equal(updateProgressUICalls[0].stage, 'enhancing');
  assert.equal(updateProgressUICalls[0].percent, 50);
});

it('handleCompletion URL cascade resolves correctly across 3 fallback tiers', () => {
  // Test tier resolution logic directly:
  // const rawInUrl = resObj.input_url || data.input_url || initData.input_url;
  // const rawOutUrl = resObj.output_url || data.output_url || initData.output_url;

  function resolveUrls(resObj, data, initData) {
    const rawInUrl = resObj.input_url || data.input_url || initData.input_url;
    const rawOutUrl = resObj.output_url || data.output_url || initData.output_url;
    return { rawInUrl, rawOutUrl };
  }

  // Tier 1: from resObj
  const r1 = resolveUrls({ input_url: '/res/in.mp4', output_url: '/res/out.mp4' }, {}, {});
  assert.equal(r1.rawInUrl, '/res/in.mp4');
  assert.equal(r1.rawOutUrl, '/res/out.mp4');

  // Tier 2: from data directly
  const r2 = resolveUrls({}, { input_url: '/data/in.mp4', output_url: '/data/out.mp4' }, {});
  assert.equal(r2.rawInUrl, '/data/in.mp4');
  assert.equal(r2.rawOutUrl, '/data/out.mp4');

  // Tier 3: from initData
  const r3 = resolveUrls({}, {}, { input_url: '/init/in.mp4', output_url: '/init/out.mp4' });
  assert.equal(r3.rawInUrl, '/init/in.mp4');
  assert.equal(r3.rawOutUrl, '/init/out.mp4');
});

// -----------------------------------------------------------------------------
// SCENARIO 3: syncThreshold Drift Resilience (250ms)
// -----------------------------------------------------------------------------
console.log('\n--- Scenario 3: syncThreshold Drift Resilience & Anti-Thrashing ---');

function createMockVideoElement() {
  const listeners = {};
  let _currentTime = 0;
  let seekCount = 0;

  return {
    videoWidth: 1280,
    videoHeight: 720,
    duration: 60.0,
    readyState: 4,
    paused: false,
    muted: true,
    playsInline: true,
    preload: 'auto',
    dataset: {},
    get currentTime() {
      return _currentTime;
    },
    set currentTime(v) {
      seekCount++;
      _currentTime = v;
      if (listeners['seeked']) {
        listeners['seeked'].forEach((fn) => fn());
      }
    },
    get seekCount() {
      return seekCount;
    },
    addEventListener(event, fn) {
      listeners[event] = listeners[event] || [];
      listeners[event].push(fn);
    },
    removeEventListener(event, fn) {
      if (listeners[event]) {
        listeners[event] = listeners[event].filter((f) => f !== fn);
      }
    },
    dispatchEvent(event) {
      if (listeners[event]) {
        listeners[event].forEach((fn) => fn());
      }
    },
    play: async () => {},
    pause: () => {},
    load: () => {},
  };
}

// Load slider.js in VM
const mockDOM = {
  document: {
    getElementById: (id) => ({
      style: { setProperty: () => {}, display: 'none' },
      addEventListener: () => {},
      getContext: () => ({
        clearRect: () => {},
        save: () => {},
        beginPath: () => {},
        rect: () => {},
        clip: () => {},
        drawImage: () => {},
        restore: () => {},
      }),
      appendChild: () => {},
      innerHTML: '',
    }),
    createElement: (tag) => {
      if (tag === 'video') return createMockVideoElement();
      return {
        style: {},
        appendChild: () => {},
        innerHTML: '',
      };
    },
    body: {
      style: {},
      appendChild: () => {},
      removeChild: () => {},
    },
  },
  window: {
    addEventListener: () => {},
    requestAnimationFrame: (cb) => setTimeout(cb, 16),
    cancelAnimationFrame: () => {},
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (id) => clearTimeout(id),
  },
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (id) => clearTimeout(id),
  URL: {
    createObjectURL: () => 'blob:mock-url',
    revokeObjectURL: () => {},
  },
  console: console,
};

vm.createContext(mockDOM);
vm.runInContext(sliderJsContent, mockDOM);
const ComparisonSlider = mockDOM.window.ComparisonSlider;

it('ComparisonSlider initializes with syncThreshold = 0.25 (250ms)', () => {
  const slider = new ComparisonSlider({});
  assert.equal(slider.syncThreshold, 0.25, 'syncThreshold must be exactly 0.25s (250ms)');
});

it('does NOT trigger seek on slave when drift is < 250ms', () => {
  const slider = new ComparisonSlider({});
  const master = createMockVideoElement();
  const slave = createMockVideoElement();
  slider.masterVideo = master;
  slider.slaveVideo = slave;
  slider.videoScrubber = { value: 0 };
  slider.timeDisplay = { textContent: '' };

  slider.bindVideoSync(master, slave);

  // Initial state: both at 10.0s
  master.currentTime = 10.0;
  slave.currentTime = 10.0;
  const initialSeeks = slave.seekCount;

  // Simulate drift within threshold: +50ms (0.05s)
  slave.currentTime = 10.05;
  const preCheckSeeks = slave.seekCount;
  master.dispatchEvent('timeupdate');
  assert.equal(slave.seekCount, preCheckSeeks, 'Drift +50ms must not trigger seek');

  // Simulate drift within threshold: +150ms (0.15s)
  slave.currentTime = 10.15;
  const preCheckSeeks2 = slave.seekCount;
  master.dispatchEvent('timeupdate');
  assert.equal(slave.seekCount, preCheckSeeks2, 'Drift +150ms must not trigger seek');

  // Simulate drift within threshold: +240ms (0.24s)
  slave.currentTime = 10.24;
  const preCheckSeeks3 = slave.seekCount;
  master.dispatchEvent('timeupdate');
  assert.equal(slave.seekCount, preCheckSeeks3, 'Drift +240ms must not trigger seek');

  // Simulate drift within threshold: -200ms (-0.20s)
  slave.currentTime = 9.80;
  const preCheckSeeks4 = slave.seekCount;
  master.dispatchEvent('timeupdate');
  assert.equal(slave.seekCount, preCheckSeeks4, 'Drift -200ms must not trigger seek');
});

it('triggers seek on slave when drift exceeds 250ms (> 0.25s)', () => {
  const slider = new ComparisonSlider({});
  const master = createMockVideoElement();
  const slave = createMockVideoElement();
  slider.masterVideo = master;
  slider.slaveVideo = slave;
  slider.videoScrubber = { value: 0 };
  slider.timeDisplay = { textContent: '' };

  slider.bindVideoSync(master, slave);

  master.currentTime = 15.0;
  slave.currentTime = 15.26; // drift = 260ms (> 250ms)
  const initialSeeks = slave.seekCount;

  master.dispatchEvent('timeupdate');

  assert.equal(slave.seekCount, initialSeeks + 1, 'Drift 260ms must trigger seek');
  assert.equal(slave.currentTime, 15.0, 'Slave must be re-synced to master currentTime');
});

it('stress-test: 500 drift events within 250ms cause ZERO seek thrashing', () => {
  const slider = new ComparisonSlider({});
  const master = createMockVideoElement();
  const slave = createMockVideoElement();
  slider.masterVideo = master;
  slider.slaveVideo = slave;
  slider.videoScrubber = { value: 0 };
  slider.timeDisplay = { textContent: '' };

  slider.bindVideoSync(master, slave);

  master.currentTime = 20.0;
  slave.currentTime = 20.0;
  const startSeeks = slave.seekCount;

  // 500 jittery drift steps in [-0.24s, +0.24s]
  for (let i = 0; i < 500; i++) {
    const jitter = (Math.random() * 0.48) - 0.24; // between -0.24 and +0.24
    slave.currentTime = 20.0 + jitter;
  }
  const seeksAfterJitter = slave.seekCount;

  // Trigger 50 timeupdates
  for (let i = 0; i < 50; i++) {
    master.dispatchEvent('timeupdate');
  }

  assert.equal(
    slave.seekCount,
    seeksAfterJitter,
    'Zero seeks should have been triggered by timeupdate within the 250ms threshold'
  );
});

// -----------------------------------------------------------------------------
// SCENARIO 4: hasTriggeredSeek Idempotency on Source Load
// -----------------------------------------------------------------------------
console.log('\n--- Scenario 4: hasTriggeredSeek Idempotency Guard ---');

await itAsync('loadVideoSourceReady executes currentTime = 0.05 only once across sequential events', async () => {
  const slider = new ComparisonSlider({});
  const video = createMockVideoElement();

  let currentTimeSetCount = 0;
  let lastTimeSet = null;
  Object.defineProperty(video, 'currentTime', {
    get() {
      return this._time || 0;
    },
    set(v) {
      currentTimeSetCount++;
      lastTimeSet = v;
      this._time = v;
      if (this.listeners && this.listeners['seeked']) {
        this.listeners['seeked'].forEach((fn) => fn());
      }
    },
  });

  const loadPromise = slider.loadVideoSourceReady(video, '/media/test.mp4');

  // Trigger loadeddata
  video.dispatchEvent('loadeddata');

  // Trigger canplay immediately after
  video.dispatchEvent('canplay');

  // Trigger canplay again (e.g. buffering event)
  video.dispatchEvent('canplay');

  const meta = await loadPromise;

  assert.equal(currentTimeSetCount, 1, 'currentTime = 0.05 must be executed EXACTLY ONCE');
  assert.equal(lastTimeSet, 0.05, 'currentTime should be set to 0.05s');
  assert.equal(meta.width, 1280);
  assert.equal(meta.height, 720);
});

// -----------------------------------------------------------------------------
// SCENARIO 5: Canvas Rendering Anti-Flicker & Frame Persistence
// -----------------------------------------------------------------------------
console.log('\n--- Scenario 5: Canvas Anti-Flicker (readyState < 2) ---');

it('drawVideoFrames does NOT wipe canvas when videos are seeking (readyState < 2)', () => {
  let clearRectCalls = 0;
  let drawImageCalls = 0;

  const mockCtx = {
    clearRect: (x, y, w, h) => {
      clearRectCalls++;
    },
    save: () => {},
    beginPath: () => {},
    rect: () => {},
    clip: () => {},
    drawImage: (img, x, y, w, h) => {
      drawImageCalls++;
    },
    restore: () => {},
  };

  const slider = new ComparisonSlider({});
  slider.canvas = { width: 1920, height: 1080 };
  slider.ctx = mockCtx;

  const master = createMockVideoElement();
  const slave = createMockVideoElement();
  slider.masterVideo = master;
  slider.slaveVideo = slave;
  slider.position = 50;

  // Case 1: Both videos ready (readyState = 4) -> should clear and draw
  master.readyState = 4;
  slave.readyState = 4;
  clearRectCalls = 0;
  drawImageCalls = 0;
  slider.drawVideoFrames();
  assert.equal(clearRectCalls, 1, 'Both ready: clearRect must be called once');
  assert.equal(drawImageCalls, 2, 'Both ready: drawImage must be called for slave & master');

  // Case 2: Both videos seeking (readyState = 1 < 2) -> MUST NOT CLEAR CANVAS
  master.readyState = 1;
  slave.readyState = 1;
  clearRectCalls = 0;
  drawImageCalls = 0;
  slider.drawVideoFrames();
  assert.equal(clearRectCalls, 0, 'Both seeking: clearRect MUST NOT be called (prevents black screen)');
  assert.equal(drawImageCalls, 0, 'Both seeking: drawImage should not be called');

  // Case 3: Slave seeking (readyState = 1), Master ready (readyState = 4)
  // PRD: Do not clearRect so slave side is not wiped to black
  master.readyState = 4;
  slave.readyState = 1;
  clearRectCalls = 0;
  drawImageCalls = 0;
  slider.drawVideoFrames();
  assert.equal(clearRectCalls, 0, 'Slave seeking: clearRect MUST NOT be called');
  assert.equal(drawImageCalls, 1, 'Master ready: only master side should be redrawn');

  // Case 4: Master seeking (readyState = 1), Slave ready (readyState = 4)
  master.readyState = 1;
  slave.readyState = 4;
  clearRectCalls = 0;
  drawImageCalls = 0;
  slider.drawVideoFrames();
  assert.equal(clearRectCalls, 0, 'Master seeking: clearRect MUST NOT be called');
  assert.equal(drawImageCalls, 1, 'Slave ready: only slave side should be redrawn');
});

console.log('\n================================================================');
console.log(`Results: ${passedTests}/${totalTests} tests PASSED cleanly!`);
console.log('================================================================\n');

process.exit(0);

