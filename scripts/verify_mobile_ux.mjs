/**
 * Verification Suite for Lumina HD Studio:
 * 1. Mobile UX & Touch Navigation
 * 2. 69-Frame Logo Animation Engine & Assets
 * 3. Cloudflare Tunnel & QR Integration
 */

import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('================================================================');
console.log('Lumina HD Studio — Mobile UX, Logo Animation & Tunnel Verification');
console.log('================================================================\n');

let passed = 0;
let total = 0;

function it(name, fn) {
  total++;
  try {
    fn();
    passed++;
    console.log(`  ✓ PASS: ${name}`);
  } catch (err) {
    console.error(`  ✗ FAIL: ${name}`);
    console.error(`    Error: ${err.message}`);
    throw err;
  }
}

// 1. Assets Verification
console.log('--- 1. Logo Animation Assets (Ujikom Parity) ---');
const animDir = path.join(rootDir, 'app', 'static', 'assets', 'animasi');

it('animasi directory exists and contains all 69 JPG animation frames', () => {
  assert.ok(fs.existsSync(animDir), 'Animasi directory must exist');
  for (let i = 1; i <= 69; i++) {
    const num = String(i).padStart(3, '0');
    const fPath = path.join(animDir, `ezgif-frame-${num}.jpg`);
    assert.ok(fs.existsSync(fPath), `Frame ${num}.jpg must exist`);
  }
});

it('contains transparent shine loop PNG frames (60-69) from Ujikom', () => {
  for (let i = 60; i <= 69; i++) {
    const num = String(i).padStart(3, '0');
    const fPath = path.join(animDir, `ezgif-frame-${num}.png`);
    assert.ok(fs.existsSync(fPath), `Shine frame ${num}.png must exist`);
  }
});

it('favicon and square clean logo are generated in assets', () => {
  const favPath = path.join(rootDir, 'app', 'static', 'assets', 'favicon.png');
  const squareLogoPath = path.join(rootDir, 'app', 'static', 'assets', 'logo_clean_square.png');
  assert.ok(fs.existsSync(favPath), 'favicon.png must exist');
  assert.ok(fs.existsSync(squareLogoPath), 'logo_clean_square.png must exist');
});

// 2. HTML Structure & Mobile Meta Tags
console.log('\n--- 2. Mobile Viewport & Rebranding in HTML ---');
const indexHtmlPath = path.join(rootDir, 'app', 'static', 'index.html');
const indexHtml = fs.readFileSync(indexHtmlPath, 'utf-8');

it('has mobile-friendly viewport with user-scalable=no and viewport-fit=cover', () => {
  assert.ok(indexHtml.includes('viewport-fit=cover'), 'Must include viewport-fit=cover');
  assert.ok(indexHtml.includes('theme-color'), 'Must include theme-color');
  assert.ok(indexHtml.includes('apple-mobile-web-app-capable'), 'Must support iOS PWA meta');
});

it('rebranded to Lumina HD Studio in title and brand elements', () => {
  assert.ok(indexHtml.includes('Lumina HD Studio'), 'Title must be Lumina HD Studio');
  assert.ok(indexHtml.includes('headerLogoCanvas'), 'Header must contain animated canvas');
  assert.ok(indexHtml.includes('btnMobileConnect'), 'Header must contain Mobile Connect button');
});

it('contains Mobile Connect Modal with QR code and tunnel link inputs', () => {
  assert.ok(indexHtml.includes('modalMobileConnect'), 'Modal must exist');
  assert.ok(indexHtml.includes('qrCodeImg'), 'QR Code image must exist');
  assert.ok(indexHtml.includes('inputTunnelUrl'), 'Tunnel URL input must exist');
  assert.ok(indexHtml.includes('inputLocalUrl'), 'Local URL input must exist');
});

// 3. CSS Mobile Styles
console.log('\n--- 3. CSS Responsive Mobile Layout ---');
const styleCssPath = path.join(rootDir, 'app', 'static', 'css', 'style.css');
const styleCss = fs.readFileSync(styleCssPath, 'utf-8');

it('contains 768px and 480px responsive media queries', () => {
  assert.ok(styleCss.includes('@media (max-width: 768px)'), 'Must include 768px breakpoint');
  assert.ok(styleCss.includes('@media (max-width: 480px)'), 'Must include 480px breakpoint');
});

it('contains slider handle touch-action: none for frictionless swiping', () => {
  assert.ok(styleCss.includes('touch-action: none'), 'Slider handle must have touch-action: none');
});

it('contains modal and tunnel tab styles', () => {
  assert.ok(styleCss.includes('.modal-backdrop'), 'Modal backdrop styles must exist');
  assert.ok(styleCss.includes('.tunnel-card'), 'Tunnel card styles must exist');
  assert.ok(styleCss.includes('.brand-icon-animated'), 'Brand icon animated styles must exist');
});

// 4. JavaScript Integration
console.log('\n--- 4. JavaScript Engine & Modules ---');
const logoAnimJsPath = path.join(rootDir, 'app', 'static', 'js', 'logo-anim.js');
const appJsPath = path.join(rootDir, 'app', 'static', 'js', 'app.js');
const sliderJsPath = path.join(rootDir, 'app', 'static', 'js', 'slider.js');

it('logo-anim.js exists and exports setLogoProcessing', () => {
  assert.ok(fs.existsSync(logoAnimJsPath), 'logo-anim.js must exist');
  const code = fs.readFileSync(logoAnimJsPath, 'utf-8');
  assert.ok(code.includes('window.setLogoProcessing'), 'Must define window.setLogoProcessing');
  assert.ok(code.includes('SHINE_INTERVAL_MS'), 'Must define periodic shine timing');
});

it('app.js integrates Mobile Connect modal and Logo Processing hooks', () => {
  const code = fs.readFileSync(appJsPath, 'utf-8');
  assert.ok(code.includes('modalMobileConnect'), 'Must handle modalMobileConnect');
  assert.ok(code.includes('window.setLogoProcessing(true)'), 'Must trigger logo processing on enhance');
  assert.ok(code.includes('/api/tunnel/status'), 'Must poll /api/tunnel/status');
});

it('slider.js handles touchmove with passive: false and cancelable check', () => {
  const code = fs.readFileSync(sliderJsPath, 'utf-8');
  assert.ok(code.includes("this.container.addEventListener('touchmove', onMove, { passive: false })"));
  assert.ok(code.includes('e.cancelable && e.touches'));
});

// 5. Backend Tunnel Service
console.log('\n--- 5. Python Cloudflare Tunnel Backend ---');
const tunnelPyPath = path.join(rootDir, 'scripts', 'tunnel_launcher.py');
it('scripts/tunnel_launcher.py exists and defines start_tunnel_background & get_tunnel_status', () => {
  assert.ok(fs.existsSync(tunnelPyPath), 'tunnel_launcher.py must exist');
  const code = fs.readFileSync(tunnelPyPath, 'utf-8');
  assert.ok(code.includes('def start_tunnel_background'));
  assert.ok(code.includes('def get_tunnel_status'));
  assert.ok(code.includes('def generate_qr_base64'));
});

console.log('\n================================================================');
console.log(`Results: ${passed}/${total} verification checks PASSED!`);
console.log('================================================================\n');

process.exit(0);
