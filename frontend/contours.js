/**
 * AquaSense living contour map.
 *
 * Every contour line of assets/contours-orbit.svg (196 of them) is drawn on a canvas and moves on its own:
 * it drifts back and forth along ITS OWN random direction, sways (rotates) a few degrees around its own centre and
 * slowly breathes (scales). Underneath, the whole map also revolves very slowly, and a fainter copy of the map
 * revolves the other way, so the lines slide past one another like a real chart of moving water.
 *
 * Cheap on purpose: Path2D objects are built once, drawing is capped at 30 fps, the loop stops when the tab is hidden,
 * and with "reduce motion" enabled the map is drawn once and stays still. Colours come from the theme variables, so
 * light and dark work without any change here. Tune the numbers in CFG.
 */
(() => {
  if (/jsdom/i.test(navigator.userAgent)) return;                       // no canvas in the test DOM

  const CFG = {
    fps: 20,
    map: 2400,                                                         // size of the artwork (square)
    drift: [16, 48],           // px each line travels away from its rest position (per side): a few line-gaps, so rings stay nested
    driftPeriod: [40, 100],    // seconds for one there-and-back
    sway: [1, 3.5],            // degrees each line rotates about its own centre
    swayPeriod: [50, 130],
    breathe: [0.006, 0.02],   // fraction each line grows / shrinks
    breathePeriod: [40, 100],
    revolution: 900,           // seconds per revolution of the whole map (clockwise)
    revolutionBack: 1500,      // fainter copy, counter-clockwise
    backScale: 1.35,
    backAlpha: 0.55,
  };

  const canvas = document.createElement('canvas');
  canvas.id = 'contour-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  document.body.prepend(canvas);

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let lines = [], vw = 0, vh = 0, dpr = 1, k = 1, small = false, colour = { rgb: '11,42,60', a: 0.13 };
  let drawMs = 0, drawCount = 0;

  // deterministic random numbers: every page shows the same starting map
  const mulberry = (seed) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const between = (rand, [lo, hi]) => lo + (hi - lo) * rand();

  function readColour() {
    const cs = getComputedStyle(document.documentElement);
    const rgb = cs.getPropertyValue('--contour-rgb').trim().split(/\s+/).join(',');
    const a = parseFloat(cs.getPropertyValue('--contour-alpha')) || 0.13;
    colour = { rgb: rgb || '11,42,60', a };
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.25);
    vw = window.innerWidth; vh = window.innerHeight;
    canvas.width = Math.round(vw * dpr); canvas.height = Math.round(vh * dpr);
    k = Math.max(CFG.map, 1.8 * Math.max(vw, vh)) / CFG.map;             // oversized so rotation never shows a corner
    small = vw < 700;
  }

  // 2x3 affine matrices [a, b, c, d, e, f]
  const mul = (m, n) => [m[0]*n[0] + m[2]*n[1], m[1]*n[0] + m[3]*n[1], m[0]*n[2] + m[2]*n[3], m[1]*n[2] + m[3]*n[3],
                         m[0]*n[4] + m[2]*n[5] + m[4], m[1]*n[4] + m[3]*n[5] + m[5]];
  const T = (x, y) => [1, 0, 0, 1, x, y];
  const R = (r) => { const c = Math.cos(r), s = Math.sin(r); return [c, s, -s, c, 0, 0]; };
  const S = (s) => [s, 0, 0, s, 0, 0];

  function pass(t, scale, revolutionSeconds, direction, alphaFactor, phaseShift) {
    const angle = direction * (t / revolutionSeconds) * Math.PI * 2;
    const world = mul(mul(mul(mul([dpr, 0, 0, dpr, 0, 0], T(vw / 2, vh / 2)), R(angle)), S(k * scale)), T(-CFG.map / 2, -CFG.map / 2));
    const TAU = Math.PI * 2;
    for (const l of lines) {
      const tt = t + phaseShift;
      const along = Math.sin(TAU * tt / l.dP + l.dPh);
      const ox = Math.cos(l.dir) * l.drift * along, oy = Math.sin(l.dir) * l.drift * along;
      const rot = (l.sway * Math.PI / 180) * Math.sin(TAU * tt / l.sP + l.sPh);
      const sc = 1 + l.breathe * Math.sin(TAU * tt / l.bP + l.bPh);
      const m = mul(world, mul(mul(T(l.cx + ox, l.cy + oy), R(rot)), S(sc)));
      ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
      ctx.lineWidth = (l.major ? 2.2 : 1.15) / (k * scale * sc);
      ctx.strokeStyle = `rgba(${colour.rgb},${((l.major ? 1 : 0.85) * colour.a * alphaFactor).toFixed(3)})`;
      ctx.stroke(l.path);
    }
  }

  function draw(t) {
    const t0 = performance.now();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    pass(t, 1, CFG.revolution, 1, 1, 0);                                             // main map, clockwise
    if (!small) pass(t, CFG.backScale, CFG.revolutionBack, -1, CFG.backAlpha, 41.7); // fainter copy, the other way
    drawMs += performance.now() - t0; drawCount++;
  }

  function build(svgText) {
    const doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
    const rand = mulberry(2026);
    const groups = [...doc.querySelectorAll('path')];                                // [0] = minor lines, [1] = major (every 5th level)
    groups.forEach((g, gi) => {
      (g.getAttribute('d') || '').split('M').filter(Boolean).forEach((chunk) => {
        const nums = chunk.match(/-?\d+(?:\.\d+)?/g);
        if (!nums || nums.length < 8) return;
        const pts = []; let sx = 0, sy = 0;
        for (let i = 0; i + 1 < nums.length; i += 2) { const x = +nums[i], y = +nums[i + 1]; pts.push(x, y); sx += x; sy += y; }
        const n = pts.length / 2, cx = sx / n, cy = sy / n;
        const path = new Path2D();
        path.moveTo(pts[0] - cx, pts[1] - cy);
        for (let i = 2; i < pts.length; i += 2) path.lineTo(pts[i] - cx, pts[i + 1] - cy);
        if (Math.hypot(pts[0] - pts[pts.length - 2], pts[1] - pts[pts.length - 1]) < 3) path.closePath();
        lines.push({
          path, cx, cy, major: gi === 1,
          dir: rand() * Math.PI * 2,                                                 // this line's own direction
          drift: between(rand, CFG.drift), dP: between(rand, CFG.driftPeriod), dPh: rand() * 6.283,
          sway: between(rand, CFG.sway), sP: between(rand, CFG.swayPeriod), sPh: rand() * 6.283,
          breathe: between(rand, CFG.breathe), bP: between(rand, CFG.breathePeriod), bPh: rand() * 6.283,
        });
      });
    });
    start();
  }

  function start() {
    readColour(); resize();
    draw(0);
    document.documentElement.classList.add('contours-live');                         // fades the static fallback out
    if (reduceMotion) return;
    let last = 0, lastScroll = -1e9;
    window.addEventListener('scroll', () => { lastScroll = performance.now(); }, { passive: true });
    const frame = (now) => {
      requestAnimationFrame(frame);
      if (now - last < 1000 / (small ? 24 : CFG.fps)) return;
      if (now - lastScroll < 180) return;                          // no extra work while the page is being scrolled
      last = now; draw(now / 1000);
    };
    requestAnimationFrame(frame);
    new MutationObserver(readColour).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });   // theme toggle
  }

  window.addEventListener('resize', () => { resize(); if (lines.length) draw(performance.now() / 1000); });
  window.AquaContours = { get lineCount() { return lines.length; }, get avgDrawMs() { return drawCount ? drawMs / drawCount : 0; },
    sample(i, t) { const l = lines[i]; const TAU = 6.283185307; const a = Math.sin(TAU * t / l.dP + l.dPh);
      return { dirDeg: +(l.dir * 180 / Math.PI).toFixed(1), driftPx: l.drift | 0, offset: [Math.round(Math.cos(l.dir) * l.drift * a), Math.round(Math.sin(l.dir) * l.drift * a)] }; } };

  fetch('assets/contours-orbit.svg').then((r) => r.text()).then(build).catch(() => { /* static fallback layer stays */ });
})();
