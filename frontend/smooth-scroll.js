/**
 * Smooth mouse-wheel scrolling.
 *
 * A classic mouse wheel scrolls in coarse 100-120 px steps. This eases the page toward the wheel's target each frame,
 * so scrolling glides instead of jumping. Left alone on purpose: trackpads and touch (already smooth natively),
 * keyboard, scrollbar dragging and anchor links (the browser handles those; we just stay in sync), pinch/zoom,
 * scrollable boxes inside the page, and anyone with "reduce motion" enabled.
 * Tune EASE (higher = snappier, lower = floatier).
 */
(() => {
  if (/jsdom/i.test(navigator.userAgent)) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const EASE = 0.12;                  // share of the remaining distance covered per 16.7 ms frame
  const LINE_PX = 40;                 // pixels per "line" for wheels that report lines

  let target = window.scrollY, current = window.scrollY, raf = 0, last = 0;
  const maxScroll = () => document.documentElement.scrollHeight - window.innerHeight;

  // Mouse wheels report multiples of 120 (legacy wheelDelta) or whole lines; trackpads report small, varying deltas.
  const isMouseWheel = (e) => e.deltaX === 0 && (e.deltaMode === 1 || (e.wheelDeltaY && e.wheelDeltaY % 120 === 0));

  // don't take over scrolling inside a box that can scroll itself (wide tables, text areas)
  const insideScroller = (el) => {
    for (let n = el; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (/(auto|scroll)/.test(s.overflowY) && n.scrollHeight > n.clientHeight + 1) return true;
    }
    return false;
  };

  function tick(now) {
    const dt = Math.min(64, now - last); last = now;
    current += (target - current) * (1 - Math.pow(1 - EASE, dt / 16.67));
    if (Math.abs(target - current) < 0.4) { window.scrollTo({ top: target, behavior: 'instant' }); raf = 0; return; }
    window.scrollTo({ top: current, behavior: 'instant' });          // 'instant' overrides the CSS smooth-scroll for anchors
    raf = requestAnimationFrame(tick);
  }

  window.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.defaultPrevented || !isMouseWheel(e) || insideScroller(e.target)) return;
    e.preventDefault();
    if (!raf) { current = target = window.scrollY; last = performance.now(); }
    target = Math.max(0, Math.min(maxScroll(), target + (e.deltaMode === 1 ? e.deltaY * LINE_PX : e.deltaY)));
    if (!raf) raf = requestAnimationFrame(tick);
  }, { passive: false });

  // scrolling done by something else (keyboard, scrollbar, anchor jump): keep our numbers in step
  window.addEventListener('scroll', () => { if (!raf) current = target = window.scrollY; }, { passive: true });
})();
