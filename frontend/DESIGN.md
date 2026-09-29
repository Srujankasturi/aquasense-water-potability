# AquaSense design system: "Hydrographic chart"

A water-quality screening tool should look like the instrument it is: a survey chart. Warm chart paper, ink, contour lines,
double-ruled plates, and one sharp signal accent for breaches. A dark **night survey** theme comes from the same tokens.

Chosen from three concepts (hydrographic chart / lab instrument / riso field report). The lab-instrument look (neon on dark)
was rejected as the most common "AI dashboard" aesthetic; riso was rejected because playful overprint weakens the contrast a
safety verdict needs.

## Files

| File | Role |
| :--- | :--- |
| `theme.css` | CSS variables (light `:root`, dark `.dark`), atmosphere, typography rules, components, motion |
| `tailwind-config.js` | Maps every Tailwind token name to `rgb(var(--c-…) / <alpha-value>)`, plus type scale, radii, spacing |
| `assets/contours.svg` | Generated isobath artwork, used as a CSS **mask** so its colour follows the theme |
| `app.js` | Shared logic. Emits markup with the token class names, so **token names must not change** |

Change the look by editing **values** in `theme.css` (both blocks) and `tailwind-config.js`; never rename tokens.

## Colour (47 tokens, both themes)

Space-separated RGB channels, so `bg-primary/10` works in both themes.

| Role | Light: chart paper | Dark: night survey | Use |
| :--- | :--- | :--- | :--- |
| `background` / `surface` | `#F2EBDC` | `#061821` | page ground |
| `surface-container-*` | `#FBF8F0` → `#D8CDB2` | `#04121A` → `#1B3B49` | sheets, tiles, table headers |
| `on-surface` | `#0B2A3C` (ink) | `#EAE3D0` (cream ink) | text |
| `on-surface-variant` | `#3D5563` | `#A9BEC5` | secondary text |
| `primary` | `#0F4C6B` survey navy | `#8FD0E0` chart aqua | actions, focus, emblem core |
| `secondary` | `#855A0A` ochre | `#EBC66B` gold | eyebrows, accents, the offset button shadow |
| `tertiary` | `#1E6B3E` | `#84D6A2` | within limit / potable |
| `error` | `#B93520` vermilion | `#FF9078` | breach / not potable |

Every text/background pair used by the UI is checked by script: **all pairs ≥ 4.5:1 in both themes** (dark secondary text 12.4:1, accents 12.9:1 after the legibility pass; dark sheets are near-opaque so contour lines never run behind text) (largest gap fixed:
`outline` 4.40 → 4.9). Colour never carries meaning alone: statuses also have text, icons or glyphs.

## Typography

| Role | Family | Notes |
| :--- | :--- | :--- |
| Display / headlines | **Young Serif** | single weight, so every serif token is fixed at 400 (no faux bold) |
| Text / UI | **Familjen Grotesk** | 400-700 |
| Measurements / eyebrows | **Sometype Mono** | tabular figures for every number the user reads |

Scale (px): display 64/42, headline 40/30/28/24, title 18, body 20/17/15, label 15/13.5. Any smaller arbitrary size in markup is lifted to 13px by `theme.css`.
Banned: Inter, Roboto, Arial, system fonts, Space Grotesk / Space Mono, Plus Jakarta Sans.

## Atmosphere

Layered, never a flat page colour: (1) soft teal / ochre washes, (2) contour lines (`body::after`, masked
`contours.svg`, drifting slowly), (3) paper grain (`html::after`, multiply in light, screen in dark). Surfaces are hairline-bordered
"sheets" with a paper shadow, not glass: all `backdrop-blur` is disabled. Key panels use a **double rule** (`.plate`, `.sheet-double`).

## Motion (CSS only)

One orchestrated load: header fade, sections rise in sequence, grid children stagger by 70ms (`--n`), plate rings draw in, the
verdict is "stamped". Everything is gated by `prefers-reduced-motion`. **Do not animate JS-populated containers**
(`#parameter-summary-grid`, `#rec-list`, the analysis tables and bars): a reveal that hides content until it runs can leave it invisible.

## Components

`.btn .btn-primary/.btn-ghost` (flat ink, offset "printed" shadow; 44px min height) · `.nav-link` (mono numerals, ochre underline) ·
`.icon-btn` · `.model-switch/.model-btn` (radio group, `is-active` + `aria-checked`) · `.preset-btn` · `.plate` (chart frame) ·
`.rec-card` (`rec-success/info/warning/danger`) · `.emblem` (verdict: nested contours + glyph) ·
`.gauge-track/-ok/-bad` (SVG strokes via classes, never hex).

## Editing principle: less on screen

Built for presenting. Every page shows only what the demo needs: Home = hero + live simulator; Predict = inputs and model
choice; Result = verdict, limits exceeded, probability, parameters, next steps; Analysis = class balance, real correlation,
feature importance, limits table, benchmark and confusion matrix. No decorative statistics, no duplicate navigation tiles.

## Content rules

Only claims the system can back: no "live", "real-time", "telemetry", "sensor", latency or compliance claims; no stock photos or
external images; numbers on the analysis page come from `/api/metadata` and `/api/analysis`, not from HTML.

## Accessibility checklist (met)

Contrast ≥ 4.5:1 both themes · visible `:focus-visible` ring · 44px touch targets on buttons and toggles · one `<h1>` per page ·
labelled radio group · SVG plate has a text alternative · reduced motion respected · no horizontal scroll at 390px.

## Known limits

Tailwind is loaded from its CDN (as before), and Material Symbols stay as the icon font (restyled to weight 300). A few markup
classes still carry blue-tinted `shadow-[…rgba(8,126,173,…)]` values; `theme.css` overrides them, so they have no visual effect.
