/* AquaSense design tokens for Tailwind (CDN build). Every colour resolves to a CSS variable defined in theme.css,
   so one class (e.g. bg-primary/10) works in both the light 'chart paper' and dark 'night survey' themes.
   Token NAMES are kept stable: the page scripts build markup with these class names. */
const c = (name) => `rgb(var(--c-${name}) / <alpha-value>)`;

tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "background": c("background"),
        "surface": c("surface"),
        "surface-bright": c("surface-bright"),
        "surface-dim": c("surface-dim"),
        "surface-container-lowest": c("surface-container-lowest"),
        "surface-container-low": c("surface-container-low"),
        "surface-container": c("surface-container"),
        "surface-container-high": c("surface-container-high"),
        "surface-container-highest": c("surface-container-highest"),
        "surface-variant": c("surface-variant"),
        "on-surface": c("on-surface"),
        "on-surface-variant": c("on-surface-variant"),
        "on-background": c("on-background"),
        "outline": c("outline"),
        "outline-variant": c("outline-variant"),
        "inverse-surface": c("inverse-surface"),
        "inverse-on-surface": c("inverse-on-surface"),
        "inverse-primary": c("inverse-primary"),
        "surface-tint": c("surface-tint"),
        "primary": c("primary"),
        "on-primary": c("on-primary"),
        "primary-container": c("primary-container"),
        "on-primary-container": c("on-primary-container"),
        "primary-fixed": c("primary-fixed"),
        "primary-fixed-dim": c("primary-fixed-dim"),
        "on-primary-fixed": c("on-primary-fixed"),
        "on-primary-fixed-variant": c("on-primary-fixed-variant"),
        "secondary": c("secondary"),
        "on-secondary": c("on-secondary"),
        "secondary-container": c("secondary-container"),
        "on-secondary-container": c("on-secondary-container"),
        "secondary-fixed": c("secondary-fixed"),
        "secondary-fixed-dim": c("secondary-fixed-dim"),
        "on-secondary-fixed": c("on-secondary-fixed"),
        "on-secondary-fixed-variant": c("on-secondary-fixed-variant"),
        "tertiary": c("tertiary"),
        "on-tertiary": c("on-tertiary"),
        "tertiary-container": c("tertiary-container"),
        "on-tertiary-container": c("on-tertiary-container"),
        "tertiary-fixed": c("tertiary-fixed"),
        "tertiary-fixed-dim": c("tertiary-fixed-dim"),
        "on-tertiary-fixed": c("on-tertiary-fixed"),
        "on-tertiary-fixed-variant": c("on-tertiary-fixed-variant"),
        "error": c("error"),
        "on-error": c("on-error"),
        "error-container": c("error-container"),
        "on-error-container": c("on-error-container")
      },
      borderRadius: {
        DEFAULT: "2px", lg: "4px", xl: "6px", "2xl": "8px", "3xl": "12px", full: "9999px"
      },
      spacing: {
        "space-xs": "0.25rem", "space-sm": "0.5rem", "space-md": "1rem", "space-lg": "1.5rem", "space-xl": "2.5rem",
        gutter: "1.5rem", "gutter-mobile": "1rem", margin: "2.5rem", "margin-mobile": "1.25rem"
      },
      fontFamily: {
        /* display: Young Serif (single weight, so weights below are 400 to avoid faux-bold) */
        "display-lg": ["Young Serif", "Georgia", "serif"], "display-lg-mobile": ["Young Serif", "Georgia", "serif"],
        "headline-lg": ["Young Serif", "Georgia", "serif"], "headline-lg-mobile": ["Young Serif", "Georgia", "serif"],
        "headline-md": ["Young Serif", "Georgia", "serif"], "headline-sm": ["Young Serif", "Georgia", "serif"],
        "title-md": ["Familjen Grotesk", "sans-serif"], "body-lg": ["Familjen Grotesk", "sans-serif"],
        "body-md": ["Familjen Grotesk", "sans-serif"], "body-sm": ["Familjen Grotesk", "sans-serif"],
        "label-md": ["Familjen Grotesk", "sans-serif"], "label-sm": ["Familjen Grotesk", "sans-serif"],
        mono: ["Sometype Mono", "ui-monospace", "monospace"]
      },
      fontSize: {
        "display-lg": ["64px", { lineHeight: "66px", letterSpacing: "-0.02em", fontWeight: "400" }],
        "display-lg-mobile": ["42px", { lineHeight: "46px", letterSpacing: "-0.02em", fontWeight: "400" }],
        "headline-lg": ["40px", { lineHeight: "46px", letterSpacing: "-0.01em", fontWeight: "400" }],
        "headline-lg-mobile": ["30px", { lineHeight: "36px", letterSpacing: "-0.01em", fontWeight: "400" }],
        "headline-md": ["28px", { lineHeight: "34px", letterSpacing: "-0.005em", fontWeight: "400" }],
        "headline-sm": ["24px", { lineHeight: "30px", fontWeight: "400" }],
        "title-md": ["18px", { lineHeight: "26px", fontWeight: "600" }],
        "body-lg": ["20px", { lineHeight: "30px", fontWeight: "400" }],
        "body-md": ["17px", { lineHeight: "25px", fontWeight: "400" }],
        "body-sm": ["15px", { lineHeight: "21px", fontWeight: "400" }],
        "label-md": ["15px", { lineHeight: "20px", letterSpacing: "0.02em", fontWeight: "600" }],
        "label-sm": ["13.5px", { lineHeight: "18px", letterSpacing: "0.06em", fontWeight: "600" }]
      }
    }
  }
};
