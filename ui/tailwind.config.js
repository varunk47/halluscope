/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    screens: {
      sm: "640px",
      md: "900px",
      lg: "1100px",
      xl: "1280px",
    },
    extend: {
      colors: {
        ink: "#0a0e14",
        panel: "#111821",
        "panel-2": "#161e29",
        line: "#1e2833",
        "line-2": "#2a3644",
        text: "#e8eef4",
        muted: "#8a97a6",
        dim: "#5a6774",
        safe: "#22d3ee",
        risk: "#f97316",
        "risk-hot": "#ef4444",
        incons: "#a78bfa",
      },
      fontFamily: {
        sans: ["Instrument Sans", "system-ui", "Segoe UI", "sans-serif"],
        display: ["Bricolage Grotesque", "Instrument Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
      },
      transitionDuration: {
        DEFAULT: "180ms",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.25), 0 0 24px rgba(34,211,238,0.12)",
        "glow-risk": "0 0 0 1px rgba(249,115,22,0.35), 0 0 28px rgba(239,68,68,0.18)",
        panel: "0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35)",
        readout: "0 1px 0 rgba(255,255,255,0.04) inset, 0 24px 60px -20px rgba(0,0,0,0.7)",
      },
      backgroundImage: {
        "ribbon-scale":
          "linear-gradient(90deg, #22d3ee 0%, #22d3ee 35%, #f97316 62%, #ef4444 100%)",
      },
      keyframes: {
        pulseDot: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        rise: {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        pulseDot: "pulseDot 1.6s ease-in-out infinite",
        rise: "rise 200ms ease-out both",
      },
    },
  },
  plugins: [],
};
