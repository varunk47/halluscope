# HalluScope UI

Front end for HalluScope v2: internal-state probes for underspecified requests.
Vite, React 18, TypeScript, Tailwind CSS v3, Recharts, react-router-dom.

## Run

```bash
cd ui
npm install
npm run dev      # http://localhost:5173, proxies /api to http://127.0.0.1:8000
npm run build    # writes ui/dist, served by the FastAPI app when present
npm run preview  # serve the built bundle on :5173
```

Start the backend first, for example `uvicorn server.app:app --port 8000` from the repo root.
The Vite dev server proxies every `/api` request to it, so the UI code only ever uses relative URLs.

## Routes

| path          | screen                                                                                  |
| ------------- | --------------------------------------------------------------------------------------- |
| `/`           | Live dialogue: build turns, score every layer, stream an ask-or-answer reply over SSE  |
| `/review`     | Dataset review with filters, inline edit, and `a` / `r` / `j` / `k` / `e` shortcuts     |
| `/results`    | Results explorer for probe, uq, behavior and loop result files                          |
| `/provenance` | Versions, hardware, seeds, split fractions, judge aliases and LLM cost                  |

## Layout

```
ui/
  index.html              fonts (Inter, JetBrains Mono) and the root mount
  vite.config.ts          dev proxy for /api, build to dist
  tailwind.config.js      palette, fonts, screens (md = 900px)
  src/
    api.ts                typed fetch helpers, ApiError, SSE-over-POST reader
    App.tsx               routes and layout
    styles.css            Tailwind layers plus component classes (.panel, .label, .btn-*)
    pages/                Live, Review, Results, Provenance
    components/           TopBar, Toast, Primitives, RiskRibbon, DialogueBuilder
    components/charts/    Recharts theme and the Live page charts
    components/results/   ProbeView, UqView, BehaviorView, LoopView
```

## Notes

- The reviewer name on `/review` is kept in `localStorage` under `halluscope.reviewer` (default `varun`).
- `POST /api/chat` streams `text/event-stream`. `EventSource` cannot POST, so `chatStream` in
  `api.ts` reads the body with a `ReadableStream` reader and parses `event:` / `data:` frames by hand.
- API errors (`{detail}` JSON) surface as toasts in the bottom right corner.
