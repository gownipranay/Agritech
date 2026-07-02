# AgriAI Frontend

React (Vite) single-page app. Farmer picks a crop, uploads a leaf photo, and
gets an AI diagnosis plus grounded, ICAR-cited treatment guidance.

## Local dev

```bash
cd frontend
npm install
cp .env.example .env.local   # point VITE_API_BASE at your backend
npm run dev                  # http://localhost:5173
```

The backend must be running (see repo root README) at the URL in
`VITE_API_BASE` (defaults to `http://localhost:8000`).

## Deploy to Netlify (free)

1. Push the repo to GitHub.
2. In Netlify: **Add new site → Import from Git**, pick the repo.
3. Netlify reads `frontend/netlify.toml` (base `frontend`, build `npm run
   build`, publish `dist`).
4. Set env var **`VITE_API_BASE`** to your deployed backend URL
   (e.g. `https://agriai-backend.onrender.com`).
5. Deploy.

## Crop support

| Support | Meaning |
|---|---|
| Photo detection | Trained CV model classifies the photo (tomato, potato, bell pepper) |
| Photo (via pepper) | Chilli — routed to the bell-pepper model (same genus), with a caveat |
| Advisory only | No trained detector yet (corn, paddy, toor dal, groundnut, cotton) — symptom-based guidance from the knowledge base |
