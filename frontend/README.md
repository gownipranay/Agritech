# Frontend (deferred to Phase 5)

Per the project build order, the React + Tailwind frontend is built last,
once the backend API contracts (`/predict-disease`, `/predict-yield`,
`/recommend-treatment`, `/advisory`) are stable. Only `/predict-disease` is
implemented so far (Phase 1).

Planned stack: Vite + React + Tailwind CSS, deployed to Netlify. Backend
deployed separately (Render) since Netlify cannot run the FastAPI + PyTorch
service.
