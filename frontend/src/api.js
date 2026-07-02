// Base URL of the FastAPI backend. In production set VITE_API_BASE in
// Netlify env vars to your Render backend URL, e.g.
// https://agriai-backend.onrender.com
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchCrops() {
  const res = await fetch(`${API_BASE}/crops`);
  if (!res.ok) throw new Error(`Failed to load crops (${res.status})`);
  return (await res.json()).crops;
}

export async function predictDisease(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict-disease`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let detail = `Prediction failed (${res.status})`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchFertilizer(cropId) {
  const res = await fetch(`${API_BASE}/crops/${cropId}/fertilizer`);
  if (!res.ok) return null;
  return res.json();
}

export async function recommendTreatment(diseaseKey) {
  const res = await fetch(`${API_BASE}/recommend-treatment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ disease_key: diseaseKey }),
  });
  if (!res.ok) return null; // healthy / no entry
  return res.json();
}
