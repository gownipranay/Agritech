import { useEffect, useState } from "react";
import { fetchCrops, fetchFertilizer, predictDisease, recommendTreatment } from "./api.js";
import {
  CROP_EMOJI,
  FALLBACK_CROPS,
  SUPPORT_LABEL,
} from "./cropsFallback.js";

function prettyDisease(key) {
  if (!key) return "";
  return key
    .replace(/_+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\s+/g, " ")
    .trim();
}

export default function App() {
  const [crops, setCrops] = useState(FALLBACK_CROPS);
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [treatment, setTreatment] = useState(null);
  const [advisoryList, setAdvisoryList] = useState(null);
  const [fertilizer, setFertilizer] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCrops()
      .then(setCrops)
      .catch(() => {}); // keep fallback list
  }, []);

  function resetResults() {
    setPrediction(null);
    setTreatment(null);
    setAdvisoryList(null);
    setError(null);
  }

  function selectCrop(crop) {
    setSelected(crop);
    resetResults();
    setFertilizer(null);
    fetchFertilizer(crop.id)
      .then(setFertilizer)
      .catch(() => setFertilizer(null));
  }

  function onFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    resetResults();
  }

  async function analyze() {
    if (!selected) return;
    resetResults();
    setLoading(true);
    try {
      if (selected.vision_support === "advisory_only") {
        // No trained detector — show curated diseases for this crop.
        const diseases = selected.common_diseases || [];
        const entries = await Promise.all(
          diseases.map(async (k) => ({ key: k, data: await recommendTreatment(k) }))
        );
        setAdvisoryList(entries.filter((e) => e.data));
      } else {
        if (!file) {
          setError("Please upload a leaf photo first.");
          setLoading(false);
          return;
        }
        const pred = await predictDisease(file, selected.id);
        setPrediction(pred);
        if (pred.is_confident && pred.predicted_class) {
          const t = await recommendTreatment(pred.predicted_class);
          setTreatment(t);
        }
      }
    } catch (err) {
      setError(err.message || "Something went wrong. The backend may be waking up — try again in ~30s.");
    } finally {
      setLoading(false);
    }
  }

  const isAdvisory = selected?.vision_support === "advisory_only";

  return (
    <>
      <header className="header">
        <h1>🌱 AgriAI — Crop Health Advisory</h1>
        <p>Upload a leaf photo for an AI diagnosis, plus grounded, ICAR-cited treatment guidance.</p>
      </header>

      <main className="container">
        {/* Step 1: crop */}
        <section className="card">
          <h2 className="section-title">1. Choose your crop</h2>
          <div className="crop-grid">
            {crops.map((c) => {
              const badge = SUPPORT_LABEL[c.vision_support] || SUPPORT_LABEL.advisory_only;
              return (
                <button
                  key={c.id}
                  className={`crop-btn ${selected?.id === c.id ? "active" : ""}`}
                  onClick={() => selectCrop(c)}
                >
                  <span className="crop-emoji">{CROP_EMOJI[c.id] || "🌿"}</span>
                  {c.name}
                  <span className={`badge ${badge.cls}`}>{badge.text}</span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Step 2: photo / advisory */}
        {selected && (
          <section className="card">
            {isAdvisory ? (
              <>
                <h2 className="section-title">2. {selected.name} — advisory</h2>
                <div className="notice warn">
                  No trained photo-detector for {selected.name} yet, so the app can't diagnose
                  from an image. Below are the common diseases and grounded management options
                  for this crop. Match your field symptoms to one.
                </div>
                <button className="btn" onClick={analyze} disabled={loading}>
                  {loading ? <span className="spinner" /> : `Show ${selected.name} diseases`}
                </button>
              </>
            ) : (
              <>
                <h2 className="section-title">2. Upload a leaf photo</h2>
                {selected.vision_note && (
                  <div className="notice info">{selected.vision_note}</div>
                )}
                {selected.vision_support === "vision_proxy" && (
                  <div className="notice warn">
                    Chilli uses the bell-pepper detector (same Capsicum genus). Leaf-spot detection
                    transfers reasonably; confirm chilli-specific issues (thrips, leaf curl, dieback)
                    with a local expert.
                  </div>
                )}
                <label className="dropzone">
                  {previewUrl ? (
                    <img src={previewUrl} alt="preview" className="preview-img" />
                  ) : (
                    <div>📷 Tap to upload or take a photo of the affected leaf</div>
                  )}
                  <input type="file" accept="image/*" capture="environment" onChange={onFile} />
                </label>
                <button className="btn" onClick={analyze} disabled={loading || !file}>
                  {loading ? <span className="spinner" /> : "Diagnose"}
                </button>
              </>
            )}
          </section>
        )}

        {/* Fertilizer plan (per selected crop) */}
        {selected && fertilizer && <FertilizerCard f={fertilizer} cropName={selected.name} />}

        {error && <div className="card"><div className="notice error">{error}</div></div>}

        {/* Prediction result */}
        {prediction && (
          <section className="card">
            <h2 className="section-title">Diagnosis</h2>
            {prediction.is_confident ? (
              <>
                <div className="result-diagnosis">{prettyDisease(prediction.predicted_class)}</div>
                <div className="confidence-bar">
                  <div className="confidence-fill" style={{ width: `${prediction.confidence * 100}%` }} />
                </div>
                <div className="top5">Confidence: {(prediction.confidence * 100).toFixed(1)}%</div>
              </>
            ) : (
              <div className="notice warn">
                {prediction.message || "Uncertain — please consult a local expert (KVK or agri-input dealer)."}
              </div>
            )}
            {prediction.top_5?.length > 0 && (
              <div className="top5">
                Other possibilities:{" "}
                {prediction.top_5.slice(0, 3).map((t) => `${prettyDisease(t.class)} (${(t.confidence * 100).toFixed(0)}%)`).join(", ")}
              </div>
            )}
            <div className="source">Model: {prediction.model_version}</div>
          </section>
        )}

        {/* Single treatment */}
        {treatment && <TreatmentCard t={treatment} title="Recommended management" />}

        {/* Advisory list */}
        {advisoryList && advisoryList.length > 0 && (
          <section className="card">
            <h2 className="section-title">Common diseases &amp; management</h2>
            {advisoryList.map(({ key, data }) => (
              <TreatmentCard key={key} t={data} embedded />
            ))}
          </section>
        )}
      </main>

      <footer className="footer">
        AgriAI · Diagnoses are decision-support, not a substitute for a Krishi Vigyan Kendra visit.
      </footer>
    </>
  );
}

function TreatmentCard({ t, title, embedded }) {
  const body = (
    <>
      <div className="result-diagnosis" style={{ fontSize: "1.1rem" }}>{t.label}</div>
      <div className="top5" style={{ marginBottom: 8 }}>Type: {t.type}</div>

      <div className="kv">
        <div className="label">Active ingredients</div>
        {t.active_ingredients.map((a, i) => <span className="chip" key={i}>{a}</span>)}
      </div>

      {t.indicative_dosage && (
        <div className="kv">
          <div className="label">Indicative dosage</div>
          <div>{t.indicative_dosage}</div>
        </div>
      )}

      {t.cultural_controls?.length > 0 && (
        <div className="kv">
          <div className="label">Cultural controls</div>
          <ul className="tight">{t.cultural_controls.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      )}

      <div className="source">Source: {t.source}</div>

      {t.products?.length > 0 && (
        <div className="kv">
          <div className="label">Branded products (matched by active ingredient)</div>
          <div className="product-list">
            {t.products.map((p, i) => (
              <a className="product" key={i} href={p.source} target="_blank" rel="noreferrer">
                <span className="product-company">{p.company}</span>
                <span className="product-name">{p.name}</span>
                <span className="product-ai">{p.active_ingredient}</span>
              </a>
            ))}
          </div>
          {t.products_disclaimer && <div className="source">{t.products_disclaimer}</div>}
        </div>
      )}

      {t.disclaimer && <div className="disclaimer">⚠️ {t.disclaimer}</div>}
    </>
  );

  if (embedded) {
    return <div style={{ borderTop: "1px solid var(--gray-200)", paddingTop: 14, marginTop: 14 }}>{body}</div>;
  }
  return (
    <section className="card">
      {title && <h2 className="section-title">{title}</h2>}
      {body}
    </section>
  );
}

function FertilizerCard({ f, cropName }) {
  return (
    <section className="card">
      <h2 className="section-title">🌾 Fertilizer plan for {cropName}</h2>

      <div className="kv">
        <div className="label">Recommended dose (per hectare)</div>
        <div className="result-diagnosis" style={{ fontSize: "1.05rem" }}>{f.npk_kg_per_ha}</div>
      </div>

      {f.schedule?.length > 0 && (
        <div className="kv">
          <div className="label">Application schedule</div>
          <ul className="tight">
            {f.schedule.map((s, i) => (
              <li key={i}><b>{s.stage}:</b> {s.apply} <span className="product-ai">({s.as})</span></li>
            ))}
          </ul>
        </div>
      )}

      {f.straight_fertilizers?.length > 0 && (
        <div className="kv">
          <div className="label">Fertilizers to use</div>
          {f.straight_fertilizers.map((s, i) => <span className="chip" key={i}>{s}</span>)}
        </div>
      )}

      {f.micronutrients && (
        <div className="kv">
          <div className="label">Micronutrients / notes</div>
          <div>{f.micronutrients}</div>
        </div>
      )}

      {f.branded_products?.length > 0 && (
        <div className="kv">
          <div className="label">Branded fertilizer products</div>
          <div className="product-list">
            {f.branded_products.map((p, i) => (
              <a className="product" key={i} href={p.source} target="_blank" rel="noreferrer">
                <span className="product-company">{p.company}</span>
                <span className="product-name">{p.name}</span>
                <span className="product-ai">{p.grade}</span>
              </a>
            ))}
          </div>
        </div>
      )}

      <div className="source">Source: {f.source}</div>
      {f.disclaimer && <div className="disclaimer">⚠️ {f.disclaimer}</div>}
    </section>
  );
}
