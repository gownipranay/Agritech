# AgriAI — Multimodal Crop Health & Yield Advisory System

A full-stack system that helps farmers by combining computer-vision disease
detection, weather/soil risk forecasting, yield prediction, and a
retrieval-grounded treatment recommendation engine, summarized for the
farmer by an LLM.

## Current status

| Phase | Status |
|---|---|
| 1. Disease/pest detection | **Trained on real PlantVillage data** (15 classes: pepper, potato, tomato) |
| 2. Soil & weather integration | Not started |
| 3. Yield prediction | Not started |
| 4. Grounded treatment recommendation | **Implemented** — curated, ICAR-cited lookup (`/recommend-treatment`, `/crops`) |
| 5. Farmer-facing UI | **Implemented** — React (Vite) app in `frontend/`, deployable to Netlify |

### Crop coverage vs. what the vision model can actually detect

The real PlantVillage mirror only contains **pepper, potato, and tomato**, so
those are the only crops the CV model can diagnose from a photo. The app
still supports the other requested crops (chilli, corn/maize, paddy/rice,
toor dal, groundnut, cotton) — honestly labelled by capability:

| Crop | How it works in the app |
|---|---|
| Tomato, Potato, Bell pepper | Photo → trained CV model diagnosis |
| Chilli | Photo → routed to the bell-pepper model (same genus *Capsicum*), with a visible caveat |
| Corn, Paddy, Toor dal, Groundnut, Cotton | **Advisory only** — no trained detector yet, so the app shows the crop's common diseases + grounded management from the knowledge base instead of guessing from a photo |

Adding real photo-detection for the advisory-only crops requires labelled
datasets for them (e.g. the rice-leaf-disease, maize, cotton-disease Kaggle
sets); the training pipeline already generalizes to any `ImageFolder`, so it's
a data problem, not a code one. See [data/sources.md](data/sources.md).

## Architecture

```
backend/           FastAPI app
  app/main.py       app entrypoint, CORS, router registration
  app/routers/       predict_disease (live), predict_yield / recommend_treatment / advisory (stubs)
  app/services/      loads the trained model once, wraps ml/ inference code
  app/schemas/        pydantic response models
  app/core/config.py  settings (env vars, model checkpoint path)

ml/
  disease_classification/
    classes.py         disease/pest taxonomy
    model.py            MobileNetV3-Small classifier head
    dataset.py           ImageFolder + augmentation (blur/brightness jitter to
                          simulate real phone photos, not just clean lab images)
    make_sample_dataset.py  generates synthetic smoke-test images (see caveat above)
    train.py / eval.py / infer.py

data/
  sources.md          where every dataset/knowledge-base source comes from
  sample_images/       synthetic smoke-test dataset (generated, gitignored)
  knowledge_base/      Phase 4 disease->active-ingredient / NPK->fertilizer tables (not yet seeded)

frontend/            React + Tailwind (Phase 5, not yet built)
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# Generate the synthetic smoke-test dataset and train Phase 1
cd ml/disease_classification
python make_sample_dataset.py --out ../../data/sample_images --per-class 40
python train.py --data ../../data/sample_images --epochs 5
python eval.py --data ../../data/sample_images/val

# Run the API
cd ../../backend
uvicorn app.main:app --reload
# -> POST http://localhost:8000/predict-disease (multipart file upload)
# -> GET  http://localhost:8000/docs for interactive API docs
```

To train on real data instead: replace `data/sample_images/` with a
PlantVillage-derived `train/`+`val/` `ImageFolder` structure whose class
folder names match `ml/disease_classification/classes.py` exactly
(alphabetical order matters — `torchvision.datasets.ImageFolder` assigns
label indices alphabetically).

## Why this system doesn't let the LLM freely recommend chemicals

*(This is the design Phase 4 will implement — not yet built, documented
here because it's the core differentiator of this project.)*

LLMs hallucinate. For most tasks that's an inconvenience; for pesticide
brand names and dosages it's a safety problem — a wrong active ingredient
or a fabricated dosage on a real farmer's crop is not a "graceful failure."
So this system is designed so an LLM is **structurally incapable** of
inventing a chemical recommendation:

1. **Disease → active ingredient is a deterministic lookup**, not a model
   call. The table is seeded only from ICAR advisories, Krishi Vigyan Kendra
   guides, and university extension publications — never scraped blogs or
   forums — and every row carries a source citation. The LLM never sees
   this decision; it only sees the already-decided output.
2. **Active ingredient → product is retrieval, not generation.** Products
   are indexed from manufacturer pages and agri e-commerce listings ahead
   of time. A product is only ever surfaced because its stored active
   ingredient matches step 1's output — the LLM cannot introduce a brand
   name that isn't already in the matched retrieval result.
3. **Fertilizer amounts come from standard NPK-deficiency lookup tables**,
   which is established agronomy, not an LLM guess.
4. **The LLM's only job is summarization/translation** of the structured
   output from steps 1–3 into plain, friendly, localized language. Its
   system prompt explicitly forbids adding, substituting, or inferring any
   chemical name, brand, or dosage not present in the input — and the
   product database's source URL is always shown so a farmer (or their
   local KVK/dealer) can verify the label directly instead of trusting a
   paraphrase.

The UI will always show: *"Confirm with your local Krishi Vigyan Kendra or
agri-input dealer before use."*

Same philosophy applies to the vision model: below a confidence threshold,
the system shows *"uncertain — please consult local expert"* instead of
forcing a guess (implemented in `ml/disease_classification/infer.py`,
`CONFIDENCE_THRESHOLD = 0.60`), and false negative rate is tracked
separately from accuracy (see `eval.py`) because a missed disease is worse
than a false alarm.

## Deployment plan

- Frontend (Phase 5, React) → Netlify (free tier)
- Backend (FastAPI + PyTorch) → Render (free tier); Netlify cannot run this
  service, only static sites and lightweight serverless functions.

## Roadmap / next steps

1. Get real PlantVillage (+ ideally a real-world field-photo set like
   PlantDoc) into `data/`, retrain, and replace the smoke-test numbers in
   `model_card.md` with real ones.
2. Phase 2: OpenWeather integration + 7-day disease-spread risk model.
3. Phase 3: yield regression with confidence intervals from data.gov.in.
4. Phase 4: the grounded recommendation engine described above, with unit
   tests proving brand names only ever surface when their active
   ingredient matches the diagnosis.
5. Phase 5: React frontend once all API contracts are stable.
