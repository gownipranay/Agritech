# Data Sources

## Phase 1 — Disease/pest image classification

**Current status (smoke test):** The checkpoint in `ml/checkpoints/` (if
present) was trained on `data/sample_images/`, a set of **procedurally
generated placeholder images** (`ml/disease_classification/make_sample_dataset.py`),
not real plant photos. This exists only to prove the training/eval/inference
pipeline runs correctly end to end. Its accuracy numbers are meaningless as
a measure of real disease-detection performance — do not cite them outside
this repo.

**Planned real training data (not yet downloaded):**
- [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) —
  ~54,000 lab-background leaf images across 38 crop/disease classes.
  Requires a free Kaggle account + API token to download
  (`kaggle datasets download -d emmarex/plantdisease`).
- A supplementary set of real-world, non-lab-background field photos
  (cluttered backgrounds, variable lighting) is needed to honestly evaluate
  generalization beyond PlantVillage's clean studio images — e.g.
  [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (~2,600 images,
  13 crops, real-world photos) is a good candidate. Not yet integrated.

## Phase 3 — Yield prediction (not yet built)

- Planned source: [data.gov.in](https://www.data.gov.in) agriculture
  datasets (district-wise crop yield, area, production).

## Phase 4 — Treatment recommendation knowledge base (not yet built)

No knowledge base rows exist yet. When Phase 4 is built, every
`disease -> active_ingredient` row must cite one of:
- ICAR (Indian Council of Agricultural Research) advisories
- Krishi Vigyan Kendra (KVK) extension publications
- State agricultural university extension guides

Scraped blogs, forums, or un-sourced listings are explicitly disallowed as
sources for this table (see README "Why this system doesn't let the LLM
freely recommend chemicals").

Product data (brand -> active ingredient -> label dosage) will be sourced
from manufacturer pages (Syngenta, Adama, UPL, Bayer Crop Science) and agri
e-commerce listings (BigHaat, AgroStar), each with a stored source URL.

## Phase 2 — Weather

- Planned source: [OpenWeather API](https://openweathermap.org/api) (free tier).
