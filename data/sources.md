# Data Sources

## Phase 1 — Disease/pest image classification

**Current status:** The shipped checkpoint in `ml/checkpoints/disease_model.pt`
is trained on the **real PlantVillage** Pepper/Potato/Tomato subset (15
classes), achieving 96.4% val accuracy / 0.33% FNR on a 730-image held-out
split (see `model_card.md`). The synthetic generator
(`make_sample_dataset.py`) is kept only for pipeline smoke tests.

**Training data used (29 classes total):**
- [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
  (emmarex/plantdisease) — pepper/potato/tomato, 15 classes. Restructured by
  `prepare_plantvillage.py`.
- [Corn/Maize Leaf Disease](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
  — common rust, gray leaf spot, blight, healthy.
- [Rice Leaf Disease Images](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)
  — bacterial blight, blast, brown spot, tungro.
- [Cotton Leaf Disease](https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset)
  — bacterial blight, curl virus, fusarium wilt, healthy.
- [Chilli Leaf Disease](https://www.kaggle.com/datasets/sohanmirza/chilli-leaf-disease)
  — leaf curl vs. healthy (only these two classes available).

  Maize/rice/cotton/chilli are merged into `data/plantvillage/` by
  `prepare_extra_crops.py`. Raw downloads live in `data/crop_raw/` and
  `data/plantvillage_raw/`; both (and `data/plantvillage/`) are gitignored
  (large), regenerate via the `kaggle datasets download` refs above.

- **Groundnut** was evaluated but excluded: the available
  [groundnut dataset](https://www.kaggle.com/datasets/muhammadazeemabbas/groundnut-leaves-dataset)
  is object-detection format (Pascal-VOC XML, no disease class folders), so
  groundnut stays advisory-only until a classification dataset is sourced.
- A supplementary set of real-world, non-lab-background field photos
  (cluttered backgrounds, variable lighting) is needed to honestly evaluate
  generalization beyond PlantVillage's clean studio images — e.g.
  [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (~2,600 images,
  13 crops, real-world photos) is a good candidate. Not yet integrated.

## Phase 3 — Yield prediction (not yet built)

- Planned source: [data.gov.in](https://www.data.gov.in) agriculture
  datasets (district-wise crop yield, area, production).

## Phase 4 — Treatment recommendation knowledge base

Implemented in `data/knowledge_base/treatments.json` (disease → active
ingredient / dosage / cultural controls, each with a source) and
`data/knowledge_base/crops.json` (crop catalogue + vision-support status).
Every `disease -> active_ingredient` row cites one of:
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
