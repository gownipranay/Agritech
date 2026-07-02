# Model Card — AgriAI Disease Classifier (Phase 1)

**Status: trained on real data, 29 classes across 7 crops. Lab/curated
images — see limitations before field deployment.**

## Summary

MobileNetV3-Small, ImageNet-pretrained backbone with the classifier head
replaced and fine-tuned, for 29-class crop disease/healthy classification
across pepper, potato, tomato, maize, rice (paddy), cotton, and chilli.
8 epochs, AdamW, lr=1e-3, batch size 16. Best checkpoint selected on
validation accuracy.

## Training data

Merged from five public Kaggle datasets (full provenance in
`data/sources.md`), restructured into one train/val `ImageFolder` by
`prepare_plantvillage.py` + `prepare_extra_crops.py`, with a per-class cap
(train ≤250, val ≤50) to keep CPU training tractable:

- **PlantVillage** (emmarex/plantdisease): pepper, potato, tomato — 15 classes
- **Corn/Maize**: common rust, gray leaf spot, northern leaf blight, healthy
- **Rice/Paddy**: bacterial leaf blight, blast, brown spot, tungro
- **Cotton**: bacterial blight, leaf curl virus, fusarium wilt, healthy
- **Chilli**: leaf curl, healthy (only these two available in the dataset)

Groundnut was evaluated but excluded — its dataset is object-detection
format (no disease class folders), so groundnut remains advisory-only.

## Evaluation metrics

Run on the **1,430-image real val split**, from `eval.py`.

- **Overall accuracy: 93.36%** (1,430 samples)
- **False negative rate (disease → predicted healthy): 1.3%** — tracked
  separately from accuracy because for a farmer a missed disease is worse
  than a false alarm.
- **Per-class accuracy (n=50 each unless noted):**

  | Class | Acc | | Class | Acc |
  |---|---|---|---|---|
  | Pepper — bacterial spot | 100% | | Tomato — mosaic virus | 100% |
  | Pepper — healthy | 100% | | Tomato — healthy | 90% |
  | Potato — early blight | 100% | | Chilli — leaf curl | 98% |
  | Potato — late blight | 98% | | Chilli — healthy | 92% |
  | Potato — healthy (n=30) | 100% | | Corn — common rust | 96% |
  | Tomato — bacterial spot | 94% | | Corn — gray leaf spot | 84% |
  | Tomato — early blight | 92% | | Corn — healthy | 100% |
  | Tomato — late blight | 96% | | Corn — northern leaf blight | 86% |
  | Tomato — leaf mold | 94% | | Cotton — bacterial blight | 84% |
  | Tomato — septoria | 90% | | Cotton — fusarium wilt | 98% |
  | Tomato — spider mites | 82% | | Cotton — healthy | 98% |
  | Tomato — target spot | 54% | | Cotton — leaf curl virus | 96% |
  | Tomato — yellow leaf curl | 100% | | Paddy — bacterial leaf blight | 100% |
  | | | | Paddy — blast | 90% |
  | | | | Paddy — brown spot | 100% |
  | | | | Paddy — tungro | 98% |

  Weakest class is **Tomato target spot (54%)** — it is visually very close to
  other tomato leaf spots (septoria, bacterial spot, early blight) and is the
  main confusion sink. Corn gray leaf spot (84%), corn northern leaf blight
  (86%), and cotton bacterial blight (84%) are the next weakest, all subtle
  lesion-texture classes. These are honest, expected failure modes surfaced
  rather than hidden behind the headline accuracy.

## Known limitations

1. **Curated/lab-style images.** The source datasets are mostly clean,
   single-leaf photos. These numbers do **not** predict accuracy on
   cluttered, variably-lit real field/phone photos. A real-world holdout set
   is still needed for an honest field benchmark.
2. **Uneven per-crop disease coverage.** Chilli has only leaf-curl vs.
   healthy; rice has no "healthy" class in its dataset; cotton/corn cover a
   handful of diseases each. The app labels each crop's covered diseases and
   falls back to knowledge-base advisory for the rest.
3. **Toor dal and groundnut have no detector** (advisory-only).
4. **Confidence threshold (0.60)** in `infer.py` is still a placeholder, not
   calibrated against a real false-negative/false-alarm tradeoff.

## Next steps

1. Add a real-world (non-lab) validation set; report lab vs. field accuracy.
2. Source classification datasets for groundnut and toor dal; add more
   chilli diseases (anthracnose, powdery mildew) and a rice healthy class.
3. Re-calibrate `CONFIDENCE_THRESHOLD` with agronomist input on acceptable
   per-disease miss rates.
