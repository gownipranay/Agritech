# Model Card — AgriAI Disease Classifier (Phase 1)

**Status: trained on real PlantVillage data (15 classes). Lab-background
images only — see limitations before field deployment.**

## Summary

MobileNetV3-Small, ImageNet-pretrained backbone with the classifier head
replaced and fine-tuned, for 15-class crop disease/healthy classification
(pepper, potato, tomato). 8 epochs, AdamW, lr=1e-3, batch size 16. Best
checkpoint selected on validation accuracy.

## Training data

- **Source:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
  (emmarex/plantdisease Kaggle mirror), the Pepper/Potato/Tomato subset —
  real leaf photographs on lab (plain) backgrounds.
- **Preparation:** `ml/disease_classification/prepare_plantvillage.py`
  restructures the raw download into a train/val `ImageFolder` layout with a
  per-class cap (train ≤250, val ≤50) to keep CPU training tractable. The
  natural class imbalance below the cap is preserved, not equalized.
- **Classes (15):** Pepper bell (bacterial spot, healthy); Potato (early
  blight, late blight, healthy); Tomato (bacterial spot, early blight, late
  blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow
  leaf curl virus, mosaic virus, healthy). See
  `ml/disease_classification/classes.py`.
- **Split:** disjoint per-class train/val (held out from the same pool by
  `prepare_plantvillage.py`, seed=42).

## Evaluation metrics

Run on the **730-image real val split**, from
`ml/disease_classification/eval.py`.

- **Overall accuracy: 96.44%** (730 samples)
- **False negative rate (disease → predicted healthy): 0.33%** — only 2 of
  ~630 true-disease samples were misclassified as a healthy class. Tracked
  separately from accuracy because for a farmer a missed disease is worse
  than a false alarm.
- **Per-class accuracy:**

  | Class | n | Accuracy |
  |---|---|---|
  | Pepper bell — bacterial spot | 50 | 100% |
  | Pepper bell — healthy | 50 | 100% |
  | Potato — early blight | 50 | 98% |
  | Potato — late blight | 50 | 98% |
  | Potato — healthy | 30 | 90% |
  | Tomato — bacterial spot | 50 | 96% |
  | Tomato — early blight | 50 | 98% |
  | Tomato — late blight | 50 | 96% |
  | Tomato — leaf mold | 50 | 98% |
  | Tomato — septoria leaf spot | 50 | 92% |
  | Tomato — spider mites | 50 | 88% |
  | Tomato — target spot | 50 | 90% |
  | Tomato — yellow leaf curl virus | 50 | 100% |
  | Tomato — mosaic virus | 50 | 100% |
  | Tomato — healthy | 50 | 100% |

  Weakest classes are the visually subtle tomato conditions (spider mites
  88%, target spot 90%, septoria 92%) — expected, as these are the hardest
  to tell apart even for humans on a single leaf photo.

## Known limitations

1. **Lab-background images only.** PlantVillage leaves are photographed on
   plain backgrounds under even lighting. These numbers do **not** predict
   accuracy on cluttered, variably-lit real field/phone photos. A separate
   real-world set (e.g. PlantDoc) is still needed to honestly test
   generalization — see `data/sources.md`.
2. **3 crops only** (pepper, potato, tomato). Chilli is served via the
   pepper model (same genus *Capsicum*). Corn, paddy, toor dal, groundnut,
   and cotton have **no trained detector** — the app serves them as
   knowledge-base advisories, not photo diagnoses.
3. **Confidence threshold (0.60)** in `infer.py` is still a placeholder, not
   calibrated against a real false-negative/false-alarm tradeoff with
   agronomist input.
4. Class distribution is capped/imbalanced (train ≤250/class); rare
   real-world classes with few photos are not well exercised.

## Next steps

1. Add a real-world (non-lab) validation set and report accuracy broken out
   by lab vs. field images.
2. Add labelled datasets for corn, rice, cotton, groundnut, pigeon pea to
   extend photo-detection to those crops (the pipeline already generalizes
   to any `ImageFolder`).
3. Re-calibrate `CONFIDENCE_THRESHOLD` against real stakes with agronomist
   input on acceptable per-disease miss rates.
