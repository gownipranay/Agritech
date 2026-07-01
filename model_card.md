# Model Card — AgriAI Disease Classifier (Phase 1)

**Status: smoke-test model. Not fit for real-world use.**

## Summary

MobileNetV3-Small, ImageNet-pretrained backbone with the classifier head
replaced and fine-tuned, for 9-class crop disease/healthy classification.
10 epochs, AdamW, lr=1e-3, batch size 16.

## Training data

- **Source:** procedurally generated placeholder images
  (`ml/disease_classification/make_sample_dataset.py`) — colored leaf
  shapes with synthetic lesion blobs per class, with random blur/jitter.
  **Not real plant photographs.**
- **Why:** this environment had no Kaggle-authenticated access to download
  the real PlantVillage dataset. This model exists only to prove the
  train/eval/inference pipeline runs correctly end to end (data loading,
  augmentation, checkpointing, confidence thresholding, FNR reporting).
- **Classes (9):** tomato_healthy, tomato_early_blight, tomato_late_blight,
  tomato_leaf_mold, potato_healthy, potato_early_blight, potato_late_blight,
  corn_healthy, corn_common_rust.
- **Split:** ~75% train / 25% val per class, generated from disjoint random
  seeds (train and val images are generated separately, not held out from
  the same pool).

## Evaluation metrics

Run on the 90-image synthetic val split (10 images/class), from
`ml/disease_classification/eval.py`. These numbers describe how well the
model discriminates the synthetic dataset's base colors and blob patterns —
a much easier task than distinguishing real lesion textures on real leaves.
**Treat these as a pipeline-correctness check, not a disease-detection
benchmark.**

- **Overall accuracy: 73.3%** (66/90)
- **False negative rate (disease → predicted healthy): 0.0%** — in this
  synthetic run the model never mistook a diseased sample for healthy; all
  its errors were disease-vs-disease or healthy-vs-healthy confusions. This
  is tracked separately from accuracy because in a real deployment, missing
  a real disease is worse than a false alarm — a 0% FNR here should not be
  read as evidence the real model will have low FNR; it just confirms the
  metric is computed correctly.
- **Per-class accuracy:**

  | Class | Accuracy (n=10) |
  |---|---|
  | corn_common_rust | 100% |
  | corn_healthy | 100% |
  | tomato_healthy | 100% |
  | tomato_late_blight | 100% |
  | tomato_leaf_mold | 100% |
  | tomato_early_blight | 90% |
  | potato_early_blight | 40% |
  | potato_late_blight | 30% |
  | **potato_healthy** | **0%** |

  `potato_healthy` was confused almost entirely with `tomato_healthy` (the
  two synthetic "healthy" classes only differ by a small green-shade
  jitter — an easy confusion for a model with no real leaf-shape/texture
  signal to lean on). This is a useful, honest failure mode to see in a
  smoke test: it shows the pipeline correctly reports per-class weaknesses
  rather than masking them behind a single accuracy number, but it also
  illustrates why synthetic-data results cannot be trusted as a real
  benchmark.
- Confirmed manually via the live API: uploading a real val-set image
  through `POST /predict-disease` reproduces this — `tomato_late_blight`
  predicted at 98.9% confidence (correct), `potato_healthy` predicted as
  `tomato_healthy` at 61.8% confidence (incorrect, but above the 0.60
  threshold, so no "uncertain" flag was raised — a real sign the threshold
  needs real-data calibration, not just a placeholder value).

## Known limitations

1. **Not trained on real images.** Zero evidence this generalizes to actual
   leaf photos, let alone blurry/low-light phone photos from the field.
2. **No real-world (non-lab-background) validation set exists yet.**
   PlantVillage itself (once integrated) is lab-background; a separate
   real-world set (e.g. PlantDoc) is needed to honestly test
   generalization, per `data/sources.md`.
3. **9 classes only**, a small subset of the ~38-class full PlantVillage
   taxonomy.
4. **Confidence threshold (0.60)** was chosen arbitrarily as a placeholder,
   not calibrated against a real validation set with real-world stakes —
   confirmed in testing above, where a wrong prediction at 61.8% confidence
   passed the threshold uncaught.
5. Class balance is artificially uniform (equal synthetic images per
   class) — real-world class distributions are heavily imbalanced, which
   this smoke test does not exercise.

## Next steps to make this a real model

1. Download PlantVillage (`data/sources.md`) and a real-world field-photo
   set (e.g. PlantDoc), replace `data/sample_images/` with them.
2. Retrain, re-run `eval.py`, and replace the metrics section above with
   real numbers, broken out by lab-background vs. real-world val sets.
3. Re-calibrate `CONFIDENCE_THRESHOLD` in `ml/disease_classification/infer.py`
   against the real false-negative/false-alarm tradeoff, ideally with
   agronomist input on acceptable miss rates per disease.
4. Expand to the full disease/pest taxonomy needed for target crops/regions.
