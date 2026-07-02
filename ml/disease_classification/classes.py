"""
Disease/pest class taxonomy.

Matches the 15 classes present in the emmarex/plantdisease Kaggle mirror of
PlantVillage (Pepper, Potato, Tomato — this particular mirror does not
include corn/maize, unlike the full ~38-class PlantVillage set). See
data/sources.md for provenance.

Kept in alphabetical order to match torchvision.datasets.ImageFolder's
label assignment (it sorts class-folder names alphabetically) — this must
match the raw Kaggle folder names exactly.
"""

CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Classes considered "healthy" — used when computing false negative rate
# (a false negative = model predicts healthy/wrong-disease when the true
# label is a disease, i.e. it misses a real problem).
HEALTHY_CLASSES = {"Pepper__bell___healthy", "Potato___healthy", "Tomato_healthy"}

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
