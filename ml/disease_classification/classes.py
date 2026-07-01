"""
Disease/pest class taxonomy for the smoke-test model.

This is a small subset of the PlantVillage class list, chosen to keep the
smoke-test dataset and training loop fast. Swap in the full ~38-class
PlantVillage taxonomy (see data/sources.md) once real training begins.

Each class is (crop, condition). "healthy" is a valid condition per crop.
"""

# Kept in alphabetical order to match torchvision.datasets.ImageFolder's
# label assignment (it sorts class-folder names alphabetically).
CLASSES = [
    "corn_common_rust",
    "corn_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
]

# Classes considered "healthy" — used when computing false negative rate
# (a false negative = model predicts healthy/wrong-disease when the true
# label is a disease, i.e. it misses a real problem).
HEALTHY_CLASSES = {"tomato_healthy", "potato_healthy", "corn_healthy"}

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
