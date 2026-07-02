"""
Disease/pest class taxonomy.

Combines the 15 PlantVillage classes (Pepper, Potato, Tomato — emmarex/
plantdisease mirror) with maize, rice (paddy), cotton, and chilli classes
merged from separate Kaggle datasets (see data/sources.md and
prepare_extra_crops.py). Chilli has only leaf-curl + healthy in its dataset.
Groundnut is intentionally excluded — its available dataset is object-detection
format, so groundnut stays advisory-only in the app.

Kept in the exact order torchvision.datasets.ImageFolder produces (it sorts
class-folder names by byte value, so all capitalized PlantVillage names sort
before the lowercase new-crop names). train.py asserts this list equals the
dataset's class order, so this must match the folder names on disk exactly.
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
    "chilli_healthy",
    "chilli_leaf_curl_virus",
    "corn_common_rust",
    "corn_gray_leaf_spot",
    "corn_healthy",
    "corn_northern_leaf_blight",
    "cotton_bacterial_blight",
    "cotton_fusarium_wilt",
    "cotton_healthy",
    "cotton_leaf_curl_virus",
    "paddy_bacterial_leaf_blight",
    "paddy_blast",
    "paddy_brown_spot",
    "paddy_tungro",
]

# Classes considered "healthy" — used when computing false negative rate
# (a false negative = model predicts healthy/wrong-disease when the true
# label is a disease, i.e. it misses a real problem).
HEALTHY_CLASSES = {
    "Pepper__bell___healthy",
    "Potato___healthy",
    "Tomato_healthy",
    "chilli_healthy",
    "corn_healthy",
    "cotton_healthy",
}

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
