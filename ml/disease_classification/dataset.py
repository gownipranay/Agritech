"""Dataset + augmentation pipeline for phone-camera crop photos.

Real farmer photos are blurry, poorly lit, and off-center — very unlike the
clean lab-background PlantVillage images. Training-time augmentation
simulates this so the model doesn't only generalize to studio conditions.
"""

from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))], p=0.4),
        transforms.RandomAdjustSharpness(sharpness_factor=0.3, p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image_folder_dataset(root: str, train: bool) -> ImageFolder:
    transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM
    return ImageFolder(root=root, transform=transform)
