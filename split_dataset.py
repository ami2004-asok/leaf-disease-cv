from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Source dataset
RAW_DIR = Path("data/raw")

# Output folders
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

# Split ratio
TRAIN_RATIO = 0.8

# Create train and val folders if they don't exist
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# Loop through each class folder
for class_dir in RAW_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    # Get all image files
    images = list(class_dir.glob("*"))

    # Split images into train and validation
    train_images, val_images = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=42
    )

    # Create class folders
    train_class_dir = TRAIN_DIR / class_dir.name
    val_class_dir = VAL_DIR / class_dir.name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    # Copy train images
    for img in train_images:
        shutil.copy(img, train_class_dir / img.name)

    # Copy validation images
    for img in val_images:
        shutil.copy(img, val_class_dir / img.name)

    print(
        f"{class_dir.name}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} val"
    )

print("\nDataset split completed successfully.")