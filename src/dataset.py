from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["early_blight", "healthy", "late_blight", "leaf_mold"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class LeafDiseaseDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*"):

                if img_path.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:
                    self.samples.append(
                        (img_path, CLASS_TO_IDX[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# TRAIN DATASET
train_ds = LeafDiseaseDataset(
    "data/train",
    transform=transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# TEST ONE BATCH
images, labels = next(iter(train_loader))

print("Dataset Size:", len(train_ds))
print("Batch Shape:", images.shape)
print("Sample Labels:", labels[:5])