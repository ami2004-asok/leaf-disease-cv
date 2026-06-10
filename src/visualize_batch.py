import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import (
    LeafDiseaseDataset,
    transform,
    CLASS_NAMES
)

dataset = LeafDiseaseDataset(
    "data/train",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(loader))

# Denormalize
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6)
)

for i, ax in enumerate(axes.flat):

    img = images[i] * std + mean

    img = img.clamp(0, 1)

    img = img.permute(
        1,
        2,
        0
    ).numpy()

    ax.imshow(img)

    ax.set_title(
        CLASS_NAMES[
            labels[i].item()
        ]
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "sample_batch.png"
)

plt.show()