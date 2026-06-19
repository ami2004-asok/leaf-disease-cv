import matplotlib.pyplot as plt
from PIL import Image
from transforms import train_transform
import torch

image_path = "data/train/early_blight/0a2726e0-3358-4a46-b6dc-563a5a9f2bdf___RS_Erly.B 7860.jpg"

image = Image.open(image_path).convert("RGB")

fig, axes = plt.subplots(2, 4, figsize=(10, 6))

for ax in axes.flat:

    aug_img = train_transform(image)

    # denormalize
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    aug_img = aug_img * std + mean
    aug_img = aug_img.permute(1,2,0)

    ax.imshow(aug_img)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "augment_samples.png"
)

plt.show()