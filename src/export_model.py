import torch
from torchvision import models

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold"
]

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        "models/resnet18_best.pth",
        map_location="cpu"
    )
)

model.eval()

example_input = torch.randn(1, 3, 224, 224)

scripted_model = torch.jit.trace(
    model,
    example_input
)

scripted_model.save(
    "models/resnet18_scripted.pt"
)

print("TorchScript model exported successfully!")