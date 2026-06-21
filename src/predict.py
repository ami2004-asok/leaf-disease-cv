import argparse
import json
import torch
from PIL import Image
from torchvision import models

from transforms import val_transform


def load_model(model_path, num_classes):

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


def predict(image_path, model, class_names):

    image = Image.open(image_path).convert("RGB")

    image = val_transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(output, dim=1)

    confidence, idx = probs.max(1)

    return (
        class_names[idx.item()],
        confidence.item()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    with open("notebooks/class_names.json") as f:
        class_names = json.load(f)

    model = load_model(
        "models/resnet18_best.pth",
        len(class_names)
    )

    label, confidence = predict(
        args.image,
        model,
        class_names
    )

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence*100:.2f}%")