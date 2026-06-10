# Plant Leaf Disease Detector

A computer vision project that uses PyTorch to detect plant leaf diseases from images.

## Environment Verification

* PyTorch Version: 2.12.0+cpu
* TorchVision Version: 0.27.0+cpu
* CUDA Available: False
* Tensor Device: cpu

### Verification Output

```text
PyTorch 2.12.0+cpu
torchvision 0.27.0+cpu
CUDA available: False
Tensor device: cpu
```

## Project Structure

```text
leaf-disease-cv/
│
├── data/
│   ├── raw/
│   │   ├── healthy/
│   │   ├── early_blight/
│   │   ├── late_blight/
│   │   └── leaf_mold/
│   ├── processed/
|   ├── train/
│   │   ├── healthy/
│   │   ├── early_blight/
│   │   ├── late_blight/
│   │   └── leaf_mold/
│   │
│   └── val/
│       ├── healthy/
│       ├── early_blight/
│       ├── late_blight/
│       └── leaf_mold/
│
├── src/
|    ├── dataset.py
│    ├── class_distribution.py
│    ├── visualize_batch.py
|    └── gpu_check.py
|
├── models/
├── notebooks/
│
├──split_dataset.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Dependencies

* PyTorch
* TorchVision
* Pillow
* Matplotlib
* scikit-learn 


## Batch visualization confirming shape:
 # Dataloader Verification

Dataset Size: 4360
Batch Shape: torch.Size([32, 3, 224, 224])
Sample Labels: tensor([3, 0, 0, 2, 1])

## Class Imbalance Summary
 # Class Distribution

early_blight    : 1000
healthy         : 1591
late_blight     : 1909
leaf_mold       : 952


## Author

AMI ASOK
