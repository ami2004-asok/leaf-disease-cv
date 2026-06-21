# Plant Leaf Disease Detector

A computer vision project developed using PyTorch for automatic plant leaf disease classification. The system identifies four classes of leaf conditions: Healthy, Early Blight, Late Blight, and Leaf Mold using deep learning models.

---

## Environment Verification

* PyTorch Version: 2.12.0+cpu
* TorchVision Version: 0.27.0+cpu
* CUDA Available: False
* Tensor Device: CPU

### Verification Output

```text
PyTorch 2.12.0+cpu
torchvision 0.27.0+cpu
CUDA available: False
Tensor device: cpu
```

---

## Dataset

### Classes

* Early Blight
* Healthy
* Late Blight
* Leaf Mold

### Dataset Statistics

| Class        | Images |
| ------------ | -----: |
| Early Blight |   1000 |
| Healthy      |   1591 |
| Late Blight  |   1909 |
| Leaf Mold    |    952 |
| Total        |   5452 |

### Dataset Split

* Training Set: 80%
* Validation Set: 20%

---

## Data Preprocessing

### Training Transformations

* Resize (256 × 256)
* Random Resized Crop (224 × 224)
* Random Horizontal Flip
* Random Rotation (±15°)
* Color Jitter
* ToTensor
* Normalize (ImageNet Mean & Std)

### Validation Transformations

* Resize (224 × 224)
* ToTensor
* Normalize (ImageNet Mean & Std)

---

## DataLoader Verification

```text
Dataset Size: 4360
Batch Shape: torch.Size([32, 3, 224, 224])
Sample Labels: tensor([0, 2, 2, 3, 2])
```

---

## CNN Baseline Model

A custom Convolutional Neural Network (CNN) was implemented as the baseline model for leaf disease classification.

### CNN Architecture

* Convolution Layer 1 (32 Filters)

* ReLU

* Max Pooling

* Convolution Layer 2 (64 Filters)

* ReLU

* Max Pooling

* Convolution Layer 3 (128 Filters)

* ReLU

* Max Pooling

* Convolution Layer 4 (256 Filters)

* ReLU

* AdaptiveAvgPool

* Fully Connected Layer

* Dropout (0.3)

* Output Layer (4 Classes)

### CNN Performance

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 93.50% |
| Validation Loss     | 0.3581 |

---

## Transfer Learning using ResNet18

Transfer learning was performed using a pretrained ResNet18 model trained on ImageNet.

### Method

* Loaded pretrained ResNet18 weights
* Replaced final classification layer with 4-class output layer
* Initially trained only the final layer
* Fine-tuned Layer4 and the classifier

### ResNet18 Performance

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 98.63% |
| Validation Loss     | 0.0645 |

### Comparison

| Model        | Validation Accuracy |
| ------------ | ------------------- |
| CNN Baseline | 93.50%              |
| ResNet18     | 98.63%              |

---

## Data Augmentation

Data augmentation was applied during training to improve generalization and reduce overfitting.

### Techniques Used

* Random Resized Crop
* Horizontal Flip
* Random Rotation
* Color Jitter

### Benefits

* Increased dataset diversity
* Improved robustness to lighting variations
* Reduced overfitting
* Improved classification performance

---

## Model Evaluation

The best-performing ResNet18 model was evaluated using classification metrics and a confusion matrix.

### Classification Report

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Early Blight | 0.98      | 0.97   | 0.98     |
| Healthy      | 0.99      | 1.00   | 1.00     |
| Late Blight  | 0.98      | 0.98   | 0.98     |
| Leaf Mold    | 0.99      | 0.98   | 0.99     |

### Overall Performance

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 99%   |
| Macro F1-Score    | 0.99  |
| Weighted F1-Score | 0.99  |

### Error Analysis

* Most errors occurred between Early Blight and Late Blight.
* Healthy leaves were classified with near-perfect accuracy.
* Misclassifications were mainly caused by visually similar disease symptoms and variations in image quality.

---

## Results

### Best Model

ResNet18 Transfer Learning Model

### Final Accuracy

99%

### Generated Reports

```text
reports/
│
├── confusion_matrix.png
├── classification_report.txt
├── error_analysis.md
│
└── errors/
    ├── error_0.png
    ├── error_1.png
    └── ...
```
---

## Preprocessing Specification

### Input Size
224 x 224

### Color Format
RGB

### Normalization

Mean:
[0.485, 0.456, 0.406]

Std:
[0.229, 0.224, 0.225]

### Class Mapping

0 -> early_blight
1 -> healthy
2 -> late_blight
3 -> leaf_mold

---

## Smoke Test Results

### Objective

Verify that the exported ResNet18 model can successfully perform inference on unseen validation images.

### Test Commands and Results

| Image Path                                                                      | Actual Class | Predicted Class | Confidence |
| ------------------------------------------------------------------------------- | ------------ | --------------- | ---------- |
| data/val/late_blight/0b52ebed-e711-41f2-bfd1-af73410b370e___RS_Late.B 5463.jpg  | late_blight  | late_blight     | 99.97%     |
| data/val/early_blight/8fd0e481-6590-4314-8dd9-975e5da774c8___RS_Erly.B 9559.jpg | early_blight | early_blight    | 99.95%     |
| data/val/leaf_mold/561e6144-f53b-477e-96ef-88deff234bc6___Crnl_L.Mold 6855.jpg  | leaf_mold    | leaf_mold       | 99.97%     |

### Summary

* Total test images: 3
* Correct predictions: 3
* Smoke test accuracy: 100%

### Conclusion

The exported ResNet18 model successfully performed inference on all three held-out validation images. The CLI prediction script correctly returned the predicted class and confidence score for each image, demonstrating that the model export and inference pipeline are functioning as expected.


---

## Project Structure

```text
leaf-disease-cv/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   └── val/
│
├── models/
│   ├── leaf_cnn_best.pth
│   ├── resnet18_best.pth
│   └── resnet18_scripted.pt
│
├── notebooks/
│   └── class_names.json
│
├── reports/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── errors/
│
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── ResNet18.py
│   ├── evaluate.py
│   ├── class_distribution.py
│   ├── gpu_check.py
│   ├── trace_shapes.py
│   ├── visualize_augmentations.py
│   ├── visualize_batch.py
│   ├── predict.py
│   └── export_model.py
│
├── augment_samples.png
├── sample_batch.png
├── training_plot.png
│
├── split_dataset.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Author

AMI ASOK
