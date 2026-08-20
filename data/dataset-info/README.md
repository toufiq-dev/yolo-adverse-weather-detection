# DAWN Dataset Information

## Overview

The DAWN (Detection in Adverse Weather Nature) dataset is designed for evaluating object detection models under adverse weather conditions. It is particularly useful for autonomous vehicle research.

## Source

- **Name**: DAWN Dataset
- **Source**: Roboflow
- **URL**: https://www.roboflow.com/datasets/dawn
- **Year**: 2024

## Dataset Structure

### Classes (7 total)
1. Bicycle
2. Bus
3. Car
4. Motorcycle
5. Person
6. Train
7. Truck

### Data Split

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 974 | 70% |
| Validation | 425 | 20% |
| Test | ~140 | 10% |

*Note: Test set was created using 70/20/10 split rule from training and validation data*

## Weather Conditions

The dataset includes images under various adverse weather conditions:

1. **Fog**: Reduces visibility and introduces blur
2. **Rain**: Adds water droplets to the lens, causing occlusion
3. **Snow**: Introduces snowflakes, causing motion blur
4. **Sandstorm**: Adds dust particles, resulting in significant noise and blur

## Data Format

- **Image Format**: JPEG/PNG
- **Annotation Format**: YOLO format (normalized bounding boxes)
- **Bounding Box**: (x_center, y_center, width, height)
- **Label Files**: Text files with object class IDs and bounding boxes

## Data Augmentation Techniques Applied

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
   - Improved local contrast in LAB color space
   - Aids feature detection in low-visibility conditions

2. **Dehazing**
   - Removed haze effects caused by fog
   - Enhanced clarity

3. **Denoising**
   - Fast Non-Local Means Denoising
   - Reduces noise while preserving edges

4. **Sharpening**
   - Kernel filter for image sharpening
   - Enhances object boundaries

5. **Weather-Specific Augmentation**
   - Fog simulation
   - Rain simulation
   - Snow simulation
   - Sandstorm simulation

## Usage

```python
# Example: Loading DAWN dataset
from ultralytics import YOLO

# Load model
model = YOLO('yolov8x.pt')

# Train on DAWN dataset
results = model.train(data='dawn_dataset.yaml', epochs=120, imgsz=640)
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{dawn_dataset_2024,
  title={DAWN: Detection in Adverse Weather Nature Dataset},
  author={Roboflow},
  year={2024},
  url={https://www.roboflow.com/datasets/dawn}
}
```
