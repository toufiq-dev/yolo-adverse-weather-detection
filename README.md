# Performance Evaluation of YOLO Models for Object Detection in Autonomous Vehicles Under Adverse Weather Conditions

## Project Overview

This repository contains the research work for evaluating YOLO-based object detection models for autonomous vehicles operating in adverse weather conditions. The study focuses on assessing model robustness and accuracy in environments with reduced visibility.

## Research Focus

- **Objective**: Evaluate YOLO model variants (YOLOv8x, YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, YOLOv11x, YOLOv12x) under adverse weather conditions
- **Dataset**: DAWN (Detection in Adverse Weather Nature) dataset
- **Weather Conditions**: Fog, Rain, Snow, Sandstorms
- **Evaluation Metrics**: Precision, Recall, F1-Score, Accuracy, mAP@50, mAP@50-95

## Project Structure

```
yolo-adverse-weather-detection/
├── README.md                    # Project overview and documentation
├── docs/                        # Documentation
│   ├── proposal/               # Phase I proposal document
│   ├── figures/                # Extracted figures from thesis
│   └── thesis-original/        # Original thesis reference
├── data/                        # Dataset information
│   └── dataset-info/           # Dataset description and metadata
├── src/                         # Source code
│   ├── __init__.py
│   ├── training.py              # YOLO training pipeline
│   ├── evaluation.py            # Metrics computation & evaluation
│   ├── augmentation.py          # Weather-specific augmentation
│   ├── visualization.py         # Performance plots & thesis figures
│   └── config/                  # Configuration files
│       ├── model_config.yaml    # Model & training config
│       └── dawn_dataset.yaml    # YOLO-format dataset config
├── requirements/                # Python dependencies
└── scripts/                     # Utility scripts
    ├── data_preprocessing.py    # Image preprocessing (CLAHE, dehazing, etc.)
    ├── extract_figures.py       # Extract figures from thesis PDF
    ├── run_experiment.py        # End-to-end experiment pipeline
    └── compare_models.py        # Generate thesis comparison tables/figures
```

## Current Status

### Phase I: Proposal (Completed)
- ✅ Proposal document formatted according to DIU MSc template
- ✅ Literature review of 8 relevant studies
- ✅ Research methodology outlined
- ✅ Preliminary results with performance tables
- ✅ Performance graphs for all 7 YOLO model variants

### Phase II: Implementation (Completed)
- ✅ Model training pipeline (`src/training.py`)
- ✅ Data preprocessing scripts (`scripts/data_preprocessing.py`)
- ✅ Weather-specific augmentation (`src/augmentation.py`)
- ✅ Evaluation metrics (`src/evaluation.py`)
- ✅ Visualization & thesis figures (`src/visualization.py`)
- ✅ End-to-end experiment runner (`scripts/run_experiment.py`)
- ✅ Model comparison script (`scripts/compare_models.py`)
- ✅ YOLO dataset config (`src/config/dawn_dataset.yaml`)

### Phase III: Results & Analysis (Next)
- 🔄 Run experiments on DAWN dataset
- 🔄 Generate final comparison tables and figures
- 🔄 Write up analysis and SOTA comparison

## Key Findings (Preliminary)

| Model | Precision | Recall | F1-Score | Accuracy | mAP@50 | mAP@50-95 |
|-------|-----------|--------|----------|----------|--------|-----------|
| YOLOv11n | 87.21% | 79.93% | 83.41% | 67.95% | 85.92% | 65.27% |
| YOLOv11s | 94.57% | 71.61% | 81.51% | 68.79% | 94.84% | 72.94% |
| YOLOv11m | 93.10% | 76.03% | 83.70% | 71.97% | 95.87% | 74.20% |
| YOLOv11l | 93.17% | 90.61% | 91.87% | 84.96% | 95.40% | 77.16% |
| YOLOv11x | 93.36% | 90.91% | 92.12% | 85.39% | 96.08% | 76.63% |
| YOLOv12x | 91.70% | 87.34% | 89.47% | 80.94% | 94.30% | 74.58% |
| **YOLOv8x** | **93.64%** | **91.96%** | **92.80%** | **86.56%** | **96.14%** | **80.44%** |

## Dataset Setup

The DAWN dataset must be downloaded from Roboflow before running experiments.

### Prerequisites
```bash
pip install roboflow
```

### Download & Setup

1. Get a free API key at https://app.roboflow.com/settings/api
2. Set the environment variable or pass it directly:

```bash
# Option A: Environment variable
export ROBOFLOW_API_KEY=your_key_here
python scripts/download_dataset.py

# Option B: Pass directly
python scripts/download_dataset.py --api-key YOUR_KEY
```

This will:
- Download the DAWN dataset (974 train + 425 valid images, 7 classes)
- Create a test split (70/20/10 rule)
- Set up the YOLO-format directory structure at `data/dawn_dataset/`

### Dataset Structure
```
data/dawn_dataset/
├── train/
│   ├── images/    # 974 training images
│   └── labels/    # YOLO-format bounding box labels
├── valid/
│   ├── images/    # 425 validation images
│   └── labels/
├── test/
│   ├── images/    # ~140 test images (auto-created)
│   └── labels/
└── data.yaml      # YOLO dataset config
```

## Author

**Toufiqur Rahman**  
Student ID: 252-25-013  
MSc. in Computer Science and Engineering  
Daffodil International University

## License

This project is for academic research purposes.

## Acknowledgments

- Supervisor: Professor Dr. Sheak Rashed Haider Noori, Professor & Head
- Department of Computer Science and Engineering
- Daffodil International University
