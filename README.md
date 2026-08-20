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
Thesis-YOLO-Autonomous-Vehicles/
├── README.md                    # Project overview and documentation
├── docs/                        # Documentation
│   ├── proposal/               # Phase I proposal document
│   ├── figures/                # Extracted figures from thesis
│   └── thesis-original/        # Original thesis reference
├── data/                        # Dataset information
│   └── dataset-info/           # Dataset description and metadata
├── src/                         # Source code (future phases)
│   └── config/                 # Configuration files
├── requirements/                # Python dependencies
└── scripts/                     # Utility scripts
```

## Current Status

### Phase I: Proposal (Completed)
- ✅ Proposal document formatted according to DIU MSc template
- ✅ Literature review of 8 relevant studies
- ✅ Research methodology outlined
- ✅ Preliminary results with performance tables
- ✅ Performance graphs for all 7 YOLO model variants

### Phase II: Implementation (Planned)
- 🔄 Model training pipeline
- 🔄 Data preprocessing scripts
- 🔄 Evaluation metrics implementation

### Phase III: Results & Analysis (Planned)
- 🔄 Comprehensive results analysis
- 🔄 Comparison with state-of-the-art
- 🔄 Final thesis documentation

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

## Author

**Md. Robiul Islam Rabby**  
Roll No: 1910046  
MSc. in Computer Science and Engineering  
Daffodil International University

## License

This project is for academic research purposes.

## Acknowledgments

- Supervisor: Ms. Fariya Tabassum, Assistant Professor
- Department of Electrical & Computer Engineering
- Daffodil International University
