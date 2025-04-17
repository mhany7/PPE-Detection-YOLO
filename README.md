# Image Classification and Object Detection System - Milestone 01

This milestone focuses on data collection, preprocessing, and exploration for the image classification and object detection system using YOLOv11.

## Features

- Data Collection: Downloads and prepares the CIFAR-10 dataset
- Data Preprocessing:
  - Image resizing to 224x224
  - Normalization
  - Data augmentation (random flips, rotations, color jittering)
- Dataset Splitting: Divides data into training (70%), validation (15%), and test (15%) sets
- Exploratory Data Analysis:
  - Sample image visualization
  - Class distribution analysis
- YOLOv11 Integration:
  - Automatic model download
  - YOLO-compatible data structure preparation
  - Object detection support

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the milestone:
```bash
python src/milestone_01.py
```

## Output

The script will generate:
- Sample images visualization in `results/sample_images.png`
- Class distribution analysis in `results/class_distribution.png`
- Processed dataset in the `data` directory
- YOLO-compatible data structure in `data/yolo_data` directory

## Data Augmentation Techniques

The following augmentation techniques are applied:
- Random horizontal flips
- Random rotations (±10 degrees)
- Color jittering (brightness and contrast adjustments)

## Directory Structure

```
.
├── data/               # Downloaded and processed datasets
│   └── yolo_data/     # YOLO-compatible data structure
│       ├── images/    # Training and validation images
│       └── labels/    # Training and validation labels
├── results/           # Generated visualizations and analysis
├── src/               # Source code
│   └── milestone_01.py
└── requirements.txt   # Project dependencies
```

## YOLOv11 Integration

The code includes support for YOLOv11 object detection:
- Automatically downloads the YOLOv11 model
- Creates the required directory structure for YOLO training
- Prepares data in YOLO-compatible format 