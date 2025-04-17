# PPE Detection System Using YOLO

## Project Overview

The **PPE Detection System** is a deep learning-based solution designed for detecting personal protective equipment (PPE) in images using the YOLO (You Only Look Once) model. This project leverages machine learning frameworks such as TensorFlow, Keras, and PyTorch, and integrates cloud tools like Azure for scalable, real-time deployment. The system is built to classify images and detect PPE objects, contributing to safety monitoring in industries such as construction, healthcare, and manufacturing.

The project follows a structured pipeline, including data collection, preprocessing, model training, deployment, and continuous monitoring, as outlined in the project documentation.

**Repository Link**: https://github.com/mhany7/PPE-Detection-YOLO

## Team Roles

- **Data Collection and Preprocessing**: Mohamed Hany
- **Model Training and Fine-Tuning**: Youssef Samir
- **Model Deployment**: Merna Mahfouz
- **Final Presentation**: Mohamed Hany
- **Documentation**: Mona Youssef

## Project Milestones

The project is divided into five key milestones, each with specific objectives and deliverables:

### Milestone 1: Data Collection, Preprocessing, and Exploration

- **Objectives**: Collect and preprocess datasets for PPE detection, and perform exploratory data analysis (EDA).
- **Tasks**:
  - Gathered labeled datasets (e.g., COCO, Pascal VOC) with PPE annotations.
  - Preprocessed images through resizing, normalization, and augmentation (e.g., rotations, flips, color jittering).
  - Conducted EDA to visualize class distributions and bounding box annotations, addressing data imbalances.
- **Deliverables**:
  - Cleaned and preprocessed dataset.
  - Preprocessing pipeline documentation.
  - EDA report with visualizations.

### Milestone 2: Model Development and Optimization

- **Objectives**: Develop and evaluate a YOLO-based object detection model.
- **Tasks**:
  - Implemented a YOLO model for PPE detection.
  - Evaluated model performance using metrics like mean Average Precision (mAP), Intersection over Union (IoU), and detection accuracy.
  - Optimized the model through hyperparameter tuning and transfer learning.
- **Deliverables**:
  - Trained YOLO model for PPE detection.
  - Model evaluation report.

### Milestone 3: Advanced Techniques and Cloud Integration

- **Objectives**: Enhance the model with transfer learning and deploy it on the cloud.
- **Tasks**:
  - Fine-tuned a pre-trained YOLO model (e.g., from ImageNet) for improved accuracy.
  - Integrated the model with Azure Cognitive Services for scalable deployment.
- **Deliverables**:
  - Enhanced YOLO model.
  - Deployed system on Azure.

### Milestone 4: MLOps, Monitoring, and Web Interface

- **Objectives**: Implement MLOps practices, continuous monitoring, and a user-friendly web interface.
- **Tasks**:
  - Developed a web interface for real-time PPE detection.
  - Established an MLOps pipeline for experiment tracking and model retraining.
  - Set up monitoring with automated alerts for performance degradation.
- **Deliverables**:
  - Deployed models with web interface.
  - MLOps pipeline documentation.
  - Model monitoring setup.

### Milestone 5: Final Documentation and Presentation

- **Objectives**: Summarize the project and present results.
- **Tasks**:
  - Compiled a comprehensive project report detailing challenges, solutions, and real-world applications.
  - Created a presentation showcasing the workflow, results, and live demo.
  - Proposed future improvements, such as edge computing for faster predictions.
- **Deliverables**:
  - Final project report.
  - Final presentation.
  - Future improvement recommendations.

## Installation

To run the PPE Detection System locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/mhany7/PPE-Detection-YOLO.git
   cd PPE-Detection-YOLO
   ```

2. **Install Dependencies**: Ensure Python 3.8+ is installed, then install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**: Download the pre-trained YOLO model weights and place them in the `models/` directory (instructions in the repository).

4. **Run the Web Interface**: Launch the web application for real-time predictions:

   ```bash
   python app.py
   ```

   Access the interface at `http://localhost:5000`.

5. **Running the web interface on Streamlit cloud**:
    https://ppe-detection-mvkxswn5kxjddtkjabcyqn.streamlit.app/   

## Usage

- **Web Interface**: Upload images via the web interface to detect PPE in real-time.

- **API**: Use the provided API endpoints for programmatic access (see `docs/api.md` for details).

- **Command Line**: Run the detection script directly:

  ```bash
  python detect.py --image path/to/image.jpg
  ```

## Demo

A live demo of the PPE Detection System is available at: \[Insert Demo URL or Placeholder, e.g., `https://ppe-detection-demo.azurewebsites.net`\].\
Please contact the team for access credentials or to schedule a live demonstration.

## Project Structure

```
PPE-Detection-YOLO/
├── data/                    # Datasets and preprocessed images
├── models/                  # Trained YOLO model weights
├── src/                     # Source code for preprocessing, training, and inference
├── app.py                   # Web interface application
├── requirements.txt         # Python dependencies
├── docs/                    # Documentation (API, MLOps pipeline, etc.)
└── README.md                # This file
```

## Future Improvements

- Incorporate advanced YOLO variants (e.g., YOLOv8) for improved accuracy and speed.
- Extend the web interface with additional features, such as batch processing.
- Deploy the system on edge devices for low-latency predictions in real-world environments.

## Conclusion

The PPE Detection System leverages YOLO and cloud technologies to provide a scalable, accurate solution for PPE monitoring. With robust MLOps practices and a user-friendly interface, the system is well-suited for real-world applications in safety-critical industries.

For more details, refer to the GitHub repository or contact the team.
