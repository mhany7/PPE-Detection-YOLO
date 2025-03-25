# SafeScan-YOLO
A Protective Personal Equipment (PPE) compliance monitoring system using YOLOv11 and the SH17 dataset.

## Project Overview
SafeScan-YOLO is a deep learning-based solution designed to monitor compliance with Personal Protective Equipment (PPE) usage in industrial and workplace environments. Leveraging the YOLOv11 object detection model and the SH17 dataset, this system detects PPE items (e.g., helmets, masks, safety vests) in real-time, ensuring safety standards are met. The project spans data preprocessing, model training, optimization, and deployment, with potential integration into cloud platforms like Azure for scalable, real-time applications.

## Objectives
- Develop an object detection system to identify PPE items in images and video streams.
- Ensure accurate detection and classification of PPE compliance.
- Optimize the system for real-time performance in safety-critical settings.
- Deploy a user-friendly interface for monitoring PPE usage.

## Technologies Used
- **Python**: Core programming language (libraries: PyTorch, OpenCV, NumPy, Pandas, Albumentations).
- **Deep Learning**: YOLOv11 for object detection.
- **Dataset**: SH17 Dataset for PPE Detection (8,099 images, 17 classes).
- **Image Processing**: OpenCV for preprocessing and visualization.
- **Cloud (Optional)**: Microsoft Azure for deployment and scalability.
- **GitHub**: Repository management and collaboration.

## Team Members & Roles
| Team Member         | Responsibility                     |
|---------------------|------------------------------------|
| Merna Mahfouz       | Data Collection & Preprocessing   |
| Youssef Samir       | Data Collection & Model Training  |
| Mohamed Hany        | Model Development & Training      |
| Mona Youssef        | Object Detection & Evaluation     |
| Mohamed Atef        | Model Optimization & Tuning       |

## Key Milestones
1. **Data Collection, Preprocessing, and Exploration**  
   - Gather and preprocess the SH17 dataset, including resizing, normalization, and augmentation.  
   - Conduct exploratory data analysis (EDA) to understand class distributions and challenges.

2. **Model Development & Training**  
   - Train the YOLOv11 model on the preprocessed SH17 dataset for PPE detection.  
   - Evaluate performance using metrics like mAP and IoU.

3. **Advanced Techniques & Optimization**  
   - Apply transfer learning and hyperparameter tuning to enhance accuracy and speed.  
   - Optimize for real-time inference.

4. **System Deployment & Monitoring**  
   - Deploy the model as a real-time endpoint (e.g., via Azure or a web interface).  
   - Implement MLOps practices for continuous monitoring and retraining.

5. **Documentation & Presentation**  
   - Compile a final report and presentation showcasing the system’s workflow and results.

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- Albumentations
- Access to the SH17 dataset (available on [Kaggle](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection))

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mhany7/SafeScan-YOLO.git
   cd SafeScan-YOLO
