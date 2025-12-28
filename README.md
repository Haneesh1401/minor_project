# minor_project


🌍 Weather-Aware ConvLSTM Spatio-Temporal Attention Framework
for Dust Source and Event Forecasting
📌 Project Overview

This project focuses on forecasting dust events and identifying dust source regions using satellite-based vegetation data (NDVI) and deep learning models, specifically ConvLSTM.

The system analyzes spatio-temporal patterns from satellite imagery to understand how dust evolves over time and space. This approach is useful for applications such as:

Climate monitoring

Air quality prediction

Environmental risk assessment

Disaster preparedness

🎯 Objective

To design and implement a deep learning–based forecasting framework that:

Identifies dust-prone regions

Learns temporal patterns from NDVI data

Predicts future dust activity using ConvLSTM networks

🧠 Core Technologies Used

Python

NumPy

Matplotlib

TensorFlow / Keras

Satellite Remote Sensing (MODIS NDVI)

ConvLSTM (Convolutional LSTM)

📂 Project Structure
Minor_project/
├── data/                     # (Ignored in Git) Generated datasets
│   └── NDVI arrays, ConvLSTM inputs
│
├── notebooks/                # Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   ├── ndvi_generation.ipynb
│   └── model_training.ipynb
│
├── scripts/                  # Python scripts (optional)
│   ├── preprocess.py
│   └── train_model.py
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md


🚫 Dataset Handling (Important)

Due to GitHub size limitations, large dataset files are not included in this repository.

The following file types are intentionally ignored:

data/
*.npy
*.npz
*.pt
*.h5


These datasets are generated locally during preprocessing.

🔄 Dataset Generation Workflow

Download satellite data (MODIS NDVI)

Preprocess raw satellite bands

Compute NDVI values

Stack NDVI over time to create spatio-temporal sequences

Feed data into ConvLSTM model

📊 Model Architecture

The project uses a ConvLSTM-based deep learning model, which combines:

Convolutional layers → capture spatial features

LSTM layers → capture temporal dependencies

This allows the model to learn how dust patterns evolve across time and space.

🧪 Training Pipeline

Load NDVI time-series data

Normalize and reshape input

Train ConvLSTM model

Evaluate loss and prediction accuracy

Visualize temporal changes

📈 Output & Visualizations

The model generates:

NDVI temporal heatmaps

Prediction maps

Training loss curves

These help analyze dust movement and intensity over time.

🛠 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Haneesh1401/minor_project.git
cd minor_project

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Preprocessing
python scripts/preprocess.py

5️⃣ Train the Model
python scripts/train_model.py

📌 Key Features

✅ Spatio-temporal modeling
✅ Deep learning–based forecasting
✅ Scalable data pipeline
✅ Real-world environmental application

🧠 Future Improvements

Add attention mechanism to ConvLSTM

Integrate meteorological data (ERA5)

Improve prediction resolution


Deploy using Streamlit or Flask

