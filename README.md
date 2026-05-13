# 🌾 END-TO-END PIPELINE & WEB APP FOR CROP RECOMMENDATION SYSTEM

> **Author:** Vinothkumar Durairaj  
> **Contact:** [vj482oo2@gmail.com](mailto:vj482oo2@gmail.com)  
> **MSc Dissertation Project** — Machine Learning, Data Science & Data Engineering

---

## 📋 Project Overview

An intelligent crop recommendation system that predicts the most suitable crop
based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall.
The system includes a full ML pipeline, model comparison, REST API backend,
and an interactive web interface.

---

## 🏗️ System Architecture

```
crop-recommendation-system/
│
├── data/                        # Raw datasets
│   ├── crop_dataset.csv         # Primary: Crop Recommendation Dataset (Kaggle)
│   └── soil_dataset.csv         # Secondary: Soil Properties Dataset (Mendeley)
│
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis notebook
│
├── src/                         # Core ML source modules
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading, cleaning, validation
│   ├── train_model.py           # Model training & comparison pipeline
│   └── predict.py               # Inference module
│
├── models/                      # Persisted model artefacts
│   ├── crop_model.pkl           # Best trained model
│   └── scaler.pkl               # Feature scaler
│
├── app/                         # Flask web application
│   ├── app.py                   # Application entry point + API routes
│   ├── templates/
│   │   └── index.html           # Frontend UI
│   └── static/
│       ├── css/style.css        # Stylesheet
│       └── js/main.js           # Frontend JavaScript
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the Project

```bash
cd crop-recommendation-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Datasets

Download datasets and place them in the `data/` folder:
- **Primary**: `data/crop_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Secondary**: `data/soil_dataset.csv` from [Mendeley](https://data.mendeley.com/datasets/36xkf32pph/1)

---

## 🚀 Running the Pipeline

### Step 1 — Preprocess & Train Models

```bash
python src/train_model.py
```

This will:
- Load and validate the dataset
- Run EDA summary stats
- Train Decision Tree, Random Forest, and Gradient Boosting
- Compare all models using Accuracy, Precision, Recall, F1
- Save the best model to `models/crop_model.pkl`

### Step 2 — Launch the Web Application

```bash
python app/app.py
```

Open your browser at: **http://localhost:5000**

---

## 🤖 Models Implemented

| Model | Description |
|---|---|
| Decision Tree | Interpretable baseline model |
| Random Forest | Ensemble — reduces overfitting |
| Gradient Boosting | High-performance sequential ensemble |

---

## 📊 Evaluation Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Confusion Matrix

---

## 🌱 Input Features

| Feature | Unit | Description |
|---|---|---|
| Nitrogen | kg/ha | Nitrogen content in soil |
| Phosphorus | kg/ha | Phosphorus content in soil |
| Potassium | kg/ha | Potassium content in soil |
| Temperature | °C | Average temperature |
| Humidity | % | Relative humidity |
| pH | 0–14 | Soil pH value |
| Rainfall | mm | Annual rainfall |

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **ML**: Scikit-learn, Joblib
- **Data**: Pandas, NumPy
- **Visualisation**: Matplotlib, Seaborn
- **Backend**: Flask
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript

---

## 👤 Author

**Vinothkumar Durairaj**  
MSc — Machine Learning, Data Science & Data Engineering  
📧 [vj482oo2@gmail.com](mailto:vj482oo2@gmail.com)

---
