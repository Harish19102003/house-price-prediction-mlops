
# 🏠 House Price Prediction - End-to-End ML Pipeline

A production-ready machine learning pipeline for predicting house prices using MLOps best practices and clean software design.

## 🚀 Features
- EDA-driven feature engineering
- Modular architecture with design patterns (Factory, Strategy, Template)
- MLflow for experiment tracking
- CI/CD-ready structure
- Streamlit UI 
- Dockerized for deployment
- [Project Roadmap](./roadmap.md) – Detailed breakdown of phases, milestones, and deliverables

## 🔧 Setup

```bash
git clone https://github.com/Harish19102003/house-price-prediction-mlops.git
cd house-price-prediction-mlops
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/              # Raw dataset (Kaggle dataset)
│   └── processed/        # Processed data (after EDA & preprocessing)
├── docs/                 # Project documentation
├── mlruns/               # MLflow tracking logs and experiments
├── notebooks/            # Jupyter Notebooks for exploratory data analysis (EDA)
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Configuration file (paths, hyperparameters, etc.)
│   ├── data_ingestion.py # Data ingestion script
│   ├── preprocessing.py  # Data preprocessing and feature engineering
│   ├── model_training.py # Model training script
│   ├── evaluation.py     # Model evaluation script
│   ├── deployment.py     # Model deployment script using MLflow
│   ├── inference.py      # Inference pipeline for prediction
│   └── utils.py          # Helper functions (e.g., logging)
├── tests/                # Unit tests for the pipeline
│   └── test_model.py     # Test model training
├── Dockerfile            # Dockerfile to containerize the project
├── docker-compose.yml    # Docker Compose for MLflow + App
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── roadmap.md            # Detailed project phases, tasks, and goals
└── setup.py              # For packaging and installation
```

## 🧪 Run the Pipeline (Example)

```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/model_training.py
python src/evaluation.py
python src/deployment.py
```

## 🧰 Tools & Tech

- Python 3.8+
- Pandas, NumPy, Scikit-Learn, XGBoost
- MLflow
- Streamlit (optional)
- Docker & Docker Compose

## 💻 Deployment

```bash
docker-compose up --build
```

## 📜 License

This project is licensed under the MIT License.
