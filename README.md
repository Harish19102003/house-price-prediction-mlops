house-price-prediction/
├── data/
│   └── raw/              # Raw dataset (kaggle dataset)
│   └── processed/        # Processed data (after EDA & preprocessing)
├── docs/                 # Project documentation
├── mlruns/               # MLflow tracking logs and experiments
├── notebooks/            # Jupyter Notebooks for exploratory data analysis (EDA)
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Configuration file (for paths, hyperparameters, etc.)
│   ├── data_ingestion.py # Data ingestion script
│   ├── preprocessing.py  # Data preprocessing and feature engineering
│   ├── model_training.py # Model training script
│   ├── evaluation.py     # Model evaluation script
│   ├── deployment.py     # Model deployment script using MLflow
│   ├── inference.py      # Inference pipeline (for prediction)
│   └── utils.py          # Helper functions (logging, etc.)
├── tests/                # Unit tests for the pipeline
│   └── test_model.py     # Test model training
├── Dockerfile            # Dockerfile to containerize the project
├── docker-compose.yml    # Docker compose file for multi-container setup (MLflow + App)
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── setup.py              # For packaging and installation
