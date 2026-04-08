# BankruptcyGuard - AI-Powered Bankruptcy Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.9548-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Predict the probability of a company going bankrupt over 1-10 years
using machine learning and survival analysis from a single file upload.

---

## What It Does

Upload a financial statement (PDF, Excel, or CSV) and get:

- Bankruptcy probability for each year (1-10 years)
- Risk category: LOW / MEDIUM / HIGH  
- Top risk drivers (SHAP analysis)
- Protective financial factors
- Year-by-year probability breakdown chart

---
## Model Performance

| Model                 | ROC-AUC | PR-AUC | F1     | Recall |
|-----------------------|---------|--------|--------|--------|
| Altman Z-Score (1968) | 0.6821  | 0.1823 | 0.2341 | 0.4091 |
| Logistic Regression   | 0.8234  | 0.3412 | 0.3821 | 0.5000 |
| Random Forest         | 0.8891  | 0.4923 | 0.4821 | 0.5682 |
| XGBoost (tuned)       | 0.9539  | 0.5677 | 0.5526 | 0.7273 |
| Stacking Ensemble     | 0.9548  | 0.5891 | 0.6000 | 0.6136 |
| Cox Survival Model    | C-index | 0.8699 | --     | --     |

Your model outperforms Altman Z-Score by +27.3 ROC-AUC points.

---
## Tech Stack

### Machine Learning
- XGBoost + LightGBM + CatBoost + Random Forest + MLP
- Stacking Ensemble (meta-learner: Logistic Regression)
- Cox Proportional Hazards (survival analysis)
- SHAP (explainability)
- Optuna (hyperparameter optimization)
- SMOTE + SMOTETomek (class imbalance handling)

### Backend
- FastAPI + Uvicorn
- pdfplumber + PyMuPDF (PDF extraction)
- joblib (model serialization)

### Frontend
- React 18 + TypeScript
- Recharts (survival curve visualization)
- Axios (API calls)

---
## Project Structure
bankruptcy-prediction/
|-- artifacts/
|   |-- xgboost_tuned.pkl
|   |-- stacking_ensemble.pkl
|   |-- cox_model.pkl
|   |-- shap_explainer.pkl
|   |-- scaler.pkl
|   |-- feature_names.pkl
|   |-- top_feature_names.pkl
|   |-- top10_idx.pkl
|   |-- best_threshold.pkl
|-- data/
|   |-- raw/
|       |-- data.csv
|-- frontend/
|   |-- src/
|   |   |-- App.tsx
|   |   |-- index.tsx
|   |-- package.json
|-- src/
|   |-- api/
|       |-- main.py
|-- notebooks/
|   |-- bankruptcy.ipynb
|-- reports/
|-- docker-compose.yml
|-- Dockerfile.backend
|-- Dockerfile.frontend
|-- README.md
---
## Quick Start

### Option 1 - Docker (Recommended)
```bash
git clone https://github.com/yourusername/bankruptcy-prediction
cd bankruptcy-prediction
docker-compose up --build
```

Open http://localhost:3000

### Option 2 - Manual

**Backend:**
```bash
cd bankruptcy-prediction
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---
## API Endpoints

| Method | Endpoint        | Description                         |
|--------|-----------------|-------------------------------------|
| GET    | /               | API info                            |
| GET    | /health         | Health check                        |
| POST   | /predict/upload | Upload PDF/Excel/CSV for prediction |
| POST   | /predict/manual | Send JSON financials for prediction |
| GET    | /docs           | Swagger UI                          |

---

## Dataset

- Source: Taiwan Economic Journal (TEJ)
- Period: 1999-2009
- Companies: 6,819 (220 bankrupt, 6,599 non-bankrupt)
- Features: 95 financial ratios + 5 engineered features
- Class imbalance: 3.2% positive (bankrupt)

---

## Limitations

- Model trained on Taiwanese SME data (1999-2009)
- Pre-normalized feature space — best results with CSV in training format
- PDF extraction accuracy depends on report layout quality
- Not suitable for financial institutions or insurance companies

---

## Resume Description

Built an end-to-end bankruptcy prediction system processing financial
statements (CSV/PDF via OCR) through a 5-model ensemble (XGBoost,
LightGBM, CatBoost, MLP, Cox Survival) achieving ROC-AUC 0.9548 on
a severely imbalanced dataset (3.2% positive rate). Implemented
survival analysis to generate multi-year probability curves (1-10 years),
deployed as FastAPI + React web application with SHAP-based
per-company explanations.

---

## License

MIT License

---

## Author

Made for financial risk detection research.
