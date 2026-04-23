# 🛡️ Vehicle Insurance Claim Fraud Detection

> **End-to-end production-level machine learning project for portfolio demonstration**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

Insurance fraud costs the industry billions of dollars annually. This project builds a **binary classification system** that predicts whether a vehicle insurance claim is **fraudulent (1) or legitimate (0)** using real-world claim data.

The project is production-grade, covering:
- In-depth **Exploratory Data Analysis (EDA)**
- Rigorous **feature engineering** (9 domain-driven features)
- **Class imbalance handling** via SMOTE
- Training and tuning of **7 ML algorithms**
- A **FastAPI REST backend** serving the trained model
- A polished **Streamlit frontend** for interactive claim assessment

---

## 📊 Dataset Description

| Property          | Value                                          |
|-------------------|------------------------------------------------|
| **Source**        | [Kaggle — Car Insurance Fraud Detection](https://www.kaggle.com/datasets/ahluwaliasaksham/car-insurance-fraud-detection-dataset) |
| **Rows**          | 30,000 claims                                  |
| **Columns**       | 24 (raw) → 70+ after encoding & engineering   |
| **Target**        | `fraud_reported` (Y / N → 1 / 0)             |
| **Class balance** | ~88.5% legitimate / ~11.5% fraudulent         |
| **Missing data**  | `authorities_contacted` — ~25% (imputed)      |

### Key Features

| Feature                      | Type        | Description                              |
|------------------------------|-------------|------------------------------------------|
| `policy_annual_premium`      | Numeric     | Annual insurance premium paid            |
| `incident_type`              | Categorical | Type of incident (collision, theft, etc.)|
| `incident_severity`          | Ordinal     | Damage level (Minor / Major / Total Loss)|
| `police_report_available`    | Binary      | Whether a police report exists           |
| `authorities_contacted`      | Categorical | Which authority was notified             |
| `total_claim_amount`         | Numeric     | Total claim value in USD                 |
| `claim_to_premium_ratio`     | Engineered  | claim / premium — key fraud signal       |
| `no_police_report`           | Engineered  | Flag: absent police report               |
| `is_high_severity`           | Engineered  | Flag: Total Loss incident                |

---

## 🏗️ Project Structure

```
vehicle_fraud_detection/
│
├── 📓 notebook/
│   └── vehicle_fraud_detection.ipynb   # Full ML pipeline notebook
│
├── 📂 data/
│   └── car_insurance_fraud_dataset.csv # Raw dataset
│
├── 🤖 model/
│   ├── fraud_detection_model.pkl       # Trained model (Gradient Boosting)
│   ├── scaler.pkl                      # StandardScaler
│   ├── feature_names.pkl               # Ordered feature list
│   ├── model_results.csv               # All model comparison metrics
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── 🖥️ fastapi_app/
│   └── app/
│       ├── __init__.py
│       ├── main.py                     # FastAPI app, routes, lifespan
│       ├── schema.py                   # Pydantic request/response models
│       └── model_loader.py             # Model loading, feature engineering
│
├── 🌐 streamlit_app/
│   └── streamlit_app.py               # Interactive frontend UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

| Layer           | Technology                          |
|-----------------|--------------------------------------|
| Language        | Python 3.10+                         |
| ML Framework    | scikit-learn 1.5                     |
| Imbalance       | imbalanced-learn (SMOTE)             |
| API Backend     | FastAPI + Uvicorn                    |
| Validation      | Pydantic v2                          |
| Frontend        | Streamlit                            |
| Visualisation   | matplotlib, seaborn                  |
| Serialisation   | pickle                               |
| Notebook        | Jupyter / Google Colab               |

---

## 🚀 How to Run the Project

### Prerequisites
```bash
git clone https://github.com/your-username/vehicle-fraud-detection.git
cd vehicle-fraud-detection
pip install -r requirements.txt
```

### Step 1 — Train the Model

Open and run the notebook:
```bash
jupyter notebook notebook/vehicle_fraud_detection.ipynb
```
Or open directly in Google Colab. After running all cells, trained artifacts
will be saved to the `model/` directory.

### Step 2 — Start the FastAPI Backend

```bash
cd fastapi_app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 3 — Launch the Streamlit Frontend

```bash
cd streamlit_app
streamlit run streamlit_app.py
```

Opens automatically at: http://localhost:8501

---

## 🔌 API Usage

### `POST /predict`

Submit a claim for fraud assessment.

**Request Body (JSON):**
```json
{
  "policy_state": "CA",
  "policy_deductible": 500,
  "policy_annual_premium": 1200.0,
  "insured_age": 35,
  "insured_sex": "MALE",
  "insured_education_level": "College",
  "insured_occupation": "Manager",
  "insured_hobbies": "reading",
  "incident_type": "Single Vehicle Collision",
  "collision_type": "Rear",
  "incident_severity": "Total Loss",
  "authorities_contacted": "None",
  "incident_state": "CA",
  "incident_hour_of_the_day": 2,
  "number_of_vehicles_involved": 1,
  "bodily_injuries": 0,
  "witnesses": 0,
  "police_report_available": "No",
  "claim_amount": 45000.0,
  "total_claim_amount": 52000.0,
  "incident_month": 12,
  "incident_is_weekend": 1
}
```

**Response (JSON):**
```json
{
  "prediction": 1,
  "prediction_label": "FRAUD",
  "fraud_probability": 0.8923,
  "risk_level": "HIGH"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @- < payload.json
```

**Python Example:**
```python
import requests

payload = { ... }  # see above
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

### Other Endpoints

| Method | Endpoint       | Description                     |
|--------|---------------|---------------------------------|
| GET    | `/`           | Health check                    |
| GET    | `/health`     | Model load status                |
| GET    | `/model-info` | Model type & feature list        |
| GET    | `/docs`       | Interactive Swagger UI           |

---

## 📈 Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Gradient Boosting   | 0.8812   | 0.6234    | 0.8156 | 0.7068   | 0.8721  |
| Random Forest       | 0.8743   | 0.6089    | 0.8023 | 0.6925   | 0.8634  |
| SVM                 | 0.8521   | 0.5812    | 0.7934 | 0.6712   | 0.8412  |
| Logistic Regression | 0.8234   | 0.5456    | 0.7823 | 0.6428   | 0.8289  |
| Decision Tree       | 0.8012   | 0.5234    | 0.7456 | 0.6156   | 0.7823  |
| KNN                 | 0.7912   | 0.5012    | 0.7234 | 0.5923   | 0.7634  |
| Naive Bayes         | 0.7234   | 0.4512    | 0.7856 | 0.5734   | 0.7823  |

> **Best Model: Gradient Boosting** — highest ROC-AUC (0.8721) and Fraud Recall (0.8156)

---

## 🔬 Feature Engineering

| Feature                  | Rationale                                              |
|--------------------------|--------------------------------------------------------|
| `claim_to_premium_ratio` | High ratio signals claiming far beyond premium paid    |
| `is_high_severity`       | Total Loss claims are more likely to be fabricated     |
| `no_police_report`       | Absent report is a classic fraud red flag              |
| `no_authority_contacted` | Legitimate accidents involve authorities               |
| `incident_month`         | Seasonal fraud patterns detected in EDA                |
| `incident_is_weekend`    | Weekend incidents have different risk profiles         |
| `is_multi_vehicle`       | Single-vehicle crashes easier to stage                 |
| `claim_per_vehicle`      | Normalised claim size — controls for multi-car claims  |
| `risk_hobby`             | EDA-identified hobbies with elevated fraud rates       |

---

## ⚡ Key Design Decisions

- **SMOTE** over simple oversampling: Generates synthetic minority-class samples in feature space, not just duplicates.
- **`class_weight='balanced'`** on tree models: Secondary guard against imbalance without data duplication.
- **`StratifiedKFold`** in cross-validation: Preserves class ratio across folds for reliable evaluation.
- **Ordinal encoding** for education and severity: Respects the natural ordering of categories.
- **One-Hot encoding** for nominal features: Avoids imposing false ordinality.
- **Singleton model loading** (`@lru_cache`): Prevents redundant disk I/O on every API request.

---

## 🔮 Future Improvements

- [ ] **XGBoost / LightGBM** — often outperform sklearn GBM on tabular data
- [ ] **SHAP explainability** — per-claim feature attribution for auditors
- [ ] **Model versioning** with MLflow or DVC
- [ ] **Docker + docker-compose** for one-command deployment
- [ ] **Threshold tuning** — Youden's J or precision-recall trade-off optimisation
- [ ] **Real-time monitoring** — data drift detection with Evidently AI
- [ ] **Authentication** — API key middleware for production deployment
- [ ] **Async batch endpoint** — score multiple claims in one request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Saksham Ahluwal ia on Kaggle](https://www.kaggle.com/datasets/ahluwaliasaksham/car-insurance-fraud-detection-dataset)
- Built with ❤️ using scikit-learn, FastAPI, and Streamlit
