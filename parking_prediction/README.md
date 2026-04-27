# 🅿️ Parking Demand Prediction for Alumni Meet & Conferences

> An end-to-end Machine Learning project that forecasts vehicle parking demand for large institutional events using 21 input features and 6 regression models.

---

## 🎯 Problem Statement

Large events like Alumni Meets, Tech Summits, and Academic Conferences attract hundreds to thousands of attendees. Poor parking management leads to:
- Traffic congestion and long queues
- Visitor frustration and safety hazards
- Underutilization or overflow of parking lots
- Wasted manpower and infrastructure

This project builds a predictive system to **accurately forecast parking demand** before the event, enabling proactive planning.

---

## 📁 Project Structure

```
parking_prediction/
├── app.py                          # 🌐 Streamlit Web Application
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 This file
├── data/
│   └── parking_dataset.csv         # 📊 Synthetic dataset (2000 samples)
├── models/
│   └── parking_model.pkl           # 🤖 Trained models + artifacts
└── utils/
    ├── data_generator.py           # 🔧 Dataset generation script
    └── model_trainer.py            # 🏋️ Model training & evaluation
```

---

## 📊 Dataset Description

**2000 synthetic samples** generated with domain-informed rules.

### Input Features (21)

| Feature | Type | Description |
|---|---|---|
| `event_type` | Categorical | Alumni Meet, Conference, Tech Summit, etc. |
| `venue` | Categorical | Auditorium, Convention Center, Sports Complex, etc. |
| `city` | Categorical | Delhi, Mumbai, Bangalore, Pune, etc. |
| `month` | Numeric | 1–12 |
| `day_of_week` | Numeric | 0=Mon … 6=Sun |
| `is_weekend` | Binary | 1 if Saturday/Sunday |
| `is_holiday` | Binary | 1 if public holiday |
| `expected_attendees` | Numeric | Registered/expected footfall |
| `duration_hours` | Numeric | 2, 3, 4, 6, 8, or 12 hours |
| `registration_mode` | Categorical | Online / Offline / Both |
| `is_paid_event` | Binary | 1 if entry fee applicable |
| `num_sessions` | Numeric | Number of parallel sessions |
| `has_vip_guests` | Binary | Presence of chief guests |
| `alumni_batch_size` | Numeric | Size of alumni cohort (0 if not alumni event) |
| `weather` | Categorical | Sunny, Rainy, Cloudy, Foggy, Partly Cloudy |
| `is_rainy` | Binary | 1 if weather is Rainy |
| `temperature_celsius` | Numeric | Ambient temperature |
| `total_parking_capacity` | Numeric | Max available slots |
| `public_transport_score` | Numeric | 1–10 scale |
| `distance_from_city_center_km` | Numeric | Venue remoteness |
| `carpooling_promoted` | Binary | 1 if carpooling encouraged |

### Target Variable

| Variable | Description |
|---|---|
| `parking_demand` | Number of vehicles needing parking |

---

## 🤖 ML Models Compared

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | Regularized linear regression |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble of 150 trees |
| Gradient Boosting | Sequential boosting (best performer) |
| XGBoost | GPU-accelerated gradient boosting |

**Evaluation Metrics:** MAE, RMSE, R², MAPE

---

## 🚀 How to Run

### 1. Clone / Download the project

```bash
cd parking_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate data & train models

```bash
python utils/model_trainer.py
```

This will:
- Generate `data/parking_dataset.csv`
- Train all 6 models
- Save `models/parking_model.pkl`

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📈 Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Gradient Boosting | ~0.953 | ~63 | ~98 |
| XGBoost | ~0.951 | ~64 | ~99 |
| Random Forest | ~0.927 | ~81 | ~122 |
| Ridge Regression | ~0.931 | ~83 | ~118 |
| Linear Regression | ~0.931 | ~83 | ~118 |
| Decision Tree | ~0.862 | ~117 | ~167 |

---

## 🖥️ App Features

- **🎯 Predict Tab** — Fill sidebar inputs → get instant prediction with gauges, alerts, and sensitivity charts
- **📊 Dataset Insights** — Box plots, bar charts, heatmaps, scatter plots
- **🏆 Model Performance** — Side-by-side R², MAE, RMSE comparison + feature importance
- **📖 Project Info** — Full methodology, structure, and run instructions

---

## 👨‍💻 Tech Stack

- **Python 3.10+**
- **Scikit-learn** — ML algorithms and preprocessing
- **XGBoost** — Gradient boosted trees
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Web application framework
- **Plotly** — Interactive visualizations

---

## 📝 Notes

- The dataset is **synthetic** but generated with realistic domain logic (e.g., rainy weather reduces driving, carpooling reduces demand, remoter venues see more driving)
- For real-world deployment, replace `data/parking_dataset.csv` with actual historical event parking records
- Model can be retrained by simply running `python utils/model_trainer.py` again

---

*Project: Parking Demand Prediction for Alumni Meet and Conferences*
