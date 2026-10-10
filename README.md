# AquaSense — Smart Water Consumption Prediction System

A simple, educational machine-learning application that predicts household daily
water consumption using a Random Forest regression model.

> **Scope:** IBMSkillsBuild internship project. Uses a synthetic dataset. Engineering-grade water-management system.

---

## Project Structure

```
AquaSense_BOB_Project/
├── app.py                 # Streamlit UI and prediction interface
├── train_model.py         # Data loading, model training, evaluation, saving
├── generate_dataset.py    # Synthetic dataset generator
├── water_consumption.csv  # Generated dataset (500 rows, created at first run)
├── model.pkl              # Trained model artifact (created by train_model.py)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── PROJECT_REPORT.md      # Academic project report
└── .gitignore             # Excludes venv, cache, secrets
```

---

## Dataset

| Property           | Value                                  |
|--------------------|----------------------------------------|
| Type               | Synthetic (educational)                |
| Rows               | 500 households                         |
| Features           | 8 (see below)                          |
| Target variable    | `water_consumption_lpd` (litres / day) |
| Generation basis   | Aquastat / FAO, Ofwat, AWA guidelines  |

### Features

| Feature                | Description                              | Unit      |
|------------------------|------------------------------------------|-----------|
| `occupants`            | Number of people in the household        | count     |
| `property_size_m2`     | Floor area                               | m²        |
| `num_bathrooms`        | Number of bathrooms                      | count     |
| `has_dishwasher`       | Dishwasher present                       | 0 / 1     |
| `has_washing_machine`  | Washing machine present                  | 0 / 1     |
| `avg_temp_celsius`     | Annual average outdoor temperature       | °C        |
| `garden_size_m2`       | Outdoor irrigated area                   | m²        |
| `days_away_per_year`   | Days household is unoccupied per year    | days      |

> **Important:** This dataset is synthetic and intended only for educational
> demonstration. It is **not** real recorded household data.

---

## Setup

### Prerequisites

- Python 3.10 or later
- `pip` package manager

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the model

```bash
python train_model.py
```

This will:
1. Generate `water_consumption.csv` if it does not exist.
2. Train a Random Forest Regressor on 80 % of the data.
3. Evaluate on the remaining 20 % and print MAE, RMSE, and R².
4. Save the trained model to `model.pkl`.

### Step 2 — Run the application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Technologies

| Technology       | Version (minimum) | Purpose                          |
|------------------|-------------------|----------------------------------|
| Python           | 3.10              | Core language                    |
| Pandas           | 2.0               | Data handling                    |
| Scikit-learn     | 1.4               | Machine learning (Random Forest) |
| Streamlit        | 1.32              | Web application interface        |
| Matplotlib       | 3.8               | Dataset visualisation            |
| Joblib           | 1.3               | Model serialisation              |

---

## Model Performance

From a single training run (80 % train / 20 % test, `random_state = 42`):

| Metric | Linear Regression (baseline) | Random Forest (selected) |
|--------|------------------------------|--------------------------|
| MAE    | 34.16 L/day                  | **25.21 L/day**          |
| RMSE   | 41.22 L/day                  | **35.04 L/day**          |
| R²     | 0.9601                       | **0.9712**               |

> Run `python train_model.py` to reproduce these results in your environment.

---

## Limitations

- The dataset is synthetic; real-world performance will differ.
- The model does not capture all factors (tariff structure, appliance age,
  behavioural patterns, local climate beyond temperature, etc.).
- Predictions for households very different from the training distribution may
  be less reliable.
- The application is not suitable for water-leak detection, IoT integration,
  real-time monitoring, or engineering decisions.

---

## GitHub & Deployment

### Upload to GitHub

```bash
git init
git add .
git commit -m "Initial commit — AquaSense v1.0"
git remote add origin https://github.com/<your-username>/aquasense.git
git push -u origin main
```

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (including `requirements.txt`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select your repository, set `app.py` as the main file.
4. Click **Deploy**.

> **Before deploying:** Run `python train_model.py` locally, then commit
> `model.pkl` and `water_consumption.csv` to the repository so that the
> deployed app does not need to retrain on startup.
> Check that `model.pkl` and `water_consumption.csv` are not listed in
> `.gitignore` if you intend to commit them.

---

## Licence

This project is released for educational purposes. The synthetic dataset was
generated by the project code and is not subject to third-party licences.
