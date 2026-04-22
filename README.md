# 📉 Customer Churn Predictor

An end-to-end machine learning project that predicts customer churn for a telecom company using **XGBoost**, explains predictions with **SHAP**, and serves everything through an interactive **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white)

---

## 🗂️ Project Structure

```
churn-predictor/
├── data/                   # Place Kaggle CSV here
├── outputs/                # SHAP plots (auto-generated)
├── utils.py                # Shared preprocessing logic
├── train.py                # Model training pipeline
├── explain.py              # SHAP global analysis & plots
├── app.py                  # Streamlit dashboard (2 tabs)
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 📊 Dataset

This project uses the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset from Kaggle.

**Setup:** Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it inside the `data/` folder.

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/SOUMYAJYOTI1234/churn-predictor.git
cd churn-predictor

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

Run the following commands **in order**:

### 1. Train the model

```bash
python train.py
```

Trains an XGBClassifier on the Telco dataset and saves:
- `model.pkl` — trained model
- `feature_names.pkl` — list of feature names

Prints a classification report and ROC-AUC score on the test set.

### 2. Generate SHAP explanations

```bash
python explain.py
```

Produces two plots in the `outputs/` folder:
- `shap_bar.png` — global feature importance (mean |SHAP|)
- `shap_beeswarm.png` — beeswarm plot showing direction of impact

Also prints the top 5 most important features.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Opens a dashboard with two tabs:

| Tab | Description |
|-----|-------------|
| **🔮 Predict** | Enter customer details and get a churn probability with a per-customer SHAP waterfall explanation |
| **🌍 Global Insights** | View pre-generated SHAP bar and beeswarm plots with descriptions |

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBClassifier |
| Estimators | 300 |
| Max Depth | 4 |
| Learning Rate | 0.05 |
| Class Imbalance Handling | `scale_pos_weight` (auto-computed) |
| Train/Test Split | 80/20, stratified |
| Eval Metric | Log Loss |

## 🔍 Explainability

This project uses [SHAP (SHapley Additive exPlanations)](https://github.com/shap/shap) to provide:

- **Global explanations** — which features matter most across all customers
- **Local explanations** — why the model made a specific prediction for an individual customer (waterfall plot in the Streamlit app)

## 🛠️ Tech Stack

- **ML:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Data:** Pandas

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
