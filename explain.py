"""SHAP global analysis — generates bar and beeswarm plots."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from utils import preprocess

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    # 1. Load model
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    # 2. Re-create the same test split used during training
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess(df)
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 4. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Bar chart — mean |SHAP| per feature
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150)
    plt.close()
    print(f"Saved → {OUTPUT_DIR / 'shap_bar.png'}")

    # 6. Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()
    print(f"Saved → {OUTPUT_DIR / 'shap_beeswarm.png'}")

    # 7. Print top-5 features by mean |SHAP|
    import numpy as np
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top5_idx = mean_abs_shap.argsort()[::-1][:5]
    print("\nTop 5 Most Important Features (mean |SHAP|):")
    print("-" * 45)
    for rank, idx in enumerate(top5_idx, start=1):
        print(f"  {rank}. {feature_names[idx]:25s}  {mean_abs_shap[idx]:.4f}")


if __name__ == "__main__":
    main()
