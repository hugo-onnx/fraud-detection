import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve


X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_save_eval(model, name):
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name}_best.pkl")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    # metrics
    metrics = {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_confusion_matrix.png", bbox_inches='tight')
    plt.clf()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} AUC={metrics['ROC-AUC']:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(f"results/{name}_roc.png", bbox_inches='tight')
    plt.clf()

    return metrics


# Load best params
with open("results/logreg_best_params.json") as f:
    log_params = json.load(f)
with open("results/rf_best_params.json") as f:
    rf_params = json.load(f)
with open("results/xgb_best_params.json") as f:
    xgb_params = json.load(f)

models = {
    "LogisticRegression": LogisticRegression(**log_params, class_weight='balanced', max_iter=10000),
    "RandomForest": RandomForestClassifier(**rf_params, class_weight='balanced', n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(**xgb_params, eval_metric='logloss', n_jobs=-1)
}

results = []
for name, model in models.items():
    print("Training:", name)
    m = train_save_eval(model, name)
    m["Model"] = name
    results.append(m)

pd.DataFrame(results).to_csv("results/model_comparison.csv", index=False)
print("Training & evaluation done. Models saved to models/ and metrics to results/")