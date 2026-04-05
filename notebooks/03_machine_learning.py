import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────
DATA_DIR  = r'C:\Users\Savita\Desktop\customer_churn_project\data'
MODEL_DIR = r'C:\Users\Savita\Desktop\customer_churn_project\models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Cleaned Data ──────────────────────────────────────
df = pd.read_csv(rf'{DATA_DIR}\cleaned_churn.csv')

print("=" * 50)
print("STEP 1 — Encode Categorical Columns")
print("=" * 50)

# ── Encode all text columns ────────────────────────────────
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
print(f"Columns to encode: {categorical_cols}")
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Fix NaN values (THIS was causing the error) ────────────
print(f"\nNaN values before fix: {df.isnull().sum().sum()}")
df.fillna(0, inplace=True)
print(f"NaN values after fix: {df.isnull().sum().sum()} ✓")

print("\nEncoded data preview:")
print(df.head(3))

# ── Step 2: Split features and target ─────────────────────
print("\n" + "=" * 50)
print("STEP 2 — Split Features & Target")
print("=" * 50)

X = df.drop(columns=['Churn'])
y = df['Churn']
print(f"Features shape: {X.shape}")
print(f"Target shape:   {y.shape}")

# ── Step 3: Train/Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")

# ── Step 4: Scale features ─────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✓ Features scaled!")

# ── Step 5: Train 3 Models ────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3 — Training Models...")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'model': model, 'accuracy': acc, 'auc': auc,
                     'y_pred': y_pred, 'y_prob': y_prob}
    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")

# ── Step 6: Model Comparison Chart ────────────────────────
print("\n" + "=" * 50)
print("STEP 4 — Generating Charts")
print("=" * 50)

names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]
aucs  = [results[n]['auc']      * 100 for n in names]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - 0.2, accs, 0.35, label='Accuracy %', color='#3498db')
bars2 = ax.bar(x + 0.2, aucs, 0.35, label='ROC-AUC %',  color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(60, 100)
ax.set_title('Model Comparison — Accuracy vs ROC-AUC')
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3, f'{bar.get_height():.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3, f'{bar.get_height():.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(rf'{DATA_DIR}\model_comparison.png', dpi=150)
plt.show()
print("✓ Model comparison chart saved!")

# ── Step 7: Confusion Matrix ───────────────────────────────
best_name = max(results, key=lambda n: results[n]['auc'])
best      = results[best_name]
print(f"\n✓ Best Model: {best_name}  (AUC: {best['auc']:.4f})")

cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churned'],
            yticklabels=['No Churn', 'Churned'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(rf'{DATA_DIR}\confusion_matrix.png', dpi=150)
plt.show()
print("✓ Confusion matrix saved!")

print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['No Churn', 'Churned']))

# ── Step 8: ROC Curve ─────────────────────────────────────
plt.figure(figsize=(7, 5))
for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All Models')
plt.legend()
plt.tight_layout()
plt.savefig(rf'{DATA_DIR}\roc_curve.png', dpi=150)
plt.show()
print("✓ ROC curve saved!")

# ── Step 9: Feature Importance ────────────────────────────
xgb_model   = results['XGBoost']['model']
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top10       = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top10.plot(kind='bar', color='#9b59b6')
plt.title('Top 10 Important Features — XGBoost')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(rf'{DATA_DIR}\feature_importance.png', dpi=150)
plt.show()
print("✓ Feature importance chart saved!")
print("\nTop 10 Features:")
print(top10)

# ── Step 10: Save Model ───────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5 — Saving Best Model")
print("=" * 50)

with open(rf'{MODEL_DIR}\best_model.pkl', 'wb') as f:
    pickle.dump(best['model'], f)
with open(rf'{MODEL_DIR}\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(rf'{MODEL_DIR}\feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print(f"✓ Best model saved  : models/best_model.pkl")
print(f"✓ Scaler saved      : models/scaler.pkl")
print(f"✓ Feature names saved: models/feature_names.pkl")
print(f"\n🎉 Day 2 Complete! Best model: {best_name}")
print(f"   Accuracy: {best['accuracy']*100:.2f}%  |  AUC: {best['auc']:.4f}")