import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load Data ──────────────────────────────────────────────
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_path, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

print("=" * 50)
print("STEP 1 — Basic Info")
print("=" * 50)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.dtypes)

# ── Step 2: Check Missing Values ───────────────────────────
print("\n" + "=" * 50)
print("STEP 2 — Missing Values")
print("=" * 50)
print(df.isnull().sum())

# ── Step 3: Fix TotalCharges column ───────────────────────
# It's stored as text (object) but should be a number — fix it
print("\n" + "=" * 50)
print("STEP 3 — Fix TotalCharges column")
print("=" * 50)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Null values in TotalCharges after fix: {df['TotalCharges'].isnull().sum()}")

# Fill the nulls with 0 (these are new customers with no charges yet)
df['TotalCharges'].fillna(0, inplace=True)
print("Nulls filled with 0 ✓")

# ── Step 4: Drop customerID (not useful for ML) ────────────
df.drop(columns=['customerID'], inplace=True)
print("\nDropped customerID column ✓")

# ── Step 5: Fix Churn column → 0 and 1 ────────────────────
print("\n" + "=" * 50)
print("STEP 4 — Convert Churn to 0/1")
print("=" * 50)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {df['Churn'].mean()*100:.2f}%")

# ── Step 6: Check all column values ───────────────────────
print("\n" + "=" * 50)
print("STEP 5 — Unique values in each column")
print("=" * 50)
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")

# ── Step 7: Save cleaned data ─────────────────────────────
cleaned_path = os.path.join(base_path, 'data', 'cleaned_churn.csv')
df.to_csv(cleaned_path, index=False)
print(f"\n✅ Cleaned data saved to: {cleaned_path}")

# ── Step 8: EDA Charts ────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6 — Generating EDA Charts")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Churn - Exploratory Data Analysis', fontsize=16)

# Chart 1 — Churn Distribution
churn_counts = df['Churn'].value_counts()
axes[0,0].bar(['No Churn', 'Churned'], churn_counts.values, color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Churn Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Chart 2 — Churn by Gender
gender_churn = df.groupby('gender')['Churn'].mean() * 100
axes[0,1].bar(gender_churn.index, gender_churn.values, color=['#3498db','#e91e63'])
axes[0,1].set_title('Churn Rate by Gender (%)')
axes[0,1].set_ylabel('Churn %')
for i, v in enumerate(gender_churn.values):
    axes[0,1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Chart 3 — Churn by Contract Type
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
axes[0,2].bar(contract_churn.index, contract_churn.values, color=['#e74c3c','#f39c12','#2ecc71'])
axes[0,2].set_title('Churn Rate by Contract (%)')
axes[0,2].set_ylabel('Churn %')
for i, v in enumerate(contract_churn.values):
    axes[0,2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Chart 4 — Monthly Charges Distribution
axes[1,0].hist(df[df['Churn']==0]['MonthlyCharges'], bins=30, alpha=0.7, color='#2ecc71', label='No Churn')
axes[1,0].hist(df[df['Churn']==1]['MonthlyCharges'], bins=30, alpha=0.7, color='#e74c3c', label='Churned')
axes[1,0].set_title('Monthly Charges Distribution')
axes[1,0].set_xlabel('Monthly Charges ($)')
axes[1,0].legend()

# Chart 5 — Tenure Distribution
axes[1,1].hist(df[df['Churn']==0]['tenure'], bins=30, alpha=0.7, color='#2ecc71', label='No Churn')
axes[1,1].hist(df[df['Churn']==1]['tenure'], bins=30, alpha=0.7, color='#e74c3c', label='Churned')
axes[1,1].set_title('Tenure Distribution (months)')
axes[1,1].set_xlabel('Tenure (months)')
axes[1,1].legend()

# Chart 6 — Churn by Internet Service
internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
axes[1,2].bar(internet_churn.index, internet_churn.values, color=['#9b59b6','#3498db','#1abc9c'])
axes[1,2].set_title('Churn Rate by Internet Service (%)')
axes[1,2].set_ylabel('Churn %')
for i, v in enumerate(internet_churn.values):
    axes[1,2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(base_path, 'data', 'eda_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Charts saved to: {chart_path}")
