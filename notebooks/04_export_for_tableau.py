import pandas as pd

df = pd.read_csv(r'C:\Users\Savita\Desktop\customer_churn_project\data\cleaned_churn.csv')

# Add back readable labels for Tableau (so charts look nice)
df_tableau = pd.read_csv(r'C:\Users\Savita\Desktop\customer_churn_project\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Save as Excel for Tableau
df_tableau.to_excel(
    r'C:\Users\Savita\Desktop\customer_churn_project\dashboard\churn_tableau.xlsx',
    index=False
)
print("Excel file saved to dashboard/churn_tableau.xlsx")