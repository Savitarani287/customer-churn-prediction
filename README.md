#  ChurnShield – Customer Churn Prediction

ChurnShield is a machine learning project built to predict which telecom customers are likely to leave a company. The main goal is to help businesses identify customers who may stop using their services and take steps to retain them.

The project covers the complete machine learning process, starting from data cleaning and exploratory data analysis to model training, evaluation, and deployment. It also includes an interactive Tableau dashboard and a Streamlit web application where users can enter customer details and get churn predictions.

** Live App:** [Customer Churn Shield](https://customer-churn-shield.streamlit.app/)  


---

##  Problem Statement

Customer retention is an important challenge for telecom companies. Losing existing customers can affect revenue, and finding new customers often requires more time and money.

The purpose of this project is to analyze customer data, understand the factors behind customer churn, and build a machine learning model that can predict customers who are likely to leave. These predictions can help businesses plan better customer retention strategies.

##  Dataset Overview

The project uses a telecom customer dataset containing:

- **Total Records:** 7,043 customers
- **Total Features:** 21 attributes
- **Target Variable:** Churn (Yes/No)

The dataset includes information about customer tenure, contract type, monthly charges, payment methods, and subscribed services.

##  Model Performance

I trained and compared three machine learning algorithms to find the model that performed best on the dataset.

| Evaluation Metric | Result |
|---|---|
| Best Performing Model | Random Forest |
| Accuracy | **79%** |
| ROC-AUC Score | **0.82** |

**Models Tested:**
- Random Forest
- Logistic Regression
- XGBoost

Random Forest was selected based on the evaluation results, achieving 79% accuracy and a ROC-AUC score of 0.82.

##  Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Data Visualization | Tableau Public, Matplotlib, Seaborn |
| Web Application | Streamlit |
| Version Control | Git, GitHub |

##  Project Workflow

1. **Data Cleaning:** Checked for missing values, corrected data types, and prepared the dataset for analysis.
2. **Exploratory Data Analysis:** Explored customer information to understand churn patterns based on contract type, tenure, payment methods, and other factors.
3. **Feature Engineering:** Prepared categorical and numerical features using label encoding and feature scaling.
4. **Model Building:** Trained Random Forest, Logistic Regression, and XGBoost models.
5. **Model Evaluation:** Compared model performance using accuracy, ROC-AUC score, and confusion matrices.
6. **Dashboard Development:** Created an interactive Tableau dashboard to present customer churn trends and important insights.
7. **Deployment:** Developed and deployed a Streamlit application that allows users to enter customer details and receive churn predictions.

##  Project Structure

```text
ChurnShield/
│
├── dashboard/
│   └── churn_tableau.xlsx       # Tableau dashboard data
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
│   ├── cleaned_churn.csv        # Cleaned dataset
│   ├── confusion_matrix.png     # Model evaluation
│   ├── eda_charts.png           # Exploratory data analysis
│   ├── feature_importance.png   # Important model features
│   ├── model_comparison.png     # Model performance comparison
│   └── roc_curve.png            # ROC curve visualization
│
├── models/
│   ├── best_model.pkl           # Trained machine learning model
│   ├── encoders.pkl             # Saved feature encoders
│   ├── feature_names.pkl        # Model feature names
│   └── scaler.pkl               # Saved feature scaler
│
├── notebooks/
│   ├── 01_data_exploration.py   # Exploratory data analysis
│   ├── 02_data_cleaning.py      # Data cleaning and preprocessing
│   ├── 03_machine_learning.py   # Model training and evaluation
│   └── 04_export_for_tableau.py # Export data for Tableau
│
├── app.py                       # Streamlit web application
├── requirements.txt             # Required Python libraries
└── README.md                    # Project documentation
```

##  Future Improvements

There are several ways this project can be improved further:

- Apply SMOTE to handle class imbalance.
- Use GridSearchCV for hyperparameter tuning.
- Add SHAP to understand which features influence individual predictions.
- Display the main factors contributing to a customer's churn risk.
- Further improve model performance through feature selection and experimentation.

##  About Me

**Savita Rani**  
B.Tech Computer Science (AI & Data Science)  


 [LinkedIn Profile](https://www.linkedin.com/in/savita-rani-28394328b/)

---

 If you find this project useful, feel free to star the repository!
