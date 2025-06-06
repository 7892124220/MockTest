import mysql.connector
# MYSQL CONNECTION TO PYTHON
"""mydb=mysql.connector.connect(
  host="localhost",
  user='root',
  password='system',
  database='Customer'
)

query="select * from traficrec"
print(query)
df = pd.read_sql(query, mydb)
print(df)
mydb.close()"""

#  Python
# Clean and Encode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset (exported from MySQL or directly from the source)
df = pd.read_csv('Customerproject.csv')
print(df)
#Step 2: Check Data types of all columns
print("Data types of all columns:")
print(df.dtypes)

# Handle missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# EDA

"""df['Churn']= df['Churn'].map({'Yes': 1, 'No': 0})  # Convert 'Yes'/'No' to 1/0
# Create Tenure Group
df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, np.inf], labels=['0-12', '13-24', '25-36', '37-48', '48+'])
# Churn Rate by Gender
gender_churn = df.groupby('gender')['Churn'].mean().reset_index()
gender_churn.columns = ['gender', 'Churn Rate']
print("\nChurn Rate by Gender:\n", gender_churn)


# Churn Rate by Tenure Group
tenure_churn = df.groupby('Tenure_Group')['Churn'].mean().reset_index()
tenure_churn.columns = ['Tenure Group', 'Churn Rate']
print("\nChurn Rate by Tenure Group:\n", tenure_churn)

# ====== Optional: Visualization ======
# Set plot style
sns.set(style='whitegrid')

# Gender plot
plt.figure(figsize=(6,4))
sns.barplot(x='gender', y='Churn Rate', data=gender_churn)
plt.title('Churn Rate by Gender')
plt.show()

# Tenure group plot
plt.figure(figsize=(6,4))
sns.barplot(x='Tenure Group', y='Churn Rate', data=tenure_churn)
plt.title('Churn Rate by Tenure Group')
plt.show()"""

# Visualize correlation between features and churn

# Correlation with churn only
# Select numeric columns (including Churn)
"""numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr()
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
print(churn_corr)

# Bar plot of correlation with churn
plt.figure(figsize=(8, 5))
churn_corr.drop('Churn').plot(kind='bar', color='skyblue')
plt.title('Correlation of Features with Churn')
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()"""
# You can add more demographic columns as needed

# Preprocess the data
# Encode categorical variables
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])
df['Churn'] = le.fit_transform(df['Churn'])  # Yes -> 1, No -> 0

# Handle missing values (if not already done in MySQL)
df.fillna(0, inplace=True)

# Logestic Regression and Random Forest Classifier for Churn Prediction
# Ensure 'TotalCharges' is numeric, convert if necessary

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)  # Fill NaN values with 0

# Define features and target
X = df[['tenure', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Scale numerical features
scaler = StandardScaler()
X[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")

evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, rf_pred, "Random Forest")
df.to_csv('Customerproject_cleaned.csv', index=False)  # Save cleaned data if needed
 




    





