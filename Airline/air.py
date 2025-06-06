import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Step 1: Load the dataset
df= pd.read_csv('airline_on_time_performance_100_rows.csv')
print(df)

# Step 2: Check Data types of all columns
print("Data types of all columns:")
print(df.dtypes)
# Step 3: Identify columns with non-numeric data types
non_numeric_columns = df.select_dtypes(include=['object']).columns
print("\nColumns with non-numeric data types:")
print(non_numeric_columns)
# Step 4: Handle missing values
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Timestamp columns to datetime
df['FlightDate'] = pd.to_datetime(df['FlightDate'], format='%Y-%m-%d', errors='coerce')

# Fill missing values for numeric columns with mean
# EDA: Analyze and visualize delay distribution

# 1. Basic statistics for delay columns (e.g., DepartureDelay, ArrivalDelay)
print(df['DepartureDelay'].describe())
print(df['ArrivalDelay'].describe())
df.to_csv('cleaned_airline_data.csv', index=False)
# 2. Histogram of Departure Delay
plt.figure(figsize=(8, 5))
sns.histplot(df['DepartureDelay'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Departure Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()

# 3. Histogram of Arrival Delay
plt.figure(figsize=(8, 5))
sns.histplot(df['ArrivalDelay'].dropna(), bins=30, kde=True, color='salmon')
plt.title('Distribution of Arrival Delay')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.show()
# 4. Correlation heatmap for numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()




