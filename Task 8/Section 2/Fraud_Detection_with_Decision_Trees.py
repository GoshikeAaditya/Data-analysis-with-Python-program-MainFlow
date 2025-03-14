import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)
data = {
    'Transaction ID': range(1, 1001),
    'Amount': np.random.uniform(10, 1000, 1000),
    'Type': np.random.choice(['credit', 'debit'], 1000),
    'Is Fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])  
}
df = pd.DataFrame(data)

print(df.head())

print(df.isnull().sum())

label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

df['Amount_Category'] = pd.cut(df['Amount'], bins=[0, 100, 500, 1000], labels=['Low', 'Medium', 'High'])

df['Amount_Category'] = label_encoder.fit_transform(df['Amount_Category'])

X = df.drop(['Transaction ID', 'Is Fraud'], axis=1)
y = df['Is Fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

print("\nRecommendations to improve fraud detection accuracy:")
print("- Use ensemble methods like Random Forest or Gradient Boosting.")
print("- Implement anomaly detection techniques to identify unusual patterns.")
print("- Collect more data to improve model training.")
print("- Perform more advanced feature engineering to capture complex patterns.")