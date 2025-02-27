import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Create the Dataset
data = {
    'Age': [52, 45, 60, 55, 65, 50, 70, 48, 58, 62],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Cholesterol': [230, 200, 280, 210, 260, 190, 290, 220, 240, 250],
    'Blood Pressure': ['140/90', '130/85', '150/95', '135/88', '145/92', '125/80', '160/100', '140/90', '150/95', '155/98'],
    'Heart Disease': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file (optional)
df.to_csv('heart_disease.csv', index=False)

# Step 2: Data Preprocessing
# Handle categorical columns (Gender and Heart Disease)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Heart Disease'] = df['Heart Disease'].map({'Yes': 1, 'No': 0})

# Split Blood Pressure into Systolic and Diastolic
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop('Blood Pressure', axis=1, inplace=True)

# Check for missing values (optional, as this dataset is clean)
print(df.isnull().sum())

# Step 3: Feature Engineering
# Normalize numerical features (Age, Cholesterol, Systolic, Diastolic)
scaler = StandardScaler()
df[['Age', 'Cholesterol', 'Systolic', 'Diastolic']] = scaler.fit_transform(df[['Age', 'Cholesterol', 'Systolic', 'Diastolic']])

# Step 4: Split Data into Features and Target
X = df.drop('Heart Disease', axis=1)  # Features
y = df['Heart Disease']  # Target

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the Model (Optional)
import joblib
joblib.dump(model, 'heart_disease_model.pkl')