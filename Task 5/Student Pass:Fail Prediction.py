import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Study Hours': [5, 7, 3, 8, 6, 9, 4, 7, 2, 8],
    'Attendance': [70, 85, 60, 90, 75, 95, 65, 80, 55, 88],
    'Pass': [1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print(df.describe())
sns.pairplot(df, hue='Pass')
plt.show()

X = df[['Study Hours', 'Attendance']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')




