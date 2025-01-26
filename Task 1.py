import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "/Users/aaditya/Downloads/student-mat.csv"  
data = pd.read_csv(data_path, delimiter=';') 


print("First 5 rows of the dataset:")
print(data.head())


missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)


print("\nData types of each column:")
print(data.dtypes)


print("\nShape of the dataset:")
print(data.shape)


data = data.fillna(data.median(numeric_only=True))  


data = data.drop_duplicates()


average_score = data['G3'].mean()
print(f"The average score in math (G3) is: {average_score:.2f}")


students_above_15 = data[data['G3'] > 15].shape[0]
print(f"The number of students who scored above 15 in their final grade (G3) is: {students_above_15}")


correlation = data['studytime'].corr(data['G3'])
print(f"The correlation between study time and final grade (G3) is: {correlation:.2f}")


average_g3_by_gender = data.groupby('sex')['G3'].mean()
print("\nAverage G3 by gender:")
print(average_g3_by_gender)


plt.figure(figsize=(12, 10))


plt.subplot(3, 1, 1)
data['G3'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')


plt.subplot(3, 1, 2)
sns.scatterplot(x='studytime', y='G3', data=data, hue='sex')
plt.title('Study Time vs Final Grade (G3)')
plt.xlabel('Study Time')
plt.ylabel('Final Grade (G3)')


plt.subplot(3, 1, 3)
average_g3_by_gender.plot(kind='bar', color=['blue', 'pink'], edgecolor='black')
plt.title('Average Final Grade (G3) by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Final Grade (G3)')

plt.tight_layout()