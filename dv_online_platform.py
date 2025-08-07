import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('user_data.csv', delimiter='\t')

# Drop rows with too many missing values
df_clean = df.dropna(subset=['Age', 'CourseProgress', 'LastLoginDaysAgo'])

# Convert numerical columns
df_clean['Age'] = df_clean['Age'].astype(float)
df_clean['CourseProgress'] = df_clean['CourseProgress'].astype(float)
df_clean['LastLoginDaysAgo'] = df_clean['LastLoginDaysAgo'].astype(float)

#  1. Course Progress by Course Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='CourseCategory', y='CourseProgress', data=df_clean)
plt.title('Course Progress by Course Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  2. Distribution of Users by Enrolled Plan
plt.figure(figsize=(6, 6))
df_clean['EnrolledPlan'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Enrolled Plans')
plt.ylabel('')
plt.show()

#  3. Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df_clean)
plt.title('Gender Distribution')
plt.show()

#  4. Average Course Progress by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Gender', y='CourseProgress', data=df_clean)
plt.title('Average Course Progress by Gender')
plt.show()

#  5. Heatmap of Numerical Feature Correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df_clean[['Age', 'CourseProgress', 'LastLoginDaysAgo']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Features')
plt.show()
