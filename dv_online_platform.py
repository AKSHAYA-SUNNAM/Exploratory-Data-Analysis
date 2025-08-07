# 📦 Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🧾 Load Data
df = pd.read_csv('user_data.csv', delimiter='\t')

# 🧹 Data Cleaning
df_clean = df.dropna(subset=['Age', 'CourseProgress', 'LastLoginDaysAgo'])
df_clean['Age'] = df_clean['Age'].astype(float)
df_clean['CourseProgress'] = df_clean['CourseProgress'].astype(float)
df_clean['LastLoginDaysAgo'] = df_clean['LastLoginDaysAgo'].astype(float)

# ✅ Set seaborn style
sns.set(style="whitegrid")

# 📊 1. Boxplot – Course Progress by Course Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='CourseCategory', y='CourseProgress', data=df_clean)
plt.title('Course Progress by Course Category')
plt.xlabel('Course Category')
plt.ylabel('Course Progress (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🥧 2. Pie Chart – Enrolled Plan Distribution
plt.figure(figsize=(6, 6))
df_clean['EnrolledPlan'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Enrolled Plans')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 👥 3. Count Plot – Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df_clean)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 📈 4. Bar Chart – Average Course Progress by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Gender', y='CourseProgress', data=df_clean, ci=None)
plt.title('Average Course Progress by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Progress (%)')
plt.tight_layout()
plt.show()

# 🔥 5. Correlation Heatmap – Numerical Feature Correlation
plt.figure(figsize=(6, 4))
correlation = df_clean[['Age', 'CourseProgress', 'LastLoginDaysAgo']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numerical Features')
plt.tight_layout()
plt.show()
