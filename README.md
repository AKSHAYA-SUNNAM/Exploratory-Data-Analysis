# 📊 Student Learning Analytics - Data Visualization Project

This project analyzes student data from an online learning platform to understand user demographics, engagement, and progress trends through effective data visualizations.

## 📁 Files Included

- `online_platform.ipynb` – Jupyter Notebook with complete:
  - Data loading and preprocessing
  - Visualization using matplotlib and seaborn
  - Graph-wise conclusions using print statements
- `user_data.csv` – Tab-separated dataset containing user demographics, course engagement metrics, and login history

## 📌 Dataset Features

- **Demographics**: Age, Gender, Location
- **Course Info**: Course Category, Enrolled Plan
- **Engagement Metrics**: Course Progress (%), Last Login (Days Ago)

## 📊 Visualizations

1. **Boxplot** – Course progress by course category  
2. **Pie Chart** – Distribution of enrolled plans (Free vs Premium)  
3. **Count Plot** – Gender distribution  
4. **Bar Chart** – Average course progress by gender  
5. **Correlation Heatmap** – Age, login frequency, and progress

## ✅ Key Conclusions

- Students in **AI** and **Cloud** categories tend to show higher course progress.
- Majority of users are on the **Free plan**, suggesting potential for conversion to Premium.
- **Gender** distribution is fairly balanced, with **females slightly ahead** in average progress.
- **Frequent logins** are strongly associated with better course completion.
- **Older users** often display more consistent progress.

## 🛠 Tech Stack

- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn

## 🚀 How to Run

1. Open `online_platform.ipynb` in Jupyter Notebook or VS Code
2. Run all cells in order to view visualizations and insights
3. Ensure `user_data.csv` is in the same directory

