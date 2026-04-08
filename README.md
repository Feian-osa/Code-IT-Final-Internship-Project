# Code-IT-Final-Internship-Project
This is probably the last project for internship in data science with python for Code-IT
# 🎓 Student Performance Prediction System

## 📌 Overview
This project is a machine learning-based system that predicts student academic performance (Pass/Fail) using various academic, behavioral, and demographic features.  

It includes:
- Data preprocessing & analysis  
- Machine learning model development  
- Feature importance analysis  
- Interactive dashboard using Streamlit  

---

## 🎯 Objectives
- Analyze how student habits affect performance  
- Build predictive models using machine learning  
- Compare model performance  
- Provide an interactive system for predictions  

---

## 📊 Dataset
The dataset contains student-related features such as:

- Age  
- Gender  
- Ethnicity  
- Parental Education  
- Study Time (weekly)  
- Absences  
- Tutoring  
- Parental Support  
- Extracurricular Activities  
- Sports, Music, Volunteering  
- GPA (used to generate result)

### 🎯 Target Variable
A new column **Result** is created:
- `1` → Pass (GPA ≥ 1.6)  
- `0` → Fail  

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## 🔍 Project Workflow

### 1. Data Preprocessing
- Handling missing values  
- Feature selection  
- Creating target variable (Result)  
- Feature scaling using StandardScaler  

---

### 2. Exploratory Data Analysis (EDA)
- Statistical summary of dataset  
- Correlation analysis  
- Visualization using Seaborn & Matplotlib  

---

### 3. Model Development

Two models were implemented:

#### 🔹 Logistic Regression
- Used for binary classification (Pass/Fail)

#### 🔹 Random Forest Classifier
- Used for better accuracy  
- Provides feature importance  

---

### 4. Model Evaluation

Metrics used:
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

### 📈 Results
| Model                | Accuracy |
|---------------------|---------|
| Logistic Regression | ~92.4%  |
| Random Forest       | ~91.8%  |

Logistic Regression performed slightly better in this case.

---

### 5. Feature Importance
Random Forest was used to identify important features.

Key influencing factors identified using Random Forest:


- Absences (strongest negative impact on performance)
- Study Time Weekly (higher study time improves results)
- Parental Support (positive influence on student success)
- Parental Education (moderate influence)
---

## 💻 Streamlit Dashboard

An interactive dashboard was developed using Streamlit with the following features:

### 📊 Dashboard Page
- Total students  
- Model accuracy comparison  
- Best model selection  
- Feature importance visualization  

### 📋 Students Page
- Full dataset display  
- High-risk (fail) students identification  

### 🤖 Prediction Page
Users can input:
- Study habits  
- Attendance  
- Lifestyle factors  

➡️ System predicts:
- ✅ Pass  
- ⚠️ At Risk  

---

## 🚀 How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/Feian-osa/Code-IT-Final-Internship-Project.git
cd Code-IT-Final-Internship-Project