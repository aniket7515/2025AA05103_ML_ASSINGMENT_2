📌 Bank Customer Churn Prediction
Machine Learning Assignment – 2

Programme: M.Tech (AIML/DSE)
Course: Machine Learning
Student ID: 2025AA05103

1️⃣ Problem Statement

Customer churn significantly affects profitability in the banking sector. Predicting whether a customer is likely to leave allows banks to take preventive action and improve retention strategies.

The objective of this project is to develop and compare multiple classification models to predict customer churn using demographic and financial attributes. The project also demonstrates an end-to-end machine learning pipeline including preprocessing, model evaluation, and deployment using Streamlit.

2️⃣ Dataset Description

The dataset used is the Bank Customer Churn Prediction Dataset obtained from Kaggle.

Dataset Overview

Total Records: ~10,000

Number of Features (raw): 13

Classification Type: Binary

Minimum Feature Requirement (≥12): ✔

Minimum Instance Requirement (≥500): ✔

Target Variable

Exited = 1 → Customer churned

Exited = 0 → Customer retained

Important Features

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

3️⃣ Data Cleaning & Feature Engineering
🔹 Data Cleaning

Removed identifier columns: CustomerId, Surname

Verified absence of missing values

Ensured data consistency

🔹 Categorical Encoding

One-hot encoding applied to:

Geography

Gender

Used drop_first=True to prevent multicollinearity

🔹 Feature Engineering

Two additional features were created:

Balance_to_Salary_Ratio
Captures financial pressure of customers.

Tenure_Age_Ratio
Normalizes tenure with respect to age to better represent loyalty behavior.

🔹 Feature Scaling

StandardScaler was applied to normalize numerical features.

Essential for Logistic Regression and KNN performance.

4️⃣ Models Implemented

The following six classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors

Naive Bayes (Gaussian)

Random Forest

XGBoost

All models were evaluated using the same train-test split and preprocessing pipeline.

5️⃣ Evaluation Metrics Used

Each model was evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

These metrics provide a balanced evaluation, especially for imbalanced churn datasets.

6️⃣ Model Performance Comparison
Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.6125	0.7756	0.3244	0.8354	0.4674	0.3151
KNN	0.8255	0.7740	0.5967	0.4398	0.5064	0.4102
Naive Bayes	0.8035	0.7875	0.5191	0.4668	0.4916	0.3710
XGBoost	0.8050	0.8347	0.5165	0.6536	0.5770	0.4578

(Values obtained from evaluation on processed test dataset through Streamlit app.)

7️⃣ Observations on Model Performance
Model	Observation
Logistic Regression	Achieved very high recall (0.835), meaning it detects most churners, but suffers from low precision and accuracy due to increased false positives.
KNN	Balanced performance with strong accuracy but moderate recall.
Naive Bayes	Stable and computationally efficient with balanced precision and recall.
XGBoost	Demonstrates the best overall trade-off between precision, recall, AUC, and MCC, making it the most reliable model for churn prediction.
Key Insight

For churn prediction, recall is critical because missing churners leads to business loss. Logistic Regression detects most churners, but XGBoost provides a better overall balance between precision and recall.

8️⃣ Streamlit Application Features

The deployed Streamlit web application includes:

✅ CSV upload option (processed test dataset)

✅ Model selection dropdown

✅ Display of evaluation metrics

✅ Confusion matrix

✅ Threshold-based prediction

The application allows interactive comparison of multiple models.

9️⃣ Deployment

The application is deployed on Streamlit Community Cloud using a GitHub repository containing:

app.py

requirements.txt

README.md

model artifacts (.pkl files)

🔟 Conclusion

This project demonstrates a complete machine learning lifecycle, from data preprocessing and feature engineering to model evaluation and cloud deployment. Ensemble models such as XGBoost showed strong overall performance, while Logistic Regression achieved the highest recall.

The study highlights the importance of selecting appropriate evaluation metrics in imbalanced classification problems like customer churn prediction.
