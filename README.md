# Student Performance Impact due to Social Media - Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to predict whether a student's academic performance is affected due to social media based on behavioral, academic, and lifestyle attributes. The project implements multiple supervised machine learning classification algorithms, evaluates their performance using standard metrics, and deploys the best-performing models using an interactive Streamlit web application.

---

## 2. Dataset Description

The dataset used in this project contains information about students’ social media usage patterns, lifestyle factors, and academic impact indicators.

**Dataset Characteristics**

* Number of instances: 1000+
* Number of features: 12+ (including engineered features)
* Target variable: **Affects_Academic_Performance  (No / Yes)**
* Feature types: Mixed (numerical and categorical)

**Example features**

* Age
* Gender
* Average daily usage hours
* Sleep hours per night
* Academic performance impact
* Usage-to-sleep ratio (engineered feature)

The dataset was preprocessed by:

* Handling missing values using median/mode imputation
* Feature engineering (Usage-to-Sleep Ratio)
* One-hot encoding of categorical variables
* Standard feature scaling

---

## 3. Models Used and Evaluation Metrics

The following classification models were implemented:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes (Gaussian)
* Random Forest (Ensemble)
* XGBoost (Ensemble)

Each model was evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 4. Model Performance Comparison

| ML Model            | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
|--------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.87     | 0.928485 | 0.847826  | 0.866667 | 0.857143 | 0.738053 |
| Decision Tree       | 0.69     | 0.691919 | 0.640000  | 0.711111 | 0.673684 | 0.381914 |
| KNN                 | 0.71     | 0.728081 | 0.710526  | 0.600000 | 0.650602 | 0.409977 |
| Naive Bayes         | 0.80     | 0.889697 | 0.765957  | 0.800000 | 0.782609 | 0.598070 |
| Random Forest       | 0.85     | 0.927475 | 0.840909  | 0.822222 | 0.831461 | 0.696499 |
| XGBoost             | 0.81     | 0.902222 | 0.782609  | 0.800000 | 0.791209 | 0.617061 |


---

## 5. Observations on Model Performance

| ML Model            | Observation about model performance |
|--------------------|-------------------------------------|
| Logistic Regression | Achieved the highest overall performance with strong AUC and balanced Precision–Recall, indicating that the relationship between features and the target variable is reasonably linear and well captured by the model. |
| Decision Tree       | Produced comparatively lower performance due to overfitting and sensitivity to data variations, resulting in reduced MCC and generalization capability. |
| KNN                 | Showed moderate performance; although Precision was acceptable, lower Recall indicates sensitivity to class boundaries and dependence on neighborhood selection. |
| Naive Bayes         | Demonstrated stable performance with good Recall and AUC, suggesting probabilistic assumptions worked reasonably well despite feature correlations. |
| Random Forest       | Delivered strong and consistent results across all metrics, indicating effective variance reduction through ensemble learning and improved generalization over single decision trees. |
| XGBoost             | Achieved competitive performance with high AUC and balanced Precision–Recall, showing the effectiveness of gradient boosting in capturing complex non-linear relationships in the dataset. |


---

## 6. Streamlit Application

The deployed Streamlit application provides:

* Dataset CSV upload functionality
* Model selection dropdown
* Evaluation metrics display
* Confusion matrix visualization and classification report

---

## 7. Repository Structure

```
project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── training.ipynb
│   └── saved_models/
│
└── data/
    └── student-social-media-academic-performance-test-data.csv
    └── student-social-media-academic-performance.csv
```

---

## 8. Deployment

The application is deployed using **Streamlit Community Cloud**. It enables users to interactively upload test datasets, select machine learning models, and view predictions and evaluation metrics.
