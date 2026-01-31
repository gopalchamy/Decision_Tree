
This project demonstrates the application of **Decision Tree models** for both **classification and regression** using two real-world datasets.  
The goal is to understand decision tree construction, splitting criteria (Gini Index and Entropy), overfitting issues, pruning techniques, and model interpretability through visualization and feature importance.

The project was developed using **Google Colab**, and the final code and notebooks are maintained in **GitHub**.

Project Overview:

Decision Trees are intuitive machine learning models used for predictive analytics in both classification and regression tasks.  
Although simple to understand, decision trees can easily overfit the training data. This project addresses this challenge by applying **pre-pruning and post-pruning techniques** and analyzing their impact on model performance.

Two separate real-world datasets are used:
- **Employee dataset** for classification
- **Housing dataset** for regression

---

## 📂 Datasets Used

### 1️⃣ Employee Dataset (Classification)

This dataset contains employee-related information used to predict **employee attrition**.

#### Target Variable:
- `left`  
  - `0` → Employee stayed  
  - `1` → Employee left  

#### Features Include:
- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `promotion_last_5years`
- `salary`
- `department`

#### Objective:
To build a **Decision Tree Classification model** that predicts whether an employee is likely to leave the organization.

---

### 2️⃣ Housing Dataset (Regression)

This dataset is based on California Census housing data and is used to predict housing prices.

#### Target Variable:
- `median_house_value` (continuous numerical value)

#### Features Include:
- `median_income`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `latitude`
- `longitude`
- `ocean_proximity` (categorical)

#### Objective:
To build a **Decision Tree Regression model** that predicts the median house value based on demographic and geographic features.

---

## 🎯 Project Objectives

- Implement Decision Tree models for:
  - Classification (Employee Attrition)
  - Regression (House Price Prediction)
- Understand and apply:
  - Gini Index
  - Entropy
- Handle overfitting using:
  - Pre-pruning
  - Post-pruning (Cost Complexity Pruning)
- Visualize decision trees
- Interpret feature importance

---

## 🛠️ Technologies Used

- Python  
- Google Colab  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

📊 Tasks Performed

🔹 Decision Tree Classification (Employee Dataset)
- Model: `DecisionTreeClassifier`
- Splitting Criteria:
  - Gini Index
  - Entropy
- Evaluation Metric:
  - Accuracy
- Visualization of decision tree to interpret employee attrition patterns

---

🔹 Decision Tree Regression (Housing Dataset)
- Model: `DecisionTreeRegressor`
- Evaluation Metrics:
  - Mean Squared Error (MSE)
  - R² Score
- Analysis of overfitting in unpruned trees
- Improved performance using pruning techniques

---

 ✂️ Pruning Techniques Applied

### Pre-Pruning
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`

### Post-Pruning
- Cost Complexity Pruning using `ccp_alpha`

Pruning helped reduce overfitting and improved model generalization.

---

🌳 Visualization & Interpretability

- Decision trees visualized using `plot_tree`
- Feature importance extracted from trained models
- Key insights:
  - Employee satisfaction strongly influences attrition
  - Median income is a major factor in housing prices



