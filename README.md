# Rent Prediction using Machine Learning

This project aims to predict house rent prices based on various features such as area type, city, furnishing status, and floor levels. We implement several machine learning models, including **Linear Regression**, **Random Forest**, and **Support Vector Machine (SVM)**, to compare their performance in predicting rent prices.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)

## Project Overview

The project uses a dataset of house rentals and applies data preprocessing, exploratory data analysis, and machine learning techniques to predict the rent prices. The project is structured as follows:

1. **Data Exploration and Visualization**
2. **Data Preprocessing**
3. **Building Multiple Models**
4. **Model Evaluation and Comparison**
5. **Visualizing Predictions**

## Features

- **Data Exploration**: Visualize the distribution of categorical and numerical variables, identify outliers, and explore relationships between features.
- **Data Preprocessing**: Clean the dataset by handling missing values, encoding categorical variables, and normalizing the data.
- **Model Building**: Implement multiple machine learning models including:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **Support Vector Machine (SVM)**
- **Model Evaluation**: Evaluate the performance of models using metrics such as **Mean Absolute Error (MAE)** and **R² Score**.
- **Prediction Comparison**: Compare predicted rent values with actual values to assess model performance.

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Seaborn & Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning models and preprocessing

## How to Run the Project

### Prerequisites

Ensure you have Python installed with the following libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
###Running the Project
1. Clone the repository
2. Navigate to the project directory
3. Run the Jupyter notebook or Python script that contains the code:

## Results

- **Linear Regression**: While this model provides a reasonable **R² Score**, it suffers from higher errors when predicting house rents.
- **Random Forest**: Performs significantly better, achieving the lowest **MAE** and highest **R² Score** of 0.7482, indicating a strong fit to the data.
- **SVM**: This model struggles to predict rent accurately, with an **R² Score** of -0.176, meaning it performs poorly for this dataset.

## Model Comparison

| Model               | MAE (Mean Absolute Error) | R² Score |
|---------------------|---------------------------|----------|
| **Linear Regression** | 14,378.52                  | 0.5925   |
| **Random Forest**     | 9,175.37                   | 0.7482   |
| **SVM**               | 19,539.27                  | -0.1764  |

The **Random Forest** model outperforms the others in both **MAE** and **R² Score**, making it the best model for predicting rent prices in this dataset.

## Conclusion

The **Random Forest** model is the best-performing model for rent price prediction, showing significantly better results than both **Linear Regression** and **SVM**. Further improvements could involve hyperparameter tuning or adding more features to improve accuracy.

