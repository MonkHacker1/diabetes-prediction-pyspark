# Diabetes Prediction with PySpark MLlib

A comprehensive machine learning pipeline for diabetes risk prediction implemented using PySpark MLlib. This project demonstrates end-to-end data science workflow from data preprocessing to model deployment with logistic regression classification.

## 📊 Project Overview

This project builds a predictive model to classify diabetes risk based on clinical parameters using Apache Spark's machine learning library. The implementation includes data cleaning, feature engineering, correlation analysis, and model evaluation with a focus on handling real-world medical dataset challenges.

## 🚀 Features

- **Data Preprocessing**: Automated handling of missing values and zero-value imputation
- **Feature Engineering**: Vector assembly and feature selection
- **Machine Learning**: Logistic regression implementation using PySpark MLlib
- **Model Evaluation**: Comprehensive metrics including AUC score, precision, and recall
- **Model Persistence**: Save and load trained models for production use
- **Scalable Processing**: Built on PySpark for handling large datasets

## 📁 Dataset

The project uses a diabetes dataset containing 2,000 patient records with the following features:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Class variable (0 - non-diabetic, 1 - diabetic)

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Java 8 or 11
- Apache Spark 3.0+

### Clone the Repository
```bash
git clone https://github.com/MonkHacker1/diabetes-prediction-pyspark.git
cd diabetes-prediction-pyspark
```
### Install Dependencies
```bash
pip install pyspark numpy pandas matplotlib seaborn
```
### 💻 Usage
Running the Jupyter Notebook
```bash
jupyter notebook Diabetes_Prediction.ipynb
```
## 📈 Model Performance

The trained logistic regression model achieves:
- **AUC Score**: 81.3%
- **Training Accuracy**: 92.7%
- **Comprehensive evaluation** with precision, recall, and F1-score metrics

## 🔧 Technical Details

### Data Preprocessing
- Zero-value imputation using mean values
- Correlation analysis for feature selection
- Data normalization and scaling
- Train-test split (70-30)

### Machine Learning
- Logistic Regression with PySpark MLlib
- Binary classification for diabetes outcome
- Hyperparameter tuning capabilities
- Model serialization and loading

### Evaluation Metrics
- Area Under ROC Curve (AUC)
- Precision, Recall, F1-Score
- Training and validation loss curves
- Confusion matrix analysis
## 🙏 Acknowledgments

- Dataset sourced from healthcare research repositories
- Built with PySpark and Apache Spark MLlib  
- Inspired by healthcare machine learning applications
