# Final Project: Sloan Digital Sky Survey (SDSS) Data Analysis and Classification

## Project Overview

This project involves analyzing and modeling a dataset from the Sloan Digital Sky Survey (SDSS) Data Release 18 (DR18), containing 100,000 astronomical observations. The goal is to preprocess the data, perform exploratory analysis, and apply various machine learning models to classify observations into their respective categories.

## Team Members
- Eddie Cazares
- Katherine (Katie) Clark
- Aditya Behre

## Project Workflow
1. **Data Preparation**:
   - Load the dataset.
   - Handle missing values and duplicates.
   - Identify and remove outliers using the 1.5 IQR Rule.
   - Drop irrelevant or redundant features.

2. **Data Exploration**:
   - Perform descriptive statistical analysis.
   - Visualize the distribution of class labels.
   - Identify correlations between features using a correlation matrix.

3. **Feature Engineering**:
   - Select relevant features based on domain knowledge and exploratory findings.
   - Scale numerical features to improve model performance.

4. **Model Implementation**:
   - Train and evaluate various classification models:
     - Logistic Regression
     - Decision Tree Classifier
     - K-Nearest Neighbors (KNN)
     - Multi-Layer Perceptron (MLP/Neural Network)
     - Random Forest (Ensemble Classifier)
   - Compare model accuracies and select the best-performing model.

5. **Hyperparameter Tuning**:
   - Optimize hyperparameters for the best-performing model (MLPClassifier) using cross-validation.

## Key Findings
- Feature scaling and removal of irrelevant columns significantly improved model accuracy.
- MLPClassifier demonstrated the best performance among all models after hyperparameter tuning.

## Technical Details
- **Programming Language**: Python
- **Libraries Used**:
  - Data Manipulation: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
- **Dataset**: SDSS Data Release 18
