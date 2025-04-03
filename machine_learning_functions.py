#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:05:54 2024

@author: quentin
"""

"""#=============================================================================
   #=============================================================================
   #=============================================================================

    Dictionnary of Machine learning functions.

#=============================================================================
   #=============================================================================
   #============================================================================="""


import numpy as np

from scipy.stats import chi2_contingency, f_oneway, zscore
from scipy import stats

from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer

from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel, RFE, RFECV

from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn import linear_model as lm, tree, neighbors

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,  explained_variance_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVC, SVR

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from termcolor import colored

import matplotlib.pyplot as plt

import Exploratory_Data_Analysis.debug_dash_infos as ddi
import Exploratory_Data_Analysis.app_state as aps


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def make_regression_model(data_for_plot,x,y,weights,reg_type,reg_order,test_size_val):

    """
    Goal: Make ML regression and add the predictive value in the dataset.

    Parameters:
    - data_for_plot: Dataframe.
    - x: Feature matrix where each column in x corresponds to a different feature, and each row corresponds to an individual data point.
    - y: Target vector containing the values that correspond to each observation in x.
    - weights: Sample weights for each observation in the regression analysis.
    - reg_type: Type of regression for the data.
    - reg_order: Order of the regression for the data.
    - test_size_val: The ratio of testing value for the fit.

    Returns:
    - data_for_plot: Dataframe updated with the ML regression predictive values.
    """    
    
    Debug = aps.Debug
    
    print("") 
    print("Make ML "+reg_type) 
    print("") 
    
    # Split the data
    if weights is not None:
        x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(x, y, weights, test_size=test_size_val, random_state=0)
        # Invert weights (take care with zero values)
        epsilon = 1e-8  # Small constant to avoid division by zero
        weights_train = 1 / (weights_train + epsilon)
        weights_test = 1 / (weights_test + epsilon)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_val, random_state=0)

    Dict_regression_models = {
        'Linear Regression': lm.LinearRegression,
        'Decision Tree': tree.DecisionTreeRegressor,
        'k-NN': neighbors.KNeighborsRegressor,
        'Polynomial Regression': lambda degree: make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), lm.LinearRegression())  # Use a lambda to return a new instance
        }    

    # Handle Polynomial Regression with cross-validation
    if reg_type == 'Polynomial Regression':
        if reg_order is None:
            best_degree = None
            best_score = float('inf')  # Initialize with infinity as we want to minimize the score
            
            for degree in range(1, 7):  # Testing degrees from 1 to 6
                model = Dict_regression_models[reg_type](degree)
                cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')  # 5-fold CV

                # Calculate mean of the negative MSE (to minimize it, hence the negative sign)
                mean_cv_score = -cv_scores.mean()

                ddi.debug_print(f'Degree {degree} - CV Mean Squared Error: {mean_cv_score}', debug=Debug) 

                # Check if this is the best degree found
                if mean_cv_score < best_score:
                    best_score = mean_cv_score
                    best_degree = degree

            ddi.debug_print(f'Best Polynomial Degree: {best_degree} with Mean Squared Error: {best_score}', debug=Debug) 

            # Set the best degree for the final model
            reg_order = best_degree

    # Instantiate the model
    model = Dict_regression_models[reg_type](reg_order) if reg_order is not None else Dict_regression_models[reg_type]()

    # # Instantiate the model
    # model = Dict_regression_models[reg_type]()
    
    # Fit the model
    if weights is not None and reg_type == 'Polynomial Regression':
        model.fit(x_train, y_train, linearregression__sample_weight=weights_train)
    else:
        model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)    
    
    # Print valuable information depending on the regression type
    if reg_type == 'Polynomial Regression':
        # Access the named steps in the pipeline.
        poly = model.named_steps['polynomialfeatures']
        linear_reg = model.named_steps['linearregression']
        
        coefficients = linear_reg.coef_
        intercept = linear_reg.intercept_

        # Display the polynomial equation
        polynomial_equation = f"Polynomial Equation: y = {round(intercept[0], 8)}"
        for i in range(1, len(coefficients[0])):
            polynomial_equation += f" + {round(coefficients[0][i], 8)} * x^{i}"
        
        ddi.debug_print(polynomial_equation, debug=Debug) 

    else:
        # For other regression types
        coeffs_summary = f"Coefficients: {model.coef_}" if hasattr(model, "coef_") else "No coefficients to display."
        intercept_summary = f"Intercept: {model.intercept_}" if hasattr(model, "intercept_") else ""
        ddi.debug_print(f"Model Information:\n{coeffs_summary}\n{intercept_summary}", debug=Debug) 

    # Get all the errors associated to the model
    whole_errors_model(y_test, y_pred)

    # Make predictions to add at the dataset
    predictions = model.predict(x)
    data_for_plot['predicted_count'] = predictions   
    
    return data_for_plot


"""#=============================================================================
   #=============================================================================
   #============================================================================="""


def whole_errors_model(y, y_pred):

    """
    Goal: Evaluate the predictive performance of a regression model by calculating various error metrics.

    Parameters:
    - y: Target vector containing the actual values corresponding to predictions.
    - y_pred: Predicted values generated by the regression model.

    Returns:
    - None: This function prints various error metrics to the console.
    
    Error Metrics Calculated:
    - Mean Squared Error (MSE):
      - Average of the squares of the differences between predicted values (y_pred) and actual values (y). 
      - **Ideal Value**: Tends towards 0. A lower MSE indicates better model performance.

    - R-squared (R²):
      - Proportion of variance in the dependent variable that can be explained by the independent variables in the model.
      - **Ideal Value**: Tends towards 1. A higher R² indicates that the model explains a significant portion of the variability.

    - Mean Absolute Error (MAE):
      - Average of the absolute differences between predicted values (y_pred) and actual values (y).
      - **Ideal Value**: Tends towards 0. A lower MAE indicates more accurate predictions.

    - Root Mean Squared Error (RMSE):
      - Square root of the average of the squared differences between predicted values (y_pred) and actual values (y).
      - **Ideal Value**: Tends towards 0. A lower RMSE indicates better model performance.

    - Median Absolute Error (MedAE):
      - Median of the absolute differences between predicted values (y_pred) and actual values (y).
      - **Ideal Value**: Tends towards 0. A lower MedAE indicates a more accurate predictive model.
    """    
    
    Debug = aps.Debug
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)
    
    # R-squared (R²)
    r2 = r2_score(y, y_pred)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Median Absolute Error (MedAE)
    medae = median_absolute_error(y, y_pred)
    
    print("Errors of the model") 
    print(f'MSE: {mse}') 
    print(f'R^2 Score: {r2}') 
    print(f'MAE: {mae}') 
    print(f'RMSE: {rmse}') 
    print(f'MedAE: {medae}') 
    
    ddi.debug_print("", debug=Debug) 


"""#=============================================================================
   #=============================================================================
   #============================================================================="""

def make_model(type_model, df, target, ml_test_size,
                              ml_num_fea, ml_num_imp, ml_num_enc,
                              ml_ode_fea, ml_ode_imp, ml_ode_enc,
                              ml_ohe_fea, ml_ohe_imp, ml_ohe_enc,
                              ml_model):  
    
    """
    Goal: Make ML classification and add the predictive value in the dataset.

    Parameters:
    - type_model: Typ of the model (Regression/Classification)
    - df: Dataframe.
    - target: Target of the model.
    - ml_test_size: The ratio of testing value for the fit.

    - numerical_features: List of features that have numerical values.
    - ohe_categorical_features: List of features that have nominal values.
    - ode_categorical_features: List of features that have ordinal values.
        
    - num_imputer: Imputer for numerical features.
    - ohe_imputer: Imputer for nominal features.
    - ode_imputer: Imputer for ordinal features.

    - num_encoder: Encoder for numerical features.
    - ohe_encoder: Encoder for nominal features.
    - ode_encoder: Encoder for ordinal features.

    - ml_model: Model of classification for the data.
    

    Returns:
    - X
    - y
    - preprocessor
    - ml_model: The Machine learning model
    """    
    
    Debug = aps.Debug

    print()
    print("Make {type_model} model")
    print()
    
    # Filter for numerical/categorical features
    numerical_features = ml_num_fea  if ml_num_fea is not None else []
    ode_categorical_features = ml_ode_fea if ml_ode_fea is not None else []
    ohe_categorical_features = ml_ohe_fea if ml_ohe_fea is not None else []
    
    print("numerical_features:",numerical_features)
    print("ode_categorical_features:",ode_categorical_features)
    print("ohe_categorical_features:",ohe_categorical_features)   
    print("") 

    target = target

    ml_num_imp, ml_num_enc = eval(ml_num_imp), eval(ml_num_enc)
    ml_ode_imp, ml_ode_enc = eval(ml_ode_imp), eval(ml_ode_enc)
    ml_ohe_imp, ml_ohe_enc = eval(ml_ohe_imp), eval(ml_ohe_enc)
    ml_model = eval(ml_model)
    

    print(ml_num_imp, ml_num_enc)
    print(ml_ode_imp, ml_ode_enc)
    print(ml_ohe_imp, ml_ohe_enc)
    print(ml_model)
    print()
    
    numerical_pipeline = make_pipeline(ml_num_imp, ml_num_enc)
        
    ode_categorical_pipeline = make_pipeline(ml_ode_imp, ml_ode_enc)

    ohe_categorical_pipeline = make_pipeline(ml_ohe_imp, ml_ohe_enc)
    
    preprocessor = make_column_transformer((numerical_pipeline,numerical_features),
                                           (ohe_categorical_pipeline,ohe_categorical_features),
                                           (ode_categorical_pipeline,ode_categorical_features))
    
   
    X = df[numerical_features + ode_categorical_features + ohe_categorical_features]
    y = df[target]
    
    X_preprocessed = preprocessor.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=ml_test_size, random_state=42)
    
    
    ml_model.fit(X_train, y_train)
    
    y_pred = ml_model.predict(X_val)
    
    if type_model =="Classification":
        get_score_classification(y_val, y_pred)
    elif type_model =="Regression":
        # Get all the errors associated to the model
        whole_errors_model(y_test, y_pred)        
    
    # evaluation_model(ml_model, X_train, y_train)
    
    return X, y, preprocessor, ml_model


def get_score_classification(y_test, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # confusion = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    # print('Confusion Matrix:')
    # print(confusion)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

def evaluation_model(model, X_train, y_train):
    
    model.fit(X_train, y_train)
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1_macro',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.show()