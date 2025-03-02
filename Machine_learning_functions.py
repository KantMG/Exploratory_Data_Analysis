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

from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm, tree, neighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import cross_val_score

from termcolor import colored


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

    print()
    print("Make ML "+reg_type)
    print()
    
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

                print(f'Degree {degree} - CV Mean Squared Error: {mean_cv_score}')

                # Check if this is the best degree found
                if mean_cv_score < best_score:
                    best_score = mean_cv_score
                    best_degree = degree

            print(f'Best Polynomial Degree: {best_degree} with Mean Squared Error: {best_score}')

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
        
        print(polynomial_equation)

    else:
        # For other regression types
        coeffs_summary = f"Coefficients: {model.coef_}" if hasattr(model, "coef_") else "No coefficients to display."
        intercept_summary = f"Intercept: {model.intercept_}" if hasattr(model, "intercept_") else ""
        print(f"Model Information:\n{coeffs_summary}\n{intercept_summary}")

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
    
    print()
