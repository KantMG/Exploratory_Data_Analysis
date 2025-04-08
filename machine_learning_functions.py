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

def make_model(type_model, df, target, target_type, ml_test_size,
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
    - target_type: Nature of the target variable ("numerical", "ordinal", "nominal").
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
    print(f"Make {type_model} model")
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
    # Preprocess target variable
    if target_type == "ordinal":
        # If ordinal, apply custom mapping or encoding
        y = df[target].map({value: index for index, value in enumerate(sorted(df[target].unique()))})
    elif target_type == "nominal" or target_type == "numerical":
        y = df[target]
    
    X_preprocessed = preprocessor.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=ml_test_size, random_state=42)
    
    print(X_train, y_train)
    
    ml_model.fit(X_train, y_train)
    
    y_pred = ml_model.predict(X_val)
    
    if type_model =="Classification":
        get_score_classification(y_val, y_pred)
    elif type_model =="Regression":
        # Get all the errors associated to the model
        whole_errors_model(y_val, y_pred)        
    
    # evaluation_model(ml_model, X_train, y_train)
    
    return X, y, preprocessor, ml_model



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