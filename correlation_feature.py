#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 16:48:50 2025

@author: quentin
"""


import numpy as np

import pandas as pd

from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel, RFE, RFECV

from scipy.stats import chi2_contingency, f_oneway, zscore



def correlation_target(df, tar):

    numerical_features = make_column_selector(dtype_include=np.number)(df)
    categorical_features = make_column_selector(dtype_exclude=np.number)(df)

    # Calculate the correlation between tar and other numerical features
    correlation_with_target = df[numerical_features].corr()[tar]
    
    high_correlation_features = correlation_with_target[correlation_with_target > 0.5]
    
    # Display the features with high correlation
    print()
    print("High correlation features")
    print(high_correlation_features)
    
    
    # Study of the varaince
    selector = VarianceThreshold(threshold = 0.2)
    selector.fit_transform(df[numerical_features])
    print(selector.get_support())
    
    # Get the boolean mask of selected features
    support_mask = selector.get_support()
    
    # Get the list of feature names corresponding to False in the support mask
    features_to_remove = [feature for feature, is_supported in zip(numerical_features, support_mask) if not is_supported]
    
    # Display the features with low variance
    print(features_to_remove)
    print()
    
    
    # Assuming high_correlation_features is a pandas Series and categorical_features is a defined list
    high_correlation_numerical_features = [feature for feature in high_correlation_features.index 
                                           if feature in numerical_features and feature != tar]
    
    # Display the features with high correlation that are also categorical
    print("High correlation numerical features:", high_correlation_numerical_features)




    # Initialize results list
    anova_results = {}
    
    # Loop through each categorical feature
    for column in categorical_features:
        groups = [df[df[column] == category][tar] for category in df[column].unique()]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Store results
        anova_results[column] = {'f_stat': f_stat, 'p_value': p_value}
    
    # Convert results to a DataFrame for easier analysis
    anova_df = pd.DataFrame(anova_results).T
    
    # Sort by p-value
    top_results = anova_df.sort_values(by='p_value')
    
    # Get the top 10 most significant features
    # for i in top_results:
    #     print(top_results[])
    top_50_features = top_results.head(50)
    
    # Display the top 5 features
    print(top_50_features)
    
    
    
    # Initialize list for features to drop
    drop_categorical_features = []
    
    # Calculate the maximum F-statistic
    max_f_stat = top_results['f_stat'].max()
    
    print("max_f_stat=",max_f_stat)
    
    # Loop through each feature's results
    for column, metrics in top_results.iterrows():
        if (metrics['p_value'] > 0.05 or metrics['f_stat'] < max_f_stat / 10 or 
            pd.isna(metrics['f_stat']) or pd.isna(metrics['p_value'])):
            print("metrics['f_stat']=",metrics['f_stat'], "limit max=", max_f_stat / 10)
            drop_categorical_features.append(column)
    
    # Output features considered for dropping
    print("Features to consider dropping:", drop_categorical_features)
    
    remaining_categorical_features = [feature for feature in top_results.index if feature not in drop_categorical_features]
    
    # Output the remaining features
    print("Remaining features after dropping:", remaining_categorical_features)