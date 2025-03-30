#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 16:48:50 2025

@author: quentin
"""


import pandas as pd
import numpy as np
import logging
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dash import html
from termcolor import colored
from dash import dcc, html, Input, Output, dash_table, callback, callback_context


def clean_data(df, independent, dependent):
    """Cleans data by removing rows with NaN values for the independent and dependent."""
    return df[[independent, dependent]].dropna()

def check_outliers_z_score(df, independent):
    """Detects outliers in the specified independent using the Z-score method."""
    z_scores = np.abs(stats.zscore(df[independent]))
    return np.where(z_scores > 3)

def check_outliers_iqr(df, independent):
    """Detects outliers in the specified independent using the IQR method."""
    Q1 = df[independent].quantile(0.25)
    Q3 = df[independent].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[independent] < lower_bound) | (df[independent] > upper_bound)]
    return outliers, lower_bound, upper_bound

def check_outliers(df, independent):
    """Detects outliers in the specified independent using the IQR method."""
    outliers, lower_bound, upper_bound = check_outliers_iqr(df, independent)
    
    print(outliers)
    
    if not outliers.empty:

        # Create a DataFrame from outliers
        outliers_df = pd.DataFrame(outliers)

        # Transpose the DataFrame
        transposed_outliers_df = outliers_df.T
                
        table = dash_table.DataTable(
            data=transposed_outliers_df.reset_index().to_dict('records'),
            # columns=[{'name': col, 'id': col} for col in transposed_outliers_df.columns],
            fixed_rows={'headers': True},
            style_table={
                # 'className': 'table-container',  # Apply table container style
                'overflowX': 'auto',
                'paddingLeft': '2px',
                'paddingRight': '20px',
                'marginLeft': '8px'
            },
            style_header={
                # 'className': 'table-header',    # Apply header style
                'backgroundColor': '#343a40',
                'color': 'white',
                'whiteSpace': 'normal',
                'textAlign': 'center',
                'height': 'auto',
            },
            style_cell={
                # 'className': 'table-cell',       # Apply cell style
                'backgroundColor': '#1e1e1e',
                'color': '#f8f9fa',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'normal',
                'textAlign': 'center',
                'height': 'auto',
            },
            style_data={
                'whiteSpace': 'nowrap',
                'textAlign': 'center',
            },
        )

        return (
            f"⚠️ Outliers detected in {independent} using the IQR method (see table below): "
            f"lower_bound = {lower_bound:.4e}  |  upper_bound = {upper_bound:.4e}", table
        )
    else:
        return f"✅ No outliers detected in {independent}.",  None



def plot_normality(groups, independent, p_values):
    """Creates QQ plots and histograms for each group for normality visualization using Plotly, with a dark theme."""

    num_groups = len(groups)
    
    # Create subplot titles including p-values
    subplot_titles = [
        f'QQ Plot for {independent} (Group {i + 1})<br><span style="color: #FFD700;">p-value: {p_values[i]:.4e}</span>' 
        for i in range(num_groups)
    ]
    
    # Create subplots
    fig = make_subplots(rows=2, cols=num_groups,
                        subplot_titles=subplot_titles + 
                                       [f'Histogram & Density for {independent} (Group {i + 1})' for i in range(num_groups)],
                        specs=[[{'type': 'scatter'} for _ in range(num_groups)],
                               [{'type': 'histogram'} for _ in range(num_groups)]])

    for i, (group, p_value) in enumerate(zip(groups, p_values)):
        # Calculate QQ plot data
        # Calculate QQ plot data
        sorted_stats = np.sort(group)  # Sort the group data
        theoretical_quantiles = stats.norm.ppf(np.linspace(0, 1, len(sorted_stats)))
        
        
        # Add QQ plot
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_stats, mode='markers', name='QQ Plot',
                                 marker=dict(color='lightblue')),
                      row=1, col=i + 1)
        
        # Add y=x reference line
        # min_x = min(theoretical_quantiles[~np.isnan(theoretical_quantiles) & ~np.isinf(theoretical_quantiles)])
        # max_x = max(theoretical_quantiles[~np.isnan(theoretical_quantiles) & ~np.isinf(theoretical_quantiles)])
        # min_y = min(sorted_stats)
        # max_y = max(sorted_stats)
        # fig.add_trace(go.Scatter(x=[min_x, max_x], y=[min_y, max_y], mode='lines', name='y=x',
        #                          line=dict(color='red', dash='dash')),
        #               row=1, col=i + 1)

        # Add histogram & density
        hist_data = []
        for val in group:
            hist_data.append(val)

        # Plot histogram with density
        fig.add_trace(go.Histogram(x=hist_data, histnorm='probability density', name='Histogram',
                                    marker=dict(color='lightblue'), opacity=0.75),
                      row=2, col=i + 1)

        # Add density line
        density_x = np.linspace(min(group), max(group), 100)
        density_y = (1 / (group.std() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((density_x - group.mean()) / group.std())**2)
        
        fig.add_trace(go.Scatter(x=density_x, y=density_y, mode='lines', name='Density',
                                 line=dict(color='darkblue')),
                      row=2, col=i + 1)

    # Update layout
    fig.update_layout(title_text='Normality Visualization', template="plotly_dark", height=800)
    
    # Update x and y axes labels
    
    fig.update_yaxes(title_text='Sample Quantiles', row=1, col=1)
    for i in range(num_groups):
        fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=i + 1)
    fig.update_yaxes(title_text='Density', row=2, col=1)
    for i in range(num_groups):
        fig.update_xaxes(title_text='Sample Quantiles', row=2, col=i + 1)
        


    # Hide legend for all traces
    fig.for_each_trace(lambda t: t.update(showlegend=False))
    
    fig.update_layout(
        plot_bgcolor='#1e1e1e',  # Darker background for the plot area
        paper_bgcolor='#101820',  # Dark gray for the paper
        font=dict(color='white'),  # White text color
        # title = figname,
        # title_font=dict(size=20, color='white')
        )
    
    return fig


def plot_normality_seaborn(groups, independent, p_values):
    """Creates QQ plots and histograms for each group for normality visualization, saving the figure as an image with a dark theme."""
    
    # Set Seaborn's style to dark
    sns.set_style("darkgrid", {"axes.facecolor": "#2E2E2E", "figure.facecolor": "#2E2E2E"})
    
    num_groups = len(groups)
    
    fig, axes = plt.subplots(nrows=2, ncols=num_groups, figsize=(14, 7))
    
    for i, (group, p_value) in enumerate(zip(groups, p_values)):
        # QQ Plot
        stats.probplot(group, dist="norm", plot=axes[0][i])
        axes[0][i].set_title(f'QQ Plot for {independent} (Group {i + 1})', color='white')
        axes[0][i].text(0.2, 0.90, f'p-value: {p_value:.4e}', fontsize=12, 
                        ha='center', va='center', color='white')  # Moved closer to the top

        # Histogram with density plot
        sns.histplot(group, kde=True, ax=axes[1][i], stat='density', color="lightblue")
        axes[1][i].set_title(f'Histogram & Density for {independent} (Group {i + 1})', color='white')

        # Set axis labels color
        axes[0][i].tick_params(axis='x', colors='white')
        axes[0][i].tick_params(axis='y', colors='white')
        axes[1][i].tick_params(axis='x', colors='white')
        axes[1][i].tick_params(axis='y', colors='white')

        # Change label colors
        axes[0][i].set_xlabel('Theoretical Quantiles', color='white')
        axes[0][i].set_ylabel('Sample Quantiles', color='white')
        axes[1][i].set_xlabel(independent, color='white')
        axes[1][i].set_ylabel('Density', color='white')

    # Adjust spacing: reduce space between top and bottom plots
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical space between rows

    plt.tight_layout(pad=2.0)  # General tight layout adjustments
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    
    # Encode it as base64 for Dash
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the figure after saving
    return f"data:image/png;base64,{image_base64}"



def test_normality(df_cleaned, independent, dependent):
    """Tests for normality using the Shapiro-Wilk test on the numeric dependent split by the categorical independent."""
    groups = [group[dependent].values for name, group in df_cleaned.groupby(independent)]
    groups = [g for g in groups if len(g) >= 3]
    
    # if len(groups) < 2:
    #     return "❌ Not enough groups for normality test."

    p_values = []
    for group in groups:
        _, p_value = stats.shapiro(group)
        p_values.append(p_value)

    fig = plot_normality(groups, independent, p_values)

    overall_p_value = min(p_values)
    if overall_p_value > 0.05:
        return "✅ Normality assumption is satisfied (see graphs below).", fig
    else:
        return f"❌ Normality assumption is not satisfied (see graphs below), overall_p_value = {overall_p_value:.4e} < 0.05", fig

def test_variance_homogeneity(df_cleaned, independent, dependent):
    """Tests for homogeneity of variances using Levene’s test."""
    groups = [df_cleaned[df_cleaned[independent] == cat][dependent].values for cat in df_cleaned[independent].unique()]
    
    stat, p_value = stats.levene(*groups)
    
    if p_value > 0.05:
        return "✅ Homogeneity of variances assumption is satisfied."
    else:
        return f"❌ Homogeneity of variances assumption is not satisfied, p-value = {p_value:.4e} < 0.05"

def check_class_balance(df_cleaned, independent, dependent):
    """Checks balance between classes in the dependent variable."""
    class_counts = df_cleaned[independent].value_counts()
    min_count = class_counts.min()
    total_count = class_counts.sum()
    
    balance_ratio = min_count / total_count

    if balance_ratio < 0.1:
        return f"❌ Class balance is not acceptable, balance_ratio = {balance_ratio:.4e} < 0.1"
    return "✅ Class balance is acceptable."

def check_expected_frequencies(contingency_table):
    """Check if expected frequencies are sufficient for Chi-squared test."""
    expected = contingency_table.values.sum() * contingency_table.divide(contingency_table.sum(axis=1), axis=0).fillna(0)
    
    if np.any(expected < 5):
        return "❌ Expected frequencies are not adequate for χ² test."
    else:
        return "✅ Expected frequencies are adequate for χ² test."

def perform_t_test(df_cleaned, independent, dependent):
    """Performs independent t-test."""
    # Check if the independent has exactly two unique categories
    if df_cleaned[independent].nunique() != 2:
        return "🚫 T-test requires exactly two groups for comparison."
    
    # Split the data into two groups based on the independent
    group1 = df_cleaned[df_cleaned[independent] == df_cleaned[independent].unique()[0]][dependent].values
    group2 = df_cleaned[df_cleaned[independent] == df_cleaned[independent].unique()[1]][dependent].values
    
    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)  # Change equal_var=False for Welch's t-test

    if p_value < 0.05:
        return f"⚖️ T-test Result: t-statistic = {t_stat:.4e}, p-value = {p_value:.4e}. ✅ Significance found between groups."
    else:
        return f"⚖️ T-test Result: t-statistic = {t_stat:.4e}, p-value = {p_value:.4e}. ✅ No significant difference found between groups."

def perform_anova(df_cleaned, independent, dependent):
    """Performs ANOVA test."""
    groups = [df_cleaned[df_cleaned[independent] == cat][dependent].values for cat in df_cleaned[independent].unique()]
    
    f_stat, p_value = stats.f_oneway(*groups)

    if p_value < 0.05:
        return f"⚖️ ANOVA Test Result: F-statistic = {f_stat:.4e}, p-value = {p_value:.4e}. ✅ Significance found between groups."
    else:
        return f"⚖️ ANOVA Test Result: F-statistic = {f_stat:.4e}, p-value = {p_value:.4e}. 🚫 No significant difference found between groups."

def perform_chi_squared_test(df_cleaned, independent, dependent):
    """Perform Chi-squared test based on the cleaned DataFrame."""
    contingency_table = pd.crosstab(df_cleaned[independent], df_cleaned[dependent])
    
    expected_freq_message = check_expected_frequencies(contingency_table)
    
    if "❌" in expected_freq_message:
        return expected_freq_message
    
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    if p_value < 0.05:
        return f"⚖️ χ² Test Result: χ² statistic = {chi2_stat:.4f}, p-value = {p_value:.4e}. ✅ Significance found."
    else:
        return f"⚖️ χ² Test Result: χ² statistic = {chi2_stat:.4f}, p-value = {p_value:.4e}. 🚫 No significant difference found."

def perform_mann_whitney(df, numerical_independent, ordinal_independent):
    """Performs the Mann-Whitney U Test between a numerical independent and a binary ordinal independent."""
    groups = [df[df[ordinal_independent] == category][numerical_independent].dropna() for category in df[ordinal_independent].unique()]

    if len(groups) != 2:
        return "❌ Mann-Whitney U Test requires exactly two groups."
    
    stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')

    if p_value < 0.05:
        return f"⚖️ Mann-Whitney U Test: ✅ Significant difference found between the groups (p-value = {p_value:.4e})."
    else:
        return f"⚖️ Mann-Whitney U Test: 🚫 No significant difference found between the groups (p-value = {p_value:.4e})."

def perform_kruskal_wallis(df_cleaned, independent, dependent):
    """Performs Kruskal-Wallis test."""
    groups = [df_cleaned[df_cleaned[independent] == cat][dependent].values for cat in df_cleaned[independent].unique()]    
    stat, p_value = stats.kruskal(*groups)

    if len(groups) < 2:
        return "❌ Not enough groups to perform Kruskal-Wallis test."

    if p_value < 0.05:
        return f"⚖️ Kruskal-Wallis test: ✅ Significant differences found between groups (p-value = {p_value:.4e})."
    else:
        return f"⚖️ Kruskal-Wallis test: 🚫 No significant differences found (p-value = {p_value:.4e})."

def perform_welch_anova(df_cleaned, independent, dependent):
    """Performs Welch's ANOVA when homogeneity of variances is not satisfied."""
    groups = [df_cleaned[df_cleaned[independent] == cat][dependent].dropna().values for cat in df_cleaned[independent].unique() if len(df_cleaned[df_cleaned[independent] == cat]) > 0]
    
    if len(groups) < 2:
        return "❌ Not enough groups to perform Welch's ANOVA."

    f_stat, p_value = stats.f_oneway(*groups)

    if p_value < 0.05:
        return f"⚖️ Welch's ANOVA Test Result: ✅ Significance found between groups (p-value = {p_value:.4e})."
    else:
        return f"⚖️ Welch's ANOVA Test Result: 🚫 No significant difference found between groups (p-value = {p_value:.4e})."

def bootstrap_test(df_cleaned, independent, dependent, n_iterations=1000):
    """Perform bootstrapping to compare group means."""
    groups = [df_cleaned[df_cleaned[independent] == cat][dependent].values for cat in df_cleaned[independent].unique()]
    means = np.array([np.mean(group) for group in groups])
    observed_diff = means.max() - means.min()

    bootstrap_diffs = []
    
    for _ in range(n_iterations):
        bootstrapped_samples = [np.random.choice(group, size=len(group), replace=True) for group in groups]
        boot_means = np.array([np.mean(sample) for sample in bootstrapped_samples])
        bootstrap_diffs.append(boot_means.max() - boot_means.min())
    
    p_value = np.mean(np.array(bootstrap_diffs) >= observed_diff)
    return f"🔄 Bootstrapping Test Result: Observed difference = {observed_diff}, p-value = {p_value:.4e}"

def Hypothesis_Testing_Methods(df, dependent, independent, dependent_type, independent_type):
    """Main function to make an Hypothesis Testing Methods (t-test / ANOVA / χ²)."""
    
    # Step 1: Clean the data
    df_cleaned = clean_data(df, independent, dependent)

    # Count number of unique groups
    num_groups = df_cleaned[independent].nunique()
    
    messages = []  # To collect messages
    
    if dependent_type == "Numerical" and num_groups==2:
        Hypothesis_test = "t-test"
    elif dependent_type == "Numerical" and num_groups>2:
        Hypothesis_test = "ANOVA"
    elif (dependent_type == "Ordinal" or dependent_type == "Nominal") and (independent_type == "Ordinal" or independent_type == "Nominal"):
        Hypothesis_test = "χ²"
    elif (dependent_type == "Ordinal" or dependent_type == "Nominal") and (independent_type == "Numerical" ):
        messages.append(f"⚠️ The Dependent and Independent variable must be interchange!")
        messages.append(f" Cannot have {independent_type} as Independent varaible when the Dependent variable is {dependent_type}.")
        Hypothesis_test = None
        return Hypothesis_test, messages, None, None
    else:
        messages.append(f"❌ Not enough groups in {independent} to get the analysis.")
        Hypothesis_test = None
        return Hypothesis_test, messages, None, None
        
    
    messages.append(f"\n🔍 Checking conditions for {Hypothesis_test} between {dependent} and {independent}")

    if dependent_type != "Nominal":
        # Check for outliers in the dependent
        outlier_check, outlier_table = check_outliers(df_cleaned, dependent)
        messages.append(outlier_check)  # Collect outlier check message
    else:
        outlier_table = None
    
    print("Hypothesis_test : ", Hypothesis_test)
    
    if Hypothesis_test == "t-test" or Hypothesis_test == "ANOVA":

        # Step 2: Check assumptions
        normality_message, normality_fig = test_normality(df_cleaned, independent, dependent)
        messages.append(normality_message)
        if "❌" in normality_message:
            if Hypothesis_test == "t-test":
                messages.append(get_explanation_on_Hypothesis_test('Mann-Whitney U Test'))
                mann_whitney_message = perform_mann_whitney(df_cleaned, dependent, independent)
                messages.append(mann_whitney_message)
            else:
                messages.append(get_explanation_on_Hypothesis_test('Kruskal-Wallis test'))
                kruskal_message = perform_kruskal_wallis(df_cleaned, independent, dependent)
                messages.append(kruskal_message)
            return Hypothesis_test, messages, normality_fig, outlier_table
    
        variance_message = test_variance_homogeneity(df_cleaned, independent, dependent)
        messages.append(variance_message)
        if "❌" in variance_message:       
            messages.append(get_explanation_on_Hypothesis_test('Welch\'s ANOVA'))
            welch_message = perform_welch_anova(df_cleaned, independent, dependent)
            messages.append(welch_message)
            return Hypothesis_test, messages, normality_fig, outlier_table
    
        class_balance_message = check_class_balance(df_cleaned, independent, dependent)
        messages.append(class_balance_message)
        if "❌" in class_balance_message:
            bootstrap_message = bootstrap_test(df_cleaned, independent, dependent)
            messages.append(bootstrap_message)
            return Hypothesis_test, messages, normality_fig, outlier_table
        
        # Step 3: Perform test
        if Hypothesis_test == "t-test":
            t_test_message = perform_t_test(df_cleaned, independent, dependent)
            messages.append(t_test_message)        
        if Hypothesis_test == "t-test":
            anova_message = perform_anova(df_cleaned, independent, dependent)
            messages.append(anova_message)

    elif Hypothesis_test == "χ²":
        # Step 2: Check expectations and perform Chi-squared test
        chi_squared_message = perform_chi_squared_test(df_cleaned, independent, dependent)
        messages.append(chi_squared_message)
        normality_fig = None
        
    else:
        messages.append(f"❌ No Hypothesis test possible {dependent_type} / {independent_type}.")
        return Hypothesis_test, messages, None, outlier_table
    

    return Hypothesis_test, messages, normality_fig, outlier_table

def get_explanation_on_Hypothesis_test(hypothesis_test):
    explanations = {
        't-test': (
            "A **t-test** determines if there is a significant difference between the means of two groups.\n"
            "### Equation:\n"
            "$$t = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_p \\sqrt{\\frac{1}{n_1} + \\frac{1}{n_2}}}$$\n"
            "where:\n"
            "- \\($$\\bar{x}_1$$, $$\\bar{x}_2$$\\) are the sample means\n"
            "- \\($$s_p = \\sqrt{\\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$\\) is the pooled standard deviation\n"
            "- \\($$n_1$$, $$n_2$$\\) are the sample sizes."
        ),
        'ANOVA': (
            "ANOVA tests for differences among three or more group means.\n"
            "### Equation:\n"
            "$$F = \\frac{MS_{between}}{MS_{within}}$$\n"
            "where:\n"
            "- \\($$MS_{between}=\\frac{\\sum_{i=1}^{k} n_{i} (\\bar{x}_{i} - \\bar{x})^{2}}{k - 1}$$\\) is the mean square between groups\n"
            "- \\($$MS_{within}=\\frac{\\sum_{i=1}^{k} \\sum_{j=1}^{n_{i}} (x_{ij} - \\bar{x}_{i})^{2}}{N - k}$$\\) is the mean square within groups."
        ),
        'χ²': (
            "The chi-squared test assesses how likely it is that an observed distribution is due to chance.\n"
            "### Equation:\n"
            "$$\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}$$\n"
            "where:\n"
            "- \\($$O_i$$\\) is the observed frequency\n"
            "- \\($$E_i$$\\) is the expected frequency."
        ),
        'Welch\'s ANOVA': (
            "Welch's ANOVA is a variation of ANOVA that is used when the assumption of equal variances is not met.\n"
            "### Equation:\n"
            "$$F = \\frac{S_{between}^2}{S_{within}^2}$$\n"
            "where:\n"
            "- \\($$S_{between}^2$$\\) is the weighted mean square between groups\n"
            "- \\($$S_{within}^2$$\\) is the weighted mean square within groups.\n"
            "It uses separate estimates of variance for each group, leading to a more robust test."
        ),
        'Kruskal-Wallis test': (
            "The Kruskal-Wallis test is a non-parametric method for comparing three or more independent groups.\n"
            "### Equation:\n"
            "$$H = \\frac{12}{N(N + 1)} \\sum_{i=1}^{k} \\frac{n_i(R_i^2)}{n_i - 1}$$\n"
            "where:\n"
            "- \\($$N$$\\) is the total number of observations\n"
            "- \\($$n_i$$\\) is the number of observations in group \\(i\\)\n"
            "- \\($$R_i$$\\) is the sum of ranks for group \\(i\\).\n"
            "It assesses whether the median ranks differ across groups."
        ),
        'Mann-Whitney U Test': (
            "The Mann-Whitney U Test is a non-parametric test for assessing whether two independent samples come from the same distribution.\n"
            "### Equation:\n"
            "$$U = R_1 - \\frac{n_1(n_1 + 1)}{2}$$\n"
            "where:\n"
            "- \\($$R_1$$\\) is the rank sum of the first group\n"
            "- \\($$n_1$$\\) is the number of observations in the first group.\n"
            "It determines if one of the two groups tends to have higher values than the other."
        ),
    }
    return explanations.get(hypothesis_test, "No explanation available for this test.")






