import json
import pandas as pd
import numpy as np
import scipy.stats as stats

def load_results(results_file):
    """
    Loads the JSON results produced by sentiment_analysis.py and returns a pandas DataFrame.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def demographic_parity_analysis(df, demographic_col='persona'):
    """
    Calculate the rate of satisfactory responses for each demographic group.
    This checks if different groups have the same probability of receiving favorable outcomes.
    """
    # Define favorable outcomes (High satisfaction score)
    df['favorable'] = df['satisfaction_score'].apply(lambda x: 1 if x == 'High' else 0)

    # Calculate the rate of favorable outcomes for each group
    favorable_rate = df.groupby(demographic_col)['favorable'].mean().reset_index()
    favorable_rate.columns = [demographic_col, 'favorable_rate']

    print("\nDemographic Parity Analysis:")
    print(favorable_rate)
    return favorable_rate

def disparate_impact_analysis(favorable_rate, threshold=0.8):
    """
    Perform the 4/5ths rule (Disparate Impact) to check if any group has
    a disproportionately low favorable rate. Flag groups with less than
    80% of the best group's favorable rate.
    """
    # Find the maximum favorable rate (best group)
    max_rate = favorable_rate['favorable_rate'].max()

    # Calculate impact ratios for each group
    favorable_rate['impact_ratio'] = favorable_rate['favorable_rate'] / max_rate

    # Flag any group with impact ratio less than the threshold (4/5ths rule)
    favorable_rate['disparate_impact'] = favorable_rate['impact_ratio'] < threshold

    print("\nDisparate Impact Analysis (4/5ths rule):")
    print(favorable_rate)
    return favorable_rate

def intersectional_analysis(df):
    """
    Perform intersectional analysis of multiple demographic attributes (e.g., race × gender × ADHD severity).
    Analyzes how these factors interact and affect the outcomes.
    """
    print("\nIntersectional Analysis:")
    intersectional_data = df.groupby(['persona', 'satisfaction_score']).size().unstack(fill_value=0)
    print(intersectional_data)
    return intersectional_data

def significance_testing(df):
    """
    Perform statistical significance testing (e.g., ANOVA, chi-square) to determine if the observed disparities
    in satisfaction scores across demographic groups are statistically significant.
    """
    # Example: ANOVA for satisfaction score across different personas
    persona_groups = df.groupby('persona')['satisfaction_score'].apply(list)
    
    # Perform one-way ANOVA test
    f_statistic, p_value = stats.f_oneway(*persona_groups)
    
    print("\nANOVA Results for Satisfaction Score across Personas:")
    print(f"F-statistic: {f_statistic}, P-value: {p_value}")
    return f_statistic, p_value

def analyze_bias_by_demographics(results_file):
    """
    Perform demographic parity and disparate impact analysis,
    and significance testing on the results.
    """
    # Load the results (JSON or CSV)
    df = load_results(results_file)

    # Demographic Parity Analysis: Calculate the favorable rates for each demographic group
    favorable_rate = demographic_parity_analysis(df, demographic_col='persona')

    # Disparate Impact Analysis: Check for disparate impact using the 4/5ths rule
    disparate_impact_analysis(favorable_rate, threshold=0.8)

    # Intersectional Analysis: Examine how race, gender, ADHD severity interact
    intersectional_analysis(df)

    # Significance Testing: Check if observed disparities in satisfaction are statistically significant
    significance_testing(df)

def main():
    # Provide the path to the results file (with sentiment analysis already included)
    results_file = "test_results_2025-04-06_15-07-08_with_sentiment.csv"  # Update with the actual file name
    analyze_bias_by_demographics(results_file)

if __name__ == "__main__":
    main()
