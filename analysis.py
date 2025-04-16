import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sentiment results from the output of sentiment analysis
def load_results(results_file):
    """Loads the processed sentiment analysis data into a DataFrame."""
    df = pd.read_csv(results_file)  # Using read_csv since the file is now saved as CSV in the main code
    return df

# Demographic Parity Analysis: Check if response length, response time, and sentiment scores differ significantly across different demographic characteristics
def demographic_parity_analysis(df):
    """Analyze biases across demographic groups for response length, response time, and sentiment score."""
    demographic_columns = ['gender', 'race', 'adhd_severity', 'age_group']
    
    # Perform group-by for demographics
    demographic_groups = df.groupby(demographic_columns).agg({
        'sentiment_score': ['mean', 'std'],
        'response_length': ['mean', 'std'],
        'response_time_seconds': ['mean', 'std']
    }).reset_index()

    print("Demographic Parity Analysis:\n", demographic_groups)

    # Perform ANOVA to check for significant differences across groups for each feature
    for feature in ['sentiment_score', 'response_length', 'response_time_seconds']:
        anova_result = stats.f_oneway(*[group[1][feature] for group in df.groupby(demographic_columns)] )
        print(f"ANOVA result for {feature} -> F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")
        if anova_result.pvalue < 0.05:
            print(f"Significant differences found for {feature}.")
        else:
            print(f"No significant differences for {feature}.")

# Analyze the intersection of demographic factors (e.g., age × race × ADHD severity)
def intersecting_demographic_factors_analysis(df):
    """Analyze the impact of intersecting demographic factors on response length, response time, and sentiment score."""
    # The 'age_group' is already categorized into: "Youth", "Teenager", "Adult", "Senior"
    # No need for further binning or conversion, it's already ready to use.
    
    # Check if 'age_group' is correctly populated
    print(f"Unique Age Groups: {df['age_group'].unique()}")

    # Create intersection of factors (age_group × race × ADHD severity)
    intersected_groups = df.groupby(['age_group', 'race', 'adhd_severity']).agg({
        'sentiment_score': ['mean', 'std'],
        'response_length': ['mean', 'std'],
        'response_time_seconds': ['mean', 'std']
    }).reset_index()

    print("Intersecting Factors Analysis:\n", intersected_groups)

    # Visualize the intersecting factors with a heatmap for sentiment score, response length, and response time
    sentiment_heatmap = df.pivot_table(index='race', columns='age_group', values='sentiment_score', aggfunc='mean')
    response_length_heatmap = df.pivot_table(index='race', columns='age_group', values='response_length', aggfunc='mean')
    response_time_heatmap = df.pivot_table(index='race', columns='age_group', values='response_time_seconds', aggfunc='mean')

    # Plot heatmaps
    plt.figure(figsize=(12, 6))
    sns.heatmap(sentiment_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Sentiment Score by Age Group and Race")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(response_length_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Response Length by Age Group and Race")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(response_time_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Response Time by Age Group and Race")
    plt.show()

# Main Function to run all analyses
def main():
    results_file = "real_responses_with_sentiment.csv"  # Example file name from the main code
    df = load_results(results_file)

    # Perform Demographic Parity Analysis to check if sentiment differs across demographic characteristics
    demographic_parity_analysis(df)

    # Perform Intersecting Demographic Factors Analysis to analyze the combined impact of age, race, and ADHD severity
    intersecting_demographic_factors_analysis(df)

if __name__ == "__main__":
    main()
