import pandas as pd
from textblob import TextBlob
import os

def analyze_sentiment_textblob(text):
    """
    Uses TextBlob to generate a polarity sentiment score in [-1, 1].
    Positive scores indicate positive sentiment, negative scores indicate negative sentiment.
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Polarity score from -1 (negative) to 1 (positive)
    return sentiment_score

def map_satisfaction(sentiment_score):
    """
    Map sentiment_score to satisfaction_score label.
    Custom thresholds: >0.1 => 'High', -0.1 <= score <= 0.1 => 'Neutral', < -0.1 => 'Low'
    """
    if sentiment_score > 0.15:
        return "High"         # Sentiment is strongly positive
    elif sentiment_score > -0.15:
        return "Neutral"      # Sentiment is neutral or mixed
    else:
        return "Low"          # Sentiment is strongly negative

def perform_sentiment_analysis(input_file, output_file):
    """
    Loads the CSV file with interaction data, performs sentiment analysis,
    and saves the updated data with sentiment and satisfaction scores to a new CSV file.
    """
    # Load the CSV file with the response data
    df = pd.read_csv(input_file)

    # List to store processed results with sentiment and satisfaction scores
    processed_results = []

    for _, item in df.iterrows():
        response_text = item['response_text']
        sentiment_score = analyze_sentiment_textblob(response_text)
        satisfaction_score = map_satisfaction(sentiment_score)
        
        response_length = item['response_length']  # Response length
        response_time = item['response_time_seconds']  # Response time in seconds

        # Add the relevant data (sentiment score, satisfaction score, etc.)
        processed_results.append({
            'persona_id': item['persona_id'],
            'promptgroup_id': item['promptgroup_id'],
            'gender': item['gender'],
            'race': item['race'],
            'adhd_severity': item['adhd_severity'],
            'age_group': item['age_group'],
            'prompt_id': item['prompt_id'],
            'prompt_text': item['prompt_text'],
            'model_name': item['model_name'],
            'response_length': response_length,
            'response_time_seconds': response_time,
            'sentiment_score': sentiment_score,
            'satisfaction_score': satisfaction_score
        })

    # Convert the processed results into a DataFrame
    sentiment_df = pd.DataFrame(processed_results)

    # Save the results to a new CSV file
    sentiment_df.to_csv(output_file, index=False)
    print(f"Sentiment analysis complete. Updated results saved to {output_file}")

def main():
    # Provide the path to the input CSV file (with interaction data)
    input_file = "balanced_responses_efficient.csv"  # Example input CSV file
    output_file = "real_responses_with_sentiment.csv"  # Output CSV file

    perform_sentiment_analysis(input_file, output_file)

if __name__ == "__main__":
    main()
