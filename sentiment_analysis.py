import json
import pandas as pd
import nltk
from textblob import TextBlob
import os

# Ensure the VADER lexicon is downloaded (for TextBlob)
# nltk.download('vader_lexicon')

def load_results(results_file):
    """
    Loads the JSON results produced by main.py and returns a list of dict.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def analyze_sentiment_textblob(text):
    """
    Uses TextBlob to generate a polarity sentiment score in [-1, 1].
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


def perform_sentiment_analysis(results_file):
    """
    Loads the JSON from main.py, performs sentiment analysis, maps to satisfaction,
    and saves a new JSON with sentiment fields.
    """
    results = load_results(results_file)
    
    # List to store processed results with sentiment and satisfaction scores
    processed_results = []
    
    for item in results:
        response_text = item['robot_response']
        sentiment_score = analyze_sentiment_textblob(response_text)
        satisfaction_score = map_satisfaction(sentiment_score)

        # Get response time from the main.py results
        response_time = item.get('response_time', 0)  # Default to 0 if not available

        # Add only the relevant data (sentiment score, satisfaction score, and response time)
        processed_results.append({
            'persona': item['persona'],
            'task': item['task'],
            'sentiment_score': sentiment_score,
            'satisfaction_score': satisfaction_score,
            'response_time': response_time
        })

    # Specify output file paths (JSON and CSV)
    output_json = results_file.replace(".json", "_with_sentiment.json")
    output_csv = results_file.replace(".json", "_with_sentiment.csv")

    # Delete the previous CSV file if it exists
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Save the results to a new JSON file
    with open(output_json, 'w') as f:
        json.dump(processed_results, f, indent=4)
    
    print(f"Sentiment analysis complete. Updated results saved to {output_json}")

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(processed_results)
    df.to_csv(output_csv, index=False)
    print(f"Also saved CSV to {output_csv}")

def main():
    # Provide your JSON file from main.py
    results_file = "test_results_2025-04-06_15-07-08.json"  # Example file name from main.py
    perform_sentiment_analysis(results_file)

if __name__ == "__main__":
    main()

# import json
# import pandas as pd
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import os

# # Ensure the VADER lexicon is downloaded
# # nltk.download('vader_lexicon')

# def load_results(results_file):
#     """
#     Loads the JSON results produced by main.py and returns a list of dict.
#     """
#     with open(results_file, 'r') as f:
#         data = json.load(f)
#     return data

# def analyze_sentiment_vader(text):
#     """
#     Uses NLTK's VADER to generate a compound sentiment score in [-1, 1].
#     """
#     sid = SentimentIntensityAnalyzer()
#     scores = sid.polarity_scores(text)
#     return scores['compound']

# def map_satisfaction(sentiment_score):
#     """
#     Map sentiment_score to satisfaction_score label.
#     Example thresholds: >0.1 => 'High', < -0.1 => 'Low', else 'Neutral'
#     """
#     if sentiment_score > 0.1:
#         return "High"
#     elif sentiment_score < -0.1:
#         return "Low"
#     else:
#         return "Neutral"

# def perform_sentiment_analysis(results_file):
#     """
#     Loads the JSON from main.py, performs sentiment analysis, maps to satisfaction,
#     and saves a new JSON with sentiment fields.
#     """
#     results = load_results(results_file)
    
#     # List to store processed results with sentiment and satisfaction scores
#     processed_results = []
    
#     for item in results:
#         response_text = item['robot_response']
#         sentiment_score = analyze_sentiment_vader(response_text)
#         satisfaction_score = map_satisfaction(sentiment_score)

#         # Add only the relevant data (sentiment score and satisfaction score)
#         processed_results.append({
#             'persona': item['persona'],
#             'task': item['task'],
#             'sentiment_score': sentiment_score,
#             'satisfaction_score': satisfaction_score
#         })

#     # Specify output file paths (JSON and CSV)
#     output_json = results_file.replace(".json", "_with_sentiment.json")
#     output_csv = results_file.replace(".json", "_with_sentiment.csv")

#     # Delete the previous CSV file if it exists
#     if os.path.exists(output_csv):
#         os.remove(output_csv)

#     # Save the results to a new JSON file
#     with open(output_json, 'w') as f:
#         json.dump(processed_results, f, indent=4)
    
#     print(f"Sentiment analysis complete. Updated results saved to {output_json}")

#     # Convert to DataFrame and save to CSV
#     df = pd.DataFrame(processed_results)
#     df.to_csv(output_csv, index=False)
#     print(f"Also saved CSV to {output_csv}")

# def main():
#     # Provide your JSON file from main.py
#     results_file = "test_results_2025-04-06_15-07-08.json"  # Example file name from main.py
#     perform_sentiment_analysis(results_file)

# if __name__ == "__main__":
#     main()
