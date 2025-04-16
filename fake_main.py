import pandas as pd
import numpy as np
import random
import time
from google import genai

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the API client and API key
API_KEY = "AIzaSyDm59hz2MQJDibagOKzHixot_6ghIdqcSo" 
client = genai.Client(api_key=API_KEY)

# Define the levels for each variable
genders = ["Male", "Female"]
races = ["White", "Black"]
adhd_severities = ["None", "Mild"]
age_groups = ["Youth", "Adult"]
models = ["gemini-2.0-flash"]

# Define simplified prompts by category
prompt_categories = {
    "Task Management": [
        "What is the best way to break down a complex task into smaller steps?",
        "How do I stop procrastinating and start working on my tasks?"
    ],
    "Focus and Distraction": [
        "Can you help me stay focused on my reading for the next 10 minutes?"
    ],
    "Memory and Reminders": [
        "Can you remind me about my homework assignment?"
    ],
    "Motivation and Emotion": [
        "Can you give me some tips for staying motivated to finish my project?"
    ]
}

# Flatten prompts and create mappings
all_prompts = []
prompt_to_group = {}
for group_name, prompts in prompt_categories.items():
    for prompt in prompts:
        prompt_id = f"T{len(all_prompts)+1:02d}"
        all_prompts.append((prompt_id, prompt))
        prompt_to_group[prompt] = group_name

# Generate data with equal distribution
data = []
counter = {
    'gender': {g: 0 for g in genders},
    'race': {r: 0 for r in races},
    'adhd': {a: 0 for a in adhd_severities},
    'age': {a: 0 for a in age_groups}
}

# Calculate how many samples we need per attribute value
total_samples = len(genders) * len(races) * len(adhd_severities) * len(age_groups) * len(all_prompts)
samples_per_value = {
    'gender': total_samples // len(genders),
    'race': total_samples // len(races),
    'adhd': total_samples // len(adhd_severities),
    'age': total_samples // len(age_groups)
}

# Create structured iterations to ensure balance
for prompt_id, prompt_text in all_prompts:
    for gender in genders:
        for race in races:
            for severity in adhd_severities:
                for age_group in age_groups:
                    model = models[0]  # Using only one model
                    
                    print(f"Processing: {gender}/{race}/{severity}/{age_group} - Prompt: {prompt_id}")
                    
                    start_time = time.time()
                    
                    try:
                        # Call the Gemini API
                        response = client.models.generate_content(
                            model=model,
                            contents=[prompt_text]
                        )
                        
                        if response and hasattr(response, 'text'):
                            response_text = response.text
                            
                            # Calculate metrics
                            response_time = time.time() - start_time
                            response_length = len(response_text.split())
                            
                            # Update counters
                            counter['gender'][gender] += 1
                            counter['race'][race] += 1
                            counter['adhd'][severity] += 1
                            counter['age'][age_group] += 1
                            
                            # Create persona ID
                            persona_id = f"{gender[0]}{race[0]}{severity[0]}{age_group[0]}"
                            
                            # Add to dataset
                            data.append({
                                "persona_id": persona_id,
                                "promptgroup_id": prompt_to_group[prompt_text],
                                "gender": gender,
                                "race": race,
                                "adhd_severity": severity,
                                "age_group": age_group,
                                "prompt_id": prompt_id,
                                "prompt_text": prompt_text,
                                "response_text": response_text,
                                "model_name": model,
                                "response_length": response_length,
                                "response_time_seconds": round(response_time, 2)
                            })
                            
                            print(f"Response received: {response_length} words")
                        else:
                            print(f"Error: No valid response for {prompt_id}")
                            
                    except Exception as e:
                        print(f"API Error: {str(e)}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save the complete dataset
df.to_csv('balanced_responses_efficient.csv', index=False)

# Print distribution summary
print("\n=== Distribution Summary ===")
print("Gender distribution:")
print(df['gender'].value_counts())

print("\nRace distribution:")
print(df['race'].value_counts())

print("\nADHD Severity distribution:")
print(df['adhd_severity'].value_counts())

print("\nAge Group distribution:")
print(df['age_group'].value_counts())

print("\nPrompt Group distribution:")
print(df['promptgroup_id'].value_counts())

print("\nCSV file created successfully: 'balanced_responses_efficient.csv'")