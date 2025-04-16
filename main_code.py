import pandas as pd
import numpy as np
import random
import time  # Import time module to track response time
from google import genai

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the API client and API key
API_KEY = "AIzaSyBU2x31lb-W1omCb2-Efra9VEQH8i5QtWA"
client = genai.Client(api_key=API_KEY)

# Define the levels for each variable
genders = ["Male", "Female"]
races = ["White", "Black", "Asian", "Latino"]
adhd_severities = ["None", "Mild", "Moderate", "Severe"]
age_groups = ["Youth", "Teenager", "Adult", "Senior"]
models = ["gemini-2.0-flash"]  # Correct model names

# List of real prompts
prompts = [
    "Can you help me stay focused on my reading for the next 10 minutes?",
    "What is the best way to break down a complex task into smaller steps?",
    "Can you remind me about my meeting in 30 minutes?",
    "How should I organize my work schedule for the week?",
    "What should I do if I feel overwhelmed by all the tasks I need to do?",
    "Can you help me organize my to-do list for the day?",
    "How can I improve my focus when I'm feeling distracted?",
    "Can you remind me about my homework assignment?",
    "How do I stop procrastinating and start working on my tasks?",
    "Can you give me some tips for staying motivated to finish my project?",
    "What are some ways to manage my anxiety when I feel overwhelmed?",
    "Can you help me set reminders for my daily tasks?",
    "How do I avoid distractions when I'm studying?",
    "Can you suggest ways to stay focused during work?",
    "What should I do if I forget something important?",
    "Can you help me plan my day more effectively?",
    "How do I organize my notes for school?",
    "Can you give me a step-by-step plan for completing my homework?",
    "What are some tips to help me remember important dates or events?",
    "Can you give me a pep talk to get me motivated to study?",
    "I have a big presentation tomorrow, and I’m feeling so scatterbrained. Can you help me outline my key talking points and rehearse effectively?",
    "I’m overloaded with house chores, and I keep forgetting small tasks like taking out the trash or doing laundry. How can I organize these tasks so I don’t miss anything?",
    "I’m supposed to study for a science quiz tonight, but I can’t stop scrolling on social media. Can you help me create a distraction-free study routine?",
    "My boss emailed me about an important project deadline in two days, but I’m freaking out because I haven’t started. What’s a good strategy to catch up quickly?",
    "I’ve promised to meet friends this weekend, but I also have to finish my homework. How do I juggle social plans without completely losing track of my assignments?",
    "I need to prepare dinner for guests tonight, but I’m overwhelmed deciding what to cook and how to manage the timing. Can you help me plan step-by-step?",
    "I’ve been asked to join a new extracurricular activity at school, but I’m worried I won’t manage my existing responsibilities. How can I schedule everything without burning out?",
    "I’m moving next week and have so many boxes to pack, but I can’t seem to start. Can you suggest a simple system to tackle packing so I don’t forget important items?",
    "I have multiple bills to pay this month, and I’m worried I’ll miss due dates. Can you help me create a payment schedule and reminders?",
    "My teacher just assigned a group project, and I don’t want to let my teammates down. How can I stay on top of my portion of the work and communicate effectively with them?"
]

# Define grouped prompt categories
prompt_categories = {
    "Task Management": [
        "What is the best way to break down a complex task into smaller steps?",
        "How should I organize my work schedule for the week?",
        "Can you help me organize my to-do list for the day?",
        "Can you help me plan my day more effectively?",
        "Can you give me a step-by-step plan for completing my homework?",
        "I’m moving next week and have so many boxes to pack, but I can’t seem to start. Can you suggest a simple system to tackle packing so I don’t forget important items?",
        "I have multiple bills to pay this month, and I’m worried I’ll miss due dates. Can you help me create a payment schedule and reminders?",
    ],
    "Focus and Distraction": [
        "Can you help me stay focused on my reading for the next 10 minutes?",
        "How can I improve my focus when I'm feeling distracted?",
        "How do I avoid distractions when I'm studying?",
        "Can you suggest ways to stay focused during work?",
        "I’m supposed to study for a science quiz tonight, but I can’t stop scrolling on social media. Can you help me create a distraction-free study routine?",
    ],
    "Memory and Reminders": [
        "Can you remind me about my meeting in 30 minutes?",
        "Can you remind me about my homework assignment?",
        "What should I do if I forget something important?",
        "What are some tips to help me remember important dates or events?",
        "Can you help me set reminders for my daily tasks?",
    ],
    "Motivation and Emotion": [
        "What should I do if I feel overwhelmed by all the tasks I need to do?",
        "How do I stop procrastinating and start working on my tasks?",
        "Can you give me some tips for staying motivated to finish my project?",
        "Can you give me a pep talk to get me motivated to study?",
        "What are some ways to manage my anxiety when I feel overwhelmed?",
        "I have a big presentation tomorrow, and I’m feeling so scatterbrained. Can you help me outline my key talking points and rehearse effectively?",
        "My boss emailed me about an important project deadline in two days, but I’m freaking out because I haven’t started. What’s a good strategy to catch up quickly?",
        "I’ve been asked to join a new extracurricular activity at school, but I’m worried I won’t manage my existing responsibilities. How can I schedule everything without burning out?",
    ],
    "Planning and Balancing Life": [
        "I’ve promised to meet friends this weekend, but I also have to finish my homework. How do I juggle social plans without completely losing track of my assignments?",
        "I’m overloaded with house chores, and I keep forgetting small tasks like taking out the trash or doing laundry. How can I organize these tasks so I don’t miss anything?",
        "I need to prepare dinner for guests tonight, but I’m overwhelmed deciding what to cook and how to manage the timing. Can you help me plan step-by-step?",
        "My teacher just assigned a group project, and I don’t want to let my teammates down. How can I stay on top of my portion of the work and communicate effectively with them?"
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
                            
                            if response_time < 5:
                                print(f"Response time is {response_time:.2f}s, adding a 5-second delay.")
                                time.sleep(5)
                            
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