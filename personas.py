# personas.py

personas = {
    # Baseline Persona
    "Persona01": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Mild forgetfulness"]
    },
    # Change 1: Gender
    "Persona02": {
        "age": 25,
        "gender": "female",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Mild forgetfulness"]
    },
    # Change 2: Age
    "Persona03": {
        "age": 16,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "High school student",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Mild forgetfulness"]
    },
    # Change 3: Race
    "Persona04": {
        "age": 25,
        "gender": "male",
        "race": "Black",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Mild forgetfulness"]
    },
    # Change 4: ADHD severity to mild
    "Persona05": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "mild",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Difficulty maintaining attention"]
    },
    # Change 5: ADHD severity to moderate
    "Persona06": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "moderate",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Frequent distractions"],
        "needs": ["More frequent reminders", "Organization"],
        "symptoms": ["Forgetfulness", "Impulsivity"]
    },
    # Change 6: ADHD severity to severe
    "Persona07": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "severe",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Severe distractibility", "Missed deadlines"],
        "needs": ["High support", "Task breakdown"],
        "symptoms": ["Very short attention span"]
    },
    # Change 7: Add comorbidity anxiety
    "Persona08": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": ["anxiety"],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Worrying about tasks"],
        "needs": ["Stress coping strategies", "Organization"],
        "symptoms": ["Occasional panic feeling"]
    },
    # Change 8: Add comorbidity depression
    "Persona09": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": ["depression"],
        "profession": "Office worker",
        "challenges": ["Low motivation", "Frequent tiredness"],
        "needs": ["Encouragement", "Task accountability"],
        "symptoms": ["Mood swings", "Sluggishness"]
    },
    # Change 9: Profession to teacher
    "Persona10": {
        "age": 25,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Teacher",
        "challenges": ["Procrastination", "Staying focused"],
        "needs": ["Reminders", "Organization"],
        "symptoms": ["Mild forgetfulness"]
    },
    # Change 10: Profession + age shift
    "Persona11": {
        "age": 65,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Retired",
        "challenges": ["Keeping track of tasks", "Procrastination"],
        "needs": ["Reminders", "Simple instructions"],
        "symptoms": ["Occasional confusion"]
    },
    # Change 11: Gender + mild ADHD
    "Persona12": {
        "age": 25,
        "gender": "female",
        "race": "White",
        "adhd_severity": "mild",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Disorganization"],
        "needs": ["Reminders", "Visual task planning"],
        "symptoms": ["Sometimes forgets deadlines"]
    },
    # Change 12: Race + moderate ADHD
    "Persona13": {
        "age": 25,
        "gender": "male",
        "race": "Asian",
        "adhd_severity": "moderate",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Frequent distractions", "Missed deadlines"],
        "needs": ["Detailed task breakdown", "Timed reminders"],
        "symptoms": ["Struggles to maintain focus"]
    },
    # Change 13: Race + severe ADHD
    "Persona14": {
        "age": 25,
        "gender": "male",
        "race": "Black",
        "adhd_severity": "severe",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Extreme distractibility", "Constant restlessness"],
        "needs": ["High structure", "Step-by-step coaching"],
        "symptoms": ["Very short attention span"]
    },
    # Change 14: Age + no ADHD + memory issues
    "Persona15": {
        "age": 40,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": ["memory issues"],
        "profession": "Office worker",
        "challenges": ["Forgetting tasks", "Losing track of appointments"],
        "needs": ["Frequent reminders", "Calendar management"],
        "symptoms": ["Frequent short-term memory lapses"]
    },
    # Change 15: Female + moderate ADHD
    "Persona16": {
        "age": 30,
        "gender": "female",
        "race": "White",
        "adhd_severity": "moderate",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Task switching", "Overwhelmed by details"],
        "needs": ["Simple instructions", "Frequent check-ins"],
        "symptoms": ["Difficulty focusing on complex tasks"]
    },
    # Change 16: Latino + mild ADHD
    "Persona17": {
        "age": 25,
        "gender": "male",
        "race": "Latino",
        "adhd_severity": "mild",
        "comorbidities": [],
        "profession": "Office worker",
        "challenges": ["Procrastination", "Organizational issues"],
        "needs": ["Reminders", "Clear structure"],
        "symptoms": ["Sometimes drifts off tasks"]
    },
    # Change 17: Latino + none ADHD + anxiety
    "Persona18": {
        "age": 25,
        "gender": "male",
        "race": "Latino",
        "adhd_severity": "none",
        "comorbidities": ["anxiety"],
        "profession": "Office worker",
        "challenges": ["Nervous about tasks", "Procrastination"],
        "needs": ["Calming strategies", "Reminders"],
        "symptoms": ["Occasional worry"]
    },
    # Change 18: Black + none ADHD + depression
    "Persona19": {
        "age": 25,
        "gender": "male",
        "race": "Black",
        "adhd_severity": "none",
        "comorbidities": ["depression"],
        "profession": "Office worker",
        "challenges": ["Low motivation", "Lack of focus"],
        "needs": ["Positive reinforcement", "Simple goals"],
        "symptoms": ["Mood dips"]
    },
    # Change 19: Female + age shift + mild ADHD
    "Persona20": {
        "age": 18,
        "gender": "female",
        "race": "White",
        "adhd_severity": "mild",
        "comorbidities": [],
        "profession": "High school student",
        "challenges": ["Finishing homework", "Staying organized"],
        "needs": ["Task breakdown", "Gentle reminders"],
        "symptoms": ["Occasional daydreaming"]
    },
    # Change 20: Age shift + moderate ADHD + teacher
    "Persona21": {
        "age": 35,
        "gender": "male",
        "race": "White",
        "adhd_severity": "moderate",
        "comorbidities": [],
        "profession": "Teacher",
        "challenges": ["Classroom prep", "Distractibility"],
        "needs": ["Clear time blocks", "Frequent refocus cues"],
        "symptoms": ["Interrupted workflow"]
    },
    # Change 21: Age shift + severe ADHD + marketing
    "Persona22": {
        "age": 28,
        "gender": "female",
        "race": "White",
        "adhd_severity": "severe",
        "comorbidities": [],
        "profession": "Marketing specialist",
        "challenges": ["Meeting deadlines", "Overthinking"],
        "needs": ["Tight structure", "Task segmentation"],
        "symptoms": ["Constant restlessness"]
    },
    # Change 22: No ADHD + retired
    "Persona23": {
        "age": 70,
        "gender": "male",
        "race": "White",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "Retired",
        "challenges": ["General forgetfulness", "Staying active"],
        "needs": ["Daily routine tips", "Reminders"],
        "symptoms": ["Occasional confusion"]
    },
    # Change 23: Latino + mild ADHD + teacher
    "Persona24": {
        "age": 40,
        "gender": "female",
        "race": "Latino",
        "adhd_severity": "mild",
        "comorbidities": [],
        "profession": "Teacher",
        "challenges": ["Class management", "Paperwork deadlines"],
        "needs": ["Structured schedule", "Task checklists"],
        "symptoms": ["Difficulty organizing materials"]
    },
    # Change 24: Latino + moderate ADHD + memory issues
    "Persona25": {
        "age": 40,
        "gender": "male",
        "race": "Latino",
        "adhd_severity": "moderate",
        "comorbidities": ["memory issues"],
        "profession": "Office worker",
        "challenges": ["Forgetting tasks", "Shifting focus too often"],
        "needs": ["Repetitive reminders", "Daily planning"],
        "symptoms": ["Frequent disorientation"]
    },
    # Change 25: Black + none ADHD + college student
    "Persona26": {
        "age": 20,
        "gender": "female",
        "race": "Black",
        "adhd_severity": "none",
        "comorbidities": [],
        "profession": "College student",
        "challenges": ["Procrastination", "Balancing classes"],
        "needs": ["Time management tips", "Study plan"],
        "symptoms": ["Occasional overwhelm"]
    },
    # Change 26: Asian + severe ADHD + marketing
    "Persona27": {
        "age": 29,
        "gender": "female",
        "race": "Asian",
        "adhd_severity": "severe",
        "comorbidities": [],
        "profession": "Marketing specialist",
        "challenges": ["Handling multiple campaigns", "Easily distracted"],
        "needs": ["High-level organization", "Frequent check-ins"],
        "symptoms": ["Difficulty focusing on details"]
    },
    # Change 27: White + mild ADHD + depression
    "Persona28": {
        "age": 26,
        "gender": "male",
        "race": "White",
        "adhd_severity": "mild",
        "comorbidities": ["depression"],
        "profession": "Graphic designer",
        "challenges": ["Creative blocks", "Low motivation"],
        "needs": ["Motivational check-ins", "Project structuring"],
        "symptoms": ["Energy spikes and drops"]
    },
    # Change 28: Black + moderate ADHD + teacher
    "Persona29": {
        "age": 34,
        "gender": "male",
        "race": "Black",
        "adhd_severity": "moderate",
        "comorbidities": [],
        "profession": "Teacher",
        "challenges": ["Grading backlog", "Disorganized class notes"],
        "needs": ["Regular to-do lists", "Time-blocking"],
        "symptoms": ["Mid-task distraction"]
    },
    # Change 29: Asian + none ADHD + depression
    "Persona30": {
        "age": 22,
        "gender": "female",
        "race": "Asian",
        "adhd_severity": "none",
        "comorbidities": ["depression"],
        "profession": "College student",
        "challenges": ["Feeling overwhelmed", "Procrastination"],
        "needs": ["Emotional support", "Clear deadlines"],
        "symptoms": ["Frequent low moods"]
    }
}
