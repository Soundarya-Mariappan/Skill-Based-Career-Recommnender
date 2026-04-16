# ==============================
# SKILL-BASED CAREER RECOMMENDER
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# 1️⃣ DATA COLLECTION
# ==============================

print("Loading Dataset...")

try:
    df = pd.read_csv("job_skill_set.csv")
except:
    print("CSV not found! Using sample dataset instead.\n")
    
    # Sample dataset (fallback)
    data = {
        "career": ["Data Scientist", "Web Developer", "AI Engineer", "Data Analyst"],
        "skills": [
            "Python, Machine Learning, SQL, Statistics",
            "HTML, CSS, JavaScript, React",
            "Python, Deep Learning, TensorFlow",
            "Excel, SQL, Data Visualization, Python"
        ]
    }
    df = pd.DataFrame(data)


# ==============================
# 2️⃣ DATA PREPROCESSING
# ==============================

print("\nChecking for null values...\n")
print(df.isnull().sum())

# Ensure correct column names
if "career" not in df.columns or "skills" not in df.columns:
    print("\nFixing column names...")
    df.columns = ["career", "skills"]

# Convert to string
df["skills"] = df["skills"].astype(str)

# Remove duplicates & nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text.strip()

# Apply cleaning
df["skills"] = df["skills"].apply(clean_text)

# Remove empty rows after cleaning
df = df[df["skills"] != ""]

print("\nRemaining data:", len(df))


# ==============================
# 3️⃣ FEATURE ENGINEERING
# ==============================

print("\nConverting Skills to Numerical Features...")

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

tfidf_matrix = vectorizer.fit_transform(df["skills"])


# ==============================
# 4️⃣ RECOMMENDATION FUNCTION
# ==============================

def recommend_career(user_input, top_n=5):
    
    if not user_input.strip():
        print("Please enter valid skills!")
        return []
    
    user_input = clean_text(user_input)
    user_vector = vectorizer.transform([user_input])
    
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    
    recommendations = []
    
    for idx in top_indices:
        recommendations.append({
            "Career": df.iloc[idx]["career"],
            "Match Score": round(similarity_scores[0][idx] * 100, 2)
        })
    
    return recommendations


# ==============================
# 5️⃣ GRAPH FUNCTION
# ==============================

def plot_recommendations(recommendations):
    
    if not recommendations:
        return
    
    careers = [rec['Career'] for rec in recommendations]
    scores = [rec['Match Score'] for rec in recommendations]
    
    plt.figure(figsize=(8,5))
    bars = plt.barh(careers, scores)
    
    plt.xlabel('Match Score (%)')
    plt.title('Top Career Recommendations')
    plt.xlim(0, 100)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                 f'{width}%', va='center')
    
    plt.gca().invert_yaxis()
    plt.show()


# ==============================
# 6️⃣ USER INTERACTION
# ==============================

print("\n===== SKILL BASED CAREER RECOMMENDER =====")

while True:
    
    user_skills = input("\nEnter your skills (comma separated) or type 'exit': ")
    
    if user_skills.lower() == "exit":
        print("Thank you for using Career Recommender!")
        break
    
    results = recommend_career(user_skills)
    
    if results:
        print("\nTop Career Recommendations:\n")
        
        for i, rec in enumerate(results, 1):
            print(f"{i}. {rec['Career']} ---> Match: {rec['Match Score']}%")
        
        # Show graph
        plot_recommendations(results)
