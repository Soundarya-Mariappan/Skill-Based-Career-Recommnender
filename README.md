# Skill-Based Career Recommender System

## Overview
The Skill-Based Career Recommender System is a Machine Learning project that suggests suitable career paths based on user-entered skills. It uses Natural Language Processing (NLP) techniques to match user skills with job roles and provides the most relevant career recommendations.

---

## Features
- Accepts user input as skills (comma-separated)
- Recommends top matching career options
- Uses TF-IDF and Cosine Similarity for matching
- Performs data preprocessing and text cleaning
- Handles missing and empty data
- Visualizes results using graphs using Matplotlib

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## Concepts Used
- Machine Learning (Basic)
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Data Preprocessing

---

## Project Structure
├── job_skill_set.csv # Dataset (career & skills)
├── career_recommender.py # Main Python script
├── README.md # Project documentation

---

## How It Works
1. Load dataset containing career roles and required skills  
2. Clean and preprocess the skill data  
3. Convert text data into numerical form using TF-IDF  
4. Take user input (skills)  
5. Compute similarity using cosine similarity  
6. Display top matching career recommendations  

---

## How to Run

1. Install required libraries:
   pip install pandas numpy scikit-learn matplotlib

2. Run the Python script:
3. python career_recommender.py


3. Enter your skills when prompted:

Python, SQL, Machine Learning


---

## Sample Output

Top Career Recommendations:

Data Scientist ---> Match: 43.99%
AI Engineer ---> Match: 18.55%
Web Developer ---> Match: 0.0%

---

## Visualization
The system displays a bar chart showing match scores for recommended careers.

---

## Notes
- Ensure the dataset contains valid skill descriptions in text format
- Avoid empty or null values in the skills column
- If dataset is missing, sample data will be used

---

## Future Improvements
- Add web interface using Flask or Streamlit  
- Improve recommendation accuracy using advanced NLP models  
- Add more detailed career insights such as salary and growth  
- Deploy as a web application  

---

## Author
Your Name

---

## Acknowledgment
This project was developed as part of learning Machine Learning and Data Analysis concepts.
