import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from docx import Document
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import Counter

# ----------------- NLTK Setup -----------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------- Load Model Artifacts -----------------
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ----------------- Skills List -----------------
TECH_SKILLS = [
    "python", "sql", "machine learning", "data analysis", "tensorflow",
    "keras", "pytorch", "streamlit", "sklearn", "pandas", "numpy", "seaborn",
    "matplotlib", "deep learning", "nlp", "cloud", "azure", "aws", "gcp"
]

# ----------------- Text Cleaning -----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# ----------------- Extract Text -----------------
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_resume(file):
    if file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""

# ----------------- Skills Extraction -----------------
def extract_skills(text):
    return [s for s in TECH_SKILLS if s in text.lower()]

# ----------------- Experience Extraction -----------------
def extract_experience(text):
    years = re.findall(r'(\d+)\s*(?:\+?\s*years?|yrs?|year)', text.lower())
    return max(map(int, years), default=0)

# ----------------- Send Email -----------------
def send_email(to_email, candidate_name, predicted_role):
    sender_email = "your_email@gmail.com"   # Replace with Gmail
    password = "your_app_password"          # Gmail App Password

    subject = f"Interview Invitation for {predicted_role}"
    body = f"""
    Dear {candidate_name},

    We reviewed your resume and found your skills suitable for the {predicted_role} role.
    We would like to invite you for an interview.

    Regards,
    HR Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True
    except Exception:
        return False

# ----------------- Streamlit UI -----------------
st.title("Resume Classifier with Visual Insights (DOCX Only)")
st.markdown("Upload DOCX resumes to classify roles, visualize skills & experience, and download best candidates list.")

uploaded_files = st.file_uploader("Upload DOCX resumes", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    results = []
    all_skills = []
    role_counts = Counter()

    for file in uploaded_files:
        # Extract & clean
        text = extract_text_from_resume(file)
        cleaned = clean_text(text)

        # Predict role
        pred_idx = model.predict(tfidf.transform([cleaned]))[0]
        predicted_role = le.inverse_transform([pred_idx])[0]

        # Extract skills & experience
        skills = extract_skills(text)
        exp_years = extract_experience(text)

        # Append aggregated data
        results.append({
            "Candidate": file.name,
            "Skills": ", ".join(skills),
            "Experience (Years)": exp_years,
            "Predicted Role": predicted_role
        })

        all_skills.extend(skills)
        role_counts[predicted_role] += 1

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # ----------------- Visualization: Skills -----------------
    st.subheader("Skills Frequency (All Resumes)")
    skill_freq = Counter(all_skills)
    if skill_freq:
        skill_df = pd.DataFrame(skill_freq.items(), columns=["Skill", "Count"]).sort_values(by="Count", ascending=False)
        st.bar_chart(skill_df.set_index("Skill"))
    else:
        st.info("No technical skills detected in uploaded resumes.")

    # ----------------- Visualization: Experience -----------------
    st.subheader("Experience Distribution (Years)")
    st.bar_chart(df_results.set_index("Candidate")["Experience (Years)"])

    # ----------------- Visualization: Predicted Roles -----------------
    st.subheader("Predicted Roles Distribution")
    role_df = pd.DataFrame(role_counts.items(), columns=["Role", "Count"]).sort_values(by="Count", ascending=False)
    st.bar_chart(role_df.set_index("Role"))

    # ----------------- Best Candidates (Sorted by Experience) -----------------
    st.subheader("Best Candidates (Highest Experience)")
    best_candidates = df_results.sort_values(by="Experience (Years)", ascending=False)
    st.dataframe(best_candidates)

    # Download CSV
    st.download_button(
        "Download Best Candidates CSV",
        best_candidates.to_csv(index=False),
        "best_candidates.csv",
        "text/csv"
    )
