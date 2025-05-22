from flask import Flask, render_template, request, redirect
import os
import fitz  # PyMuPDF
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
UPLOAD_FOLDER = 'resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def get_resume_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        jd_text = request.form['jd']
        jd_clean = clean_text(jd_text)

        uploaded_files = request.files.getlist('resumes')
        results = []

        for file in uploaded_files:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            resume_text = get_resume_text(filepath)
            if not resume_text.strip():
                continue

            resume_clean = clean_text(resume_text)
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform([jd_clean, resume_clean])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            score_pct = round(score * 100, 2)

            if score_pct >= 75:
                verdict = "Excellent Fit"
                suggestion = "Great resume! You are well aligned with the job requirements."
            elif score_pct >= 40:
                verdict = "Potential Fit"
                suggestion = "You're on the right track. Tailor your resume more closely to the job description."
            else:
                verdict = "Unfit"
                suggestion = "Consider updating your skills and keywords to better match the job role."

            results.append({
                'Resume': filename,
                'Score (%)': score_pct,
                'Verdict': verdict,
                'Suggestion': suggestion
            })



        results = sorted(results, key=lambda x: x['Score (%)'], reverse=True)
        df = pd.DataFrame(results)
        df.to_excel("results.xlsx", index=False)

        return render_template('results.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
