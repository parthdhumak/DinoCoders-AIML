import streamlit as st
import pickle
import docx  
import PyPDF2
import re
import nltk
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load saved models
clf = pickle.load(open('clf.pkl', 'rb'))  # Model for prediction
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # TF-IDF vectorizer
le = pickle.load(open('encoder.pkl', 'rb'))  # LabelEncoder

# Mock job categories for ranking (adjust as needed)
job_rankings = {
    "Software Engineer": 1,
    "Data Scientist": 2,
    "Business Analyst": 3,
    "Product Manager": 4,
    "Marketing Specialist": 5
}

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = clf.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    predicted_category = clf.predict(vectorized_text)  # Ensure encoder is a trained ML model, not LabelEncoder
    predicted_category_name = le.inverse_transform([predicted_category[0]])  # Convert encoded label back to text


# Function to rank resumes based on predefined job rankings
def rank_resumes(resumes):
    ranked_resumes = []
    for resume in resumes:
        category = resume['category']
        score = job_rankings.get(category, float('inf'))  # Lower score = higher priority
        ranked_resumes.append((resume, score))
    
    # Sort by score (ascending)
    ranked_resumes.sort(key=lambda x: x[1])
    return [resume for resume, _ in ranked_resumes]

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction & Ranking", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction & Ranking App")
    st.markdown("Upload multiple resumes in PDF, TXT, or DOCX format to get the predicted job category and ranking.")

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        resumes = []
        for uploaded_file in uploaded_files:
            try:
                resume_text = handle_file_upload(uploaded_file)
                category = pred(resume_text)

                resumes.append({
                    "name": uploaded_file.name,
                    "category": category,
                    "text": resume_text
                })

                st.write(f"✅ Successfully processed: **{uploaded_file.name}**")
                st.write(f"Predicted Category: **{category}**")

            except Exception as e:
                st.error(f"❌ Error processing **{uploaded_file.name}**: {str(e)}")

        # Ranking resumes based on predefined job priorities
        if resumes:
            st.subheader("🏆 Ranked Resumes (Best to Worst)")
            ranked_resumes = rank_resumes(resumes)

            for idx, resume in enumerate(ranked_resumes, 1):
                st.write(f"{idx}. **{resume['name']}** - Category: {resume['category']}")

if __name__ == "__main__":
    main()
