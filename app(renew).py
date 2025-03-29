import streamlit as st
import pickle
import docx  
import PyPDF2
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load saved models
clf = pickle.load(open('clf.pkl', 'rb'))  # Trained model for prediction
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # TF-IDF vectorizer
le = pickle.load(open('encoder.pkl', 'rb'))  # LabelEncoder

# Define job categories and keywords (expand as needed)
job_keywords = {
    # ✅ Existing Roles (Expanded)
    "Software Engineer": [
        "Python", "Java", "C++", "Software Development", "Algorithms", "Backend", "Frontend",
        "Microservices", "API Development", "Cloud Computing", "DevOps", "System Design"
    ],
    "Data Scientist": [
        "Machine Learning", "Deep Learning", "Data Analysis", "AI", "Python", "Statistics",
        "Big Data", "Data Mining", "TensorFlow", "PyTorch", "SQL", "Feature Engineering"
    ],
    "Business Analyst": [
        "Business Strategy", "Market Analysis", "Financial Modeling", "SQL", "Excel",
        "Data Visualization", "Power BI", "KPI Analysis", "Stakeholder Management", "Tableau"
    ],
    "Product Manager": [
        "Product Development", "Roadmap", "Stakeholder Management", "Agile", "Scrum",
        "User Research", "Customer Experience", "Product Strategy", "Wireframing", "MVP"
    ],
    "Marketing Specialist": [
        "SEO", "Digital Marketing", "Brand Management", "Social Media", "Google Ads",
        "Content Marketing", "Email Marketing", "Market Segmentation", "Lead Generation", "Analytics"
    ],

    # ✅ Newly Added Roles (25+)
    "AI/ML Engineer": [
        "Artificial Intelligence", "Neural Networks", "Deep Learning", "NLP", "Computer Vision",
        "PyTorch", "TensorFlow", "Reinforcement Learning", "Model Optimization", "Explainable AI"
    ],
    "Cloud Engineer": [
        "AWS", "Azure", "Google Cloud", "Kubernetes", "Docker",
        "CI/CD", "Terraform", "Cloud Security", "Load Balancing", "Serverless Computing"
    ],
    "DevOps Engineer": [
        "CI/CD", "Jenkins", "Docker", "Kubernetes", "Infrastructure as Code",
        "AWS", "Terraform", "GitOps", "Configuration Management", "Monitoring & Logging"
    ],
    "Full Stack Developer": [
        "React", "Node.js", "Angular", "Express.js", "MongoDB",
        "SQL", "JavaScript", "TypeScript", "REST API", "GraphQL"
    ],
    "Cybersecurity Analyst": [
        "Cybersecurity", "Penetration Testing", "Network Security", "Malware Analysis", "Risk Assessment",
        "Ethical Hacking", "Cryptography", "SOC", "Incident Response", "Security Policies"
    ],
    "Blockchain Developer": [
        "Smart Contracts", "Ethereum", "Solidity", "Hyperledger", "Cryptocurrency",
        "DApps", "NFTs", "DeFi", "Consensus Algorithms", "Web3"
    ],
    "Game Developer": [
        "Unity", "Unreal Engine", "C#", "Game Physics", "3D Rendering",
        "Game Design", "AI for Games", "Multiplayer Networking", "Shader Programming", "VR/AR"
    ],
    "Embedded Systems Engineer": [
        "Embedded C", "Microcontrollers", "RTOS", "Firmware Development", "IoT",
        "Serial Communication", "FPGA", "Circuit Design", "PCB Layout", "Edge AI"
    ],
    "Network Engineer": [
        "CCNA", "CCNP", "Routing", "Switching", "Firewalls",
        "Network Security", "Load Balancing", "VPN", "LAN/WAN", "Wireshark"
    ],
    "HR Specialist": [
        "Talent Acquisition", "Employee Engagement", "HR Analytics", "Onboarding",
        "Compliance", "Payroll", "Diversity & Inclusion", "Performance Management", "HR Policies"
    ],
    "Finance Analyst": [
        "Financial Modeling", "Budgeting", "Forecasting", "Excel", "Risk Management",
        "Investment Analysis", "Accounting", "CFA", "Market Trends", "Stock Analysis"
    ],
    "Legal Advisor": [
        "Corporate Law", "Contract Management", "Legal Compliance", "Intellectual Property",
        "GDPR", "Privacy Laws", "Employment Law", "Dispute Resolution", "Mergers & Acquisitions"
    ],
    "UI/UX Designer": [
        "Figma", "Sketch", "Adobe XD", "Wireframing", "Prototyping",
        "User Research", "Interaction Design", "Usability Testing", "A/B Testing", "Design Thinking"
    ],
    "Content Writer": [
        "SEO Writing", "Copywriting", "Technical Writing", "Blogging", "Content Strategy",
        "Social Media Writing", "Editing", "Proofreading", "Email Copywriting", "Storytelling"
    ],
    "Mechanical Engineer": [
        "CAD", "SolidWorks", "ANSYS", "Thermodynamics", "Robotics",
        "Manufacturing", "Automotive Engineering", "Finite Element Analysis", "Mechatronics", "HVAC"
    ],
    "Electrical Engineer": [
        "Circuit Design", "Power Systems", "Embedded Systems", "Renewable Energy",
        "Microcontrollers", "Power Electronics", "SCADA", "Motor Control", "PCB Design", "IoT"
    ],
    "Civil Engineer": [
        "Structural Engineering", "AutoCAD", "Revit", "Construction Management",
        "Geotechnical Engineering", "Urban Planning", "Surveying", "Sustainability", "Material Science"
    ],
    "Data Engineer": [
        "ETL", "Big Data", "Data Warehousing", "Spark", "Kafka",
        "SQL", "NoSQL", "Cloud Data Pipelines", "Database Optimization", "Data Lakes"
    ],
    "Robotics Engineer": [
        "ROS", "Mechatronics", "Computer Vision", "Actuators", "3D Motion Planning",
        "Control Systems", "Industrial Automation", "Simulations", "Kinematics", "LIDAR"
    ],
    "Healthcare Data Analyst": [
        "Healthcare Analytics", "EHR", "Medical Data", "Patient Outcomes",
        "Public Health", "Statistical Analysis", "Data Privacy", "HIPAA", "Clinical Research"
    ],
    "Pharmaceutical Researcher": [
        "Drug Discovery", "Clinical Trials", "Biomedical Research", "Regulatory Affairs",
        "Molecular Biology", "Biotechnology", "Pharmacokinetics", "Medical Devices", "Bioinformatics"
    ],
    "Supply Chain Manager": [
        "Logistics", "Inventory Management", "Procurement", "Vendor Management",
        "ERP", "Supply Chain Optimization", "Lean Management", "Warehouse Operations", "Demand Forecasting"
    ],
    "Operations Manager": [
        "Process Improvement", "Six Sigma", "Operations Strategy", "Risk Management",
        "Workflow Optimization", "KPI Monitoring", "Cost Reduction", "Lean Operations", "Automation"
    ],
    "Biotechnologist": [
        "Genetic Engineering", "CRISPR", "Bioinformatics", "Molecular Biology",
        "Microbiology", "Biomedical Research", "DNA Sequencing", "Biopharmaceuticals", "Stem Cells"
    ],
    "Aerospace Engineer": [
        "Aerodynamics", "Propulsion Systems", "Flight Dynamics", "Spacecraft Design",
        "Rocketry", "Aviation Regulations", "Satellite Communications", "Orbital Mechanics"
    ],
    "Quantum Computing Scientist": [
        "Quantum Mechanics", "Quantum Algorithms", "Superposition", "Qubits",
        "Cryogenic Systems", "Quantum Cryptography", "Quantum Gates", "Computational Physics"
    ],
    "Automotive Engineer": [
        "Vehicle Dynamics", "Powertrain", "Autonomous Vehicles", "Electric Vehicles",
        "ADAS", "Vehicle Safety", "NVH Analysis", "Aerodynamics", "Chassis Engineering"
    ]
}

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r"http\S+", " ", txt)
    cleanText = re.sub(r"RT|cc", " ", cleanText)
    cleanText = re.sub(r"#\S+", " ", cleanText)
    cleanText = re.sub(r"@\S+", "  ", cleanText)
    cleanText = re.sub(r"[^\w\s]", " ", cleanText)
    cleanText = re.sub(r"\s+", " ", cleanText)
    return cleanText.lower()

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
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Function to predict category of a resume
def predict_category(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = clf.predict(vectorized_text)
    predicted_category_name = le.inverse_transform([predicted_category[0]])[0]  # Convert label back to text
    return predicted_category_name

# Function to score a resume based on keyword relevance
def score_resume(text, selected_category):
    keywords = job_keywords.get(selected_category, [])
    text_words = cleanResume(text).split()
    
    # Count keyword occurrences
    score = sum(text_words.count(keyword.lower()) for keyword in keywords)
    
    return score

# Streamlit App
def main():
    st.set_page_config(page_title="Resume Screening & Ranking", page_icon="📄", layout="wide")
    
    st.title("📑 Resume Screening & Ranking System")
    st.markdown("Upload resumes in PDF, TXT, or DOCX format. Select a job category to rank resumes based on relevance.")

    # Dropdown for category selection
    selected_category = st.selectbox("Select the job category to rank resumes for:", list(job_keywords.keys()))

    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        resumes = []
        for uploaded_file in uploaded_files:
            try:
                resume_text = handle_file_upload(uploaded_file)
                predicted_category = predict_category(resume_text)
                relevance_score = score_resume(resume_text, selected_category)

                resumes.append({
                    "name": uploaded_file.name,
                    "predicted_category": predicted_category,
                    "score": relevance_score,
                    "text": resume_text
                })

                st.write(f"✅ Processed: *{uploaded_file.name}*")
                st.write(f"🔹 Predicted Category: *{predicted_category}*")
                st.write(f"📊 Relevance Score for *{selected_category}: *{relevance_score}**")

            except Exception as e:
                st.error(f"❌ Error processing *{uploaded_file.name}*: {str(e)}")

        # Rank resumes based on relevance to the selected job category
        if resumes:
            st.subheader("🏆 Ranked Resumes (Most Relevant to Least Relevant)")
            ranked_resumes = sorted(resumes, key=lambda x: x['score'], reverse=True)

            for idx, resume in enumerate(ranked_resumes, 1):
                st.write(f"{idx}. *{resume['name']}* - Predicted: {resume['predicted_category']} - Score: {resume['score']}")

if _name_ == "_main_":
    main()
