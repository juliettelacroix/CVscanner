import json
import re
import pandas as pd
import streamlit as st
import pdfplumber
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    model="mistralai/mistral-7b-instruct",
)

# Prompt template
template = """
You are an AI HR assistant. Given a job description and a CV, do the following:
1. Extract the candidate's full name, email address, phone number and location from the CV text.
2. Score how well the CV matches the job description on a scale from 0% to 100%.
3. Provide a clear and concise feedback explaining the score. The feedback should include matched skills, relevant experience, and missing skills.

Return a JSON object with the following keys:
- "name": string (full name or "Unknown" if not found)
- "email": string (email or "Unknown" if not found)
- "phone number": string (number or "Unknown" if not found)
- "location": string (location or "Unknown" if not found)
- "score": number (0-100)
- "feedback": list of strings (key points)

Job Description:
{job_description}

CV Text:
{cv_text}

Return only JSON, no extra text.
"""

prompt = PromptTemplate(
    input_variables=["job_description", "cv_text"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Extract text from CVs in PDF format
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(page for page in pages if page)

# Split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=100):
    # Using Langchain's text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# Function to extract email address if model not found
def extract_email(text):
    # Using regex
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Unknown"

# Streamlit App 
def main():
    st.title("AI CVs Scanner")

    st.markdown(
        """
        Paste your job description and upload all your candidate's CV. \n
        CVscanner's AI Agent analyzes and scores each candidate to speed up hiring.
        """
    )

    job_description = st.text_area("Paste your Job Description.")

    uploaded_files = st.file_uploader("Upload Candidate's Resume (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Score All CVs"):
        if not job_description.strip():
            st.warning("Please enter the job description.")
            return
        if not uploaded_files or len(uploaded_files) == 0:
            st.warning("Please upload at least one CV in PDF format.")
            return
        
        # Storing the results to build a dataframe later
        results = []

        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Extract CV text
                cv_text = extract_text_from_pdf(uploaded_file)
                # Chunk and combine text to avoid rate limit
                chunks = split_text(cv_text)
                combined_text = ""
                max_combined_chars = 2500
                for chunk in chunks:
                    if len(combined_text) + len(chunk) <= max_combined_chars:
                        combined_text += chunk + "\n"
                    else:
                        break

                 # Run model 
                try:
                    raw_result = chain.run(job_description=job_description, cv_text=combined_text)
                    parsed = json.loads(raw_result)
                    
                    # If email not found by model, try regex
                    if parsed.get("email", "Unknown").lower() == "unknown":
                        parsed["email"] = extract_email(cv_text)

                    # Ensure fields exist
                    name = parsed.get("name", "Unknown")
                    email = parsed.get("email", "Unknown")
                    location = parsed.get("location", "Unknown")
                    score = parsed.get("score", 0)
                    feedback = parsed.get("feedback", [])
                except Exception as e:
                    name, email, location, score = "Unknown", "Unknown", "Unknown", 0
                    feedback = [f"Error scoring resume: {str(e)}"]

                results.append({
                    "filename": uploaded_file.name,
                    "name": name,
                    "email": email,
                    "location": location,
                    "score": score,
                    "feedback": feedback
                })

        # Sort by descending score (shortlisted first)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Summary table data
        summary_data = []
        for res in results:
            summary_data.append({
                "Name": res["name"],
                "Email": res["email"],
                "Location": res["location"],
                "Score (%)": res["score"]
            })

        df_summary = pd.DataFrame(summary_data)

        st.subheader("Candidates Summary")
        st.dataframe(df_summary.style.format({"Score (%)": "{:.1f}"}))

        st.write("---")
        st.subheader("Detailed AI Feedback")

        # Feedback for each candidate
        for res in results:
            st.markdown(f"### {res['name']} ({res['filename']}) - Score: {res['score']}%")
            st.markdown(f"**Email:** {res['email']}")
            st.markdown(f"**Location:** {res['location']}")
            st.markdown("**Feedback:**")
            if isinstance(res["feedback"], list):
                for item in res["feedback"]:
                    st.markdown(f"- {item}")
            else:
                st.write(res["feedback"])
            st.write("---")

if __name__ == "__main__":
    main()