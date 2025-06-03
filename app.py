import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import PyPDF2
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def main():
    st.title("CV Scanner")

    st.markdown(
        """
        Upload a CV PDF file and paste the job description text below.
        CVscanner will score how well the resume matches the job description.
        """
    )

    job_description = st.text_area("Job Description")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    cv_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            cv_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted CV Text", cv_text, height=200)

    if st.button("Score CV"):
        if not job_description.strip():
            st.warning("Please enter the job description.")
            return
        if not cv_text.strip():
            st.warning("Please upload a CV (PDF).")
            return

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        prompt_template = """
You are an AI HR assistant. Given a job description and a CV, score how well the CV matches the job description on a scale from 0 to 100.
Give justifications on the score by mentioning specific skill matches, experience relevance, and gaps.

Job Description:
{job_description}

CV Text:
{cv_text}

Return the result as:

Score: <number between 0 and 100>
Feedback: <text>
"""
        prompt = PromptTemplate(
            input_variables=["job_description", "cv_text"],
            template=prompt_template
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        with st.spinner("Scoring..."):
            result = chain.run(job_description=job_description, cv_text=cv_text)

        st.subheader("Result")
        st.text(result)

if __name__ == "__main__":
    main()