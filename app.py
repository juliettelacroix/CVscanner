import os
import streamlit as st
import pdfplumber
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    model="mistralai/mistral-7b-instruct",
)

# Prompt template
template = """You are an AI HR assistant. Given a job description and a CV, score how well the CV matches the job description on a scale from 0% to 100%.
Give justifications on the score by mentioning specific skill matches, experience relevance, and identify any missing, underrepresented or insufficiently detailed skills, certifications or experiences that are important for the role.

Job Description:
{job_description}

CV Text:
{cv_text}

Return the result as:

Score: <number between 0 and 100>
Feedback: <clean and structured list>
"""

prompt = PromptTemplate(
    input_variables=["job_description", "cv_text"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Extract text from CV in PDF format
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

# Streamlit App
def main():
    st.title("AI CV Scanner")

    st.markdown(
        """
        Upload a CV PDF file and paste the job description text below.
        CVscanner will score how well the resume matches the job description.
        """
    )

    job_description = st.text_area("Paste Job Description")

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
            st.warning("Please upload a CV in PDF format.")
            return

        with st.spinner("Scanning..."):
            chunks = split_text(cv_text)

            combined_text = ""
            max_combined_chars = 2500
            for chunk in chunks:
                if len(combined_text) + len(chunk) <= max_combined_chars:
                    combined_text += chunk + "\n"
                else:
                    break

            try:
                result = chain.run(job_description=job_description, cv_text=combined_text)
                st.subheader("CV Score & Feedback")
                st.text(result)
            except Exception as e:
                st.error(f"Error during scoring: {str(e)}")

if __name__ == "__main__":
    main()