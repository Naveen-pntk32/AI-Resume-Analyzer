import streamlit as st
import PyPDF2
import io
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# Load OpenRouter settings
OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
APP_NAME = os.getenv("APP_NAME", "AI Resume Critiquer")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()

# Sidebar diagnostics
with st.sidebar:
    st.markdown("### Diagnostics")
    st.write(f"✅ API Key Loaded: {bool(OPENROUTER_API_KEY)}")
    st.write(f"🔑 Key Length: {len(OPENROUTER_API_KEY)}")
    st.write(f"🔑 Key Starts with sk-or-: {OPENROUTER_API_KEY.startswith('sk-or-')}")
    st.write(f"🌐 Site URL: {OPENROUTER_SITE_URL}")
    st.write(f"📌 Model: {OPENROUTER_MODEL}")

    run_connectivity_test = st.checkbox("Run OpenRouter connectivity test", value=False)
    if run_connectivity_test:
        if not OPENROUTER_API_KEY:
            st.error("❌ OPENROUTER_API_KEY missing; cannot run connectivity test.")
        else:
            try:
                resp = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": OPENROUTER_SITE_URL,
                        "X-Title": APP_NAME,
                    },
                    timeout=15,
                )
                st.write({"status_code": resp.status_code})
                try:
                    st.json(resp.json())
                except Exception:
                    st.text(resp.text)
            except Exception as diag_err:
                st.error(f"⚠️ Connectivity test failed: {diag_err}")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)")

analyze = st.button("Analyze Resume")

# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Helper: Extract text from file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

# Resume analysis logic
if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("❌ File does not have any content...")
            st.stop()

        prompt = f"""Please analyze this resume and provide constructive feedback. 
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}
        
        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""

        # Validate API key
        if not OPENROUTER_API_KEY:
            st.error("❌ OPENROUTER_API_KEY is not set. Please add it to your .env file.")
            st.stop()

        if not OPENROUTER_API_KEY.startswith("sk-or-"):
            st.error("❌ The OPENROUTER_API_KEY does not look correct. It should start with 'sk-or-'.")
            st.stop()

        # Initialize client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": APP_NAME,
            },
        )

        # Verify credentials with extra diagnostics
        try:
            _ = client.models.list()
        except Exception as auth_err:
            st.error("❌ Authentication failed with OpenRouter. Running detailed connectivity check...")
            try:
                resp = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": OPENROUTER_SITE_URL,
                        "X-Title": APP_NAME,
                    },
                    timeout=15,
                )
                st.write({"status_code": resp.status_code})
                try:
                    st.json(resp.json())
                except Exception:
                    st.text(resp.text)
            except Exception as deeper_err:
                st.error(f"⚠️ Detailed connectivity check failed: {deeper_err}")
            st.stop()

        # Request analysis
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            extra_body={"provider": {"allow_fallbacks": True}}
        )

        st.markdown("### Analysis Results")
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        err_text = str(e)
        if "401" in err_text or "Unauthorized" in err_text:
            st.error("❌ Authentication failed with OpenRouter (401). Please verify your API key and account status.")
        else:
            st.error(f"⚠️ An error occurred: {err_text}")
