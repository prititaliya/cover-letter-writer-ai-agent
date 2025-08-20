
import streamlit as st
import tempfile
import os

st.title("Cover Letter Assistant")

# Sidebar for PDF upload
pdf_file = st.sidebar.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Sidebar for job description URL
job_desc_url = st.sidebar.text_input("Paste the job description URL here")

# Button to generate cover letter
if st.sidebar.button("Generate Cover Letter"):
    if pdf_file is not None and job_desc_url:
        # Save uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            resume_path = tmp_file.name

        try:
            from app import app as backend_app
            init_state = {
                'resume_path': resume_path,
                'jd_url': job_desc_url,
                'iterations': 0,
                'message': []
            }
            final_state = backend_app.invoke(init_state)
            cover_letter = final_state.get('cover_letter', {})
            st.subheader("Generated Cover Letter")
            st.write(cover_letter.get('content', 'No cover letter generated.'))
            st.markdown("**Self-criticism:**")
            st.write(cover_letter.get('critism', 'No feedback available.'))
        except Exception as e:
            st.error(f"Error running backend logic: {e}")
        finally:
            os.remove(resume_path)
    else:
        st.warning("Please upload a PDF resume and paste the job description URL.")

st.markdown("""
### How to use:
1. Upload your resume in PDF format using the sidebar.
2. Paste the job description URL in the sidebar.
3. Click 'Generate Cover Letter' to get a tailored cover letter.
""")
