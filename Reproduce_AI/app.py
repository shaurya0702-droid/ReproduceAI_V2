import tempfile
import streamlit as st

from main import run_pipeline
from features import chat_with_paper

# Page Configuration
st.set_page_config(page_title="ReproduceAI", page_icon="📄", layout="centered")
st.title("ReproduceAI")
st.caption("AI Research Paper Analysis Assistant")

# Session State
if "report" not in st.session_state:
    st.session_state.report = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload + Analyze
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
analyze = st.button("Analyze Paper", type="primary", disabled=uploaded_file is None)
if analyze and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    with st.spinner("Analyzing paper... this may take a minute."):
        try:
            report, retriever, llm = run_pipeline(pdf_path)
            st.session_state.report = report
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Your paper has been analyzed successfully.\n\nI can help you with the buttons below, or you can just ask me anything about the paper."
            }]
        except Exception as e:
            if "rate_limit" in str(e).lower():
                st.error("Groq rate limit reached. Please wait 5-10 seconds and try again.")
            else:
                st.error(f"Analysis failed: {e}")

# Chat Interface
if st.session_state.report:
    report = st.session_state.report
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def show_cached(user_msg, bot_msg):
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        st.rerun()

    st.write("")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📄 Overview", use_container_width=True):
            show_cached("Show me the paper overview", report["overview"])
    with col2:
        if st.button("📊 Metadata", use_container_width=True):
            metadata = ""
            for key, value in report["metadata"].items():
                metadata += f"**{key}**: {value}\n"
            show_cached("Show me the extracted metadata", metadata)
    with col3:
        if st.button("📝 Summary", use_container_width=True):
            show_cached("Summarize the paper", report["summary"])
    with col4:
        if st.button("🛠 Plan", use_container_width=True):
            show_cached("Give me the implementation plan", report["plan"])
    with col5:
        if st.button("⚠ Risk", use_container_width=True):
            show_cached("What are the reproduction risks?", report["risk_report"])

    st.caption(
        "Try asking: *Explain the architecture* · *Why did they choose this optimizer?* · *What are the limitations?* · *Explain Section 4*"
    )
    user_question = st.chat_input("Ask anything about this paper...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.spinner("Thinking..."):
            try:
                answer = chat_with_paper(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question,
                    paper_title=report["metadata"].get("title")
                )
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    answer = "Groq rate limit reached. Please wait a few seconds and try again."
                else:
                    answer = f"Something went wrong: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
else:
    st.info("Upload a research paper and click **Analyze Paper** to get started.")
